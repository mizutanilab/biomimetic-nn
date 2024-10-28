#https://keras.io/examples/generative/ddim/

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_datasets as tfds

import keras
from keras import layers
from keras import ops

import mouselayers as mouse
import fid

import numpy as np
from numpy.random import seed
import random

num_epochs = int(os.environ.get('EPOCHS', default='1250'))
wnd2d = float(os.environ.get('WND2D', default='0.4'))
run_id = os.environ.get('RUN_ID', default='1')
rseed = int(os.environ.get('RNDSEED', default='25692'))
print('RUN_ID=', run_id)
print('RNDSEED=', rseed)
print('EPOCHS=', num_epochs)
print('WND2D=', wnd2d)
randomseed = rseed
seed(randomseed)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)
random.seed(a=randomseed, version=2)
os.environ["PYTHONHASHSEED"] = '0'

# data
dataset_repetitions = 5
#num_epochs = 1  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
#widths = [32, 64, 96, 128]
widths = [64, 128, 256, 512]
#widths = [128, 256, 256, 256, 512]
block_depth = 2

norm_groups = 8  # Number of groups used in GroupNormalization layer
#norm_groups = 16  # Number of groups used in GroupNormalization layer
attn_resolution = 16

windowrad_dn = [1.4, 1.4, wnd2d, wnd2d]
windowrad_up = [wnd2d, wnd2d, 1.4, 1.4]

# optimization
batch_size = 64
ema = 0.999
#learning_rate = 1e-3
learning_rate = 2e-4
#weight_decay = 1e-4

#def prepare_dataset(split):
def prepare_dataset(path):
    dataset = keras.utils.image_dataset_from_directory(
        path, label_mode=None, image_size=(64, 64), batch_size=None
    )
    dataset = dataset.map(lambda x: x / 255.0)
    return (
        #tfds.load(dataset_name, split=split, shuffle_files=True)
        #.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset.cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


# load dataset
train_dataset = prepare_dataset("/opt/dlami/nvme/afhq/train/cat")
#train_dataset = prepare_dataset("/opt/dlami/nvme/celeba_gan/img_align_celeba/train")
print(train_dataset)
val_dataset = prepare_dataset("/opt/dlami/nvme/afhq/val/cat")
#val_dataset = prepare_dataset("/opt/dlami/nvme/celeba_gan/img_align_celeba/val")
print(val_dataset)

def pos_neg_weight_stats(network):
    #network.summary()
    print('Type\t', 'nwpos/nw\t', 'nwneg/nw\t', 'nwpos\t', 'nwneg\t', 'nw\t', 'nzero\t', 'nbpos\t', 'nbneg\t', 'nb')
    eps = 0.0001
    swpos = 0
    swneg = 0
    sw = 0
    szero = 0
    sbpos = 0
    sbneg = 0
    sb = 0
    #for layer in model.generator.layers:
    for layer in network.layers:
      type = layer.__class__.__name__
      #print(type)
      if (type != 'Dense') and (type != 'mDense') and (type != 'Conv2DTranspose') and (type != 'mConv2DTranspose') and (type != 'SpectralNormalization') and (type != 'Conv2D') and (type != 'mConv2D'):
          continue
      w = layer.get_weights()[0]
      b = layer.get_weights()[1]
      nwpos = np.count_nonzero(w > eps)
      nwneg = np.count_nonzero(w < -eps)
      nw = np.size(w)
      nzero = 0
      if (type == 'mDense') or (type == 'mConv2DTranspose') or (type == 'mConv2D'):
        nzero = layer.get_num_zeros()
      nbpos = np.count_nonzero(b > eps)
      nbneg = np.count_nonzero(b < -eps)
      nb = np.size(b)
      print(type, '\t', '{:.5f}'.format(nwpos/nw), '\t', '{:.5f}'.format(nwneg/nw), '\t', nwpos, '\t', nwneg, '\t', nw, '\t', nzero, '\t', nbpos, '\t', nbneg, '\t', nb)
      swpos += nwpos
      swneg += nwneg
      sw += nw
      szero += nzero
      sbpos += nbpos
      sbneg += nbneg
      sb += nb
    #for
    print('Total\t', '{:.5f}'.format(swpos/sw), '\t', '{:.5f}'.format(swneg/sw), '\t', swpos, '\t', swneg, '\t', sw, '\t', szero, '\t', sbpos, '\t', sbneg, '\t', sb)
    return

@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = ops.exp(
        ops.linspace(
            ops.log(embedding_min_frequency),
            ops.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = ops.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = ops.concatenate(
        [ops.sin(angular_speeds * x), ops.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )

class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        #self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        #self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        #self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        #self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))
        self.query = layers.Conv2D(units, kernel_size=1, strides=1, padding="valid")
        self.key = layers.Conv2D(units, kernel_size=1, strides=1, padding="valid")
        self.value = layers.Conv2D(units, kernel_size=1, strides=1, padding="valid")
        self.proj = layers.Conv2D(units, kernel_size=1, strides=1, padding="valid")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

def ResidualBlock(width, wndrad=1.4):
    def apply(x):
        #width2d = round(math.sqrt(width))
        width2d = pow(2,round(math.log2(width)/2))
        #128=>16, 256=>16
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
            #residual = mouse.mConv2D(width, kernel_size=1, form='2d', input2d_width=width2d, output2d_width=width2d, window2d_width=wnd2d)(x)
        #x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.GroupNormalization(groups=norm_groups)(x)
        #x = layers.Activation('swish')(x)
        if (wndrad < 1.4):
            x = mouse.mConv2D(width, kernel_size=3, form='2d', input2d_width=width2d, output2d_width=width2d, window2d_width=wndrad, padding="same")(x)
            x = layers.GroupNormalization(groups=norm_groups)(x)
            x = layers.Activation('swish')(x)
            x = layers.Dropout(0.1)(x)
            x = mouse.mConv2D(width, kernel_size=3, form='2d', input2d_width=width2d, output2d_width=width2d, window2d_width=wndrad, padding="same")(x)
        else:
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
            x = layers.GroupNormalization(groups=norm_groups)(x)
            x = layers.Activation('swish')(x)
            x = layers.Dropout(0.1)(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth, attn_resolution, wndrad):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, wndrad)(x)
            if (x.shape[1] == attn_resolution):
                x = AttentionBlock(width, groups=norm_groups)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, attn_resolution, wndrad):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, wndrad)(x)
            if (x.shape[1] == attn_resolution):
                x = AttentionBlock(width, groups=norm_groups)(x)
        return x

    return apply

def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    i=0
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, attn_resolution, windowrad_dn[i])([x, skips])
        i+=1

    for i in range(block_depth):
        x = ResidualBlock(widths[-1], windowrad_dn[-1])(x)
        if (x.shape[1] == attn_resolution) and (i != block_depth-1):
            x = AttentionBlock(widths[-1], groups=norm_groups)(x)

    i=0
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, attn_resolution, windowrad_up[i])([x, skips])
        i+=1

    #x = layers.GroupNormalization(groups=norm_groups)(x)
    #x = layers.Activation('swish')(x)
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


@keras.saving.register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        #self.ema_network = keras.models.clone_model(self.network)
        self.ema_network = get_network(image_size, widths, block_depth)
        self.ema_network.set_weights(self.network.get_weights())
        self.seed_generator = keras.random.SeedGenerator(randomseed)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        #self.kid = KID(name="kid")

    @property
    def metrics(self):
        #return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return ops.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = ops.cast(ops.arccos(max_signal_rate), "float32")
        end_angle = ops.cast(ops.arccos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = ops.cos(diffusion_angles)
        noise_rates = ops.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = ops.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = keras.random.normal(
            shape=(num_images, image_size, image_size, 3),
            seed=self.seed_generator
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3), seed=self.seed_generator)

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0, seed=self.seed_generator
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        #return {m.name: m.result() for m in self.metrics[:-1]}
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = keras.random.normal(shape=(batch_size, image_size, image_size, 3), seed=self.seed_generator)

        # sample uniform random diffusion times
        diffusion_times = keras.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0, seed=self.seed_generator
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=10, num_cols=10):
        #print("epoch", epoch)
        diffsteps = plot_diffusion_steps

        if (epoch != num_epochs-1):
            if ((epoch+1) % (num_epochs/10)): return
            return

        if (epoch == num_epochs-1):
            pos_neg_weight_stats(self.network)

        #print('Ep=', epoch, 'checkpoint')
        #checkpoint.save(file_prefix = checkpoint_prefix)

        #print('Ep=', epoch, 'images')
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=diffsteps,
        )
        generated_images *= 255
        generated_images.numpy()
        fig = plt.figure(figsize=(num_cols*2, num_rows*2))
        for i in range(generated_images.shape[0]):
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(keras.utils.array_to_img(generated_images[i]))
            plt.axis('off')
        fig.savefig('image_{0}{1:04d}.png'.format(run_id, epoch))
        plt.close()

        print('Ep=', epoch, 'FID')
        ###FID
        fid_batch_size = 64
        fid_num_batches = 800
        cardinality = tf.data.experimental.cardinality(train_dataset).numpy()

        fd = fid.FrechetInceptionDistance(None, (0,1)) 

        real_images = []
        icount = 0
        gan_fid = -1
        itotal = 0
        while itotal < fid_num_batches:
            icard = 0
            #for x in val_dataset:
            for x in train_dataset:
                #real images
                icard += 1
                real_images.extend(x.numpy())
                icount += (x.numpy()).shape[0]
                if (icount < fid_batch_size):
                    if (icard >= cardinality-1):
                        break
                    continue
                real_images = np.array(tf.image.resize(real_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                #gen images
                generated_images = self.generate(
                    num_images=icount,
                    diffusion_steps=diffsteps,
                )
                generated_images = np.array(tf.image.resize(generated_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                #FID accum
                itotal += 1
                fd(real_images , generated_images, batch_size=fid_batch_size, num_batches_real=1, num_batches_gen=1)
                real_images = []
                icount = 0
                #fd.reset(None)
                if (itotal >= fid_num_batches):
                    break;
                if (icard >= cardinality-1):
                    break
            #print ('icard=', icard, '/', cardinality, 'itotal=', itotal)
        gan_fid = fid.frechet_distance(fd.real_mean, fd.real_cov, fd.gen_mean, fd.gen_cov)
        print('RUN_ID=', run_id, 'Ep=', epoch, ' FID=', gan_fid, ' N=', itotal * fid_batch_size)

# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)
model.compile(
    #optimizer=keras.optimizers.AdamW(
    #    learning_rate=learning_rate, weight_decay=weight_decay
    #),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9),
    #loss=keras.losses.mean_absolute_error,
    loss=keras.losses.mean_squared_error,
)
model.network.summary()
# pixelwise mean absolute error is used as loss

checkpoint_dir = './ckpt' + run_id
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

output_mode = 2
if (num_epochs == 1):
    output_mode = 1
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

#checkpoint.restore('../dif00/ckpt-9')

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],
    verbose=output_mode
)

checkpoint.save(file_prefix = checkpoint_prefix)


