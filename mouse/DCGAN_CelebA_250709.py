#https://keras.io/examples/generative/dcgan_overriding_train_step/

import keras
import tensorflow as tf

from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile

import mouselayers as mouse
import fid

import numpy as np
from numpy.random import seed
import random

os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#l1reg = float(os.environ.get('L1REG', default='0.0'))
epochs = int(os.environ.get('EPOCHS', default='25'))
ndim = int(os.environ.get('NDIM', default='64'))
wnd2d = float(os.environ.get('WND2D', default='0.4'))
run_id = os.environ.get('RUN_ID', default='1')
rseed = int(os.environ.get('RNDSEED', default='25692'))
ndrepeat = 1
#l1reg = 0.00
#wnd2d=1.4
print('RUN_ID=', run_id)
print('RNDSEED=', rseed)
print('EPOCHS=', epochs)
#print('L1REG=', l1reg)
print('NDIM=', ndim)
print('WND2D=', wnd2d)
randomseed = rseed
seed(randomseed)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)
random.seed(a=randomseed, version=2)
os.environ["PYTHONHASHSEED"] = '0'

#####download CelebA
os.makedirs("celeba_gan")
url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
gdown.download(url, output, quiet=True)
with ZipFile("celeba_gan/data.zip", "r") as zipobj:
    zipobj.extractall("celeba_gan")
#####
#scp -i "****.pem" downloads\525_bird_species.zip ubuntu@ec***.compute-1.amazonaws.com:/opt/dlami/nvme
#mkdir /opt/dlami/nvme/525birds
#cd /opt/dlami/nvme/525birds
#mv ../525_bird_species.zip .
#unzip *zip

dataset = keras.utils.image_dataset_from_directory(
    #"/opt/dlami/nvme/afhq/train/cat", label_mode=None, image_size=(64, 64), batch_size=32
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)

#show image
#for x in dataset:
#    plt.axis("off")
#    plt.imshow((x.numpy() * 255).astype("int32")[0])
#    break

class NegWeightReg(keras.regularizers.Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2
    def __call__(self, x):
        x = tf.math.minimum(tf.zeros_like(x), x)
        #x = tf.math.maximum(tf.zeros_like(x), x)
        return self.l2 * ops.sum(ops.square(x)) + self.l1 * ops.sum(ops.abs(x))

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.SpectralNormalization(layers.Conv2D(64, kernel_size=4, strides=2, padding="same")),
        #layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        layers.SpectralNormalization(layers.Conv2D(128, kernel_size=4, strides=2, padding="same")),
        layers.LeakyReLU(negative_slope=0.2),

        layers.SpectralNormalization(layers.Conv2D(256, kernel_size=4, strides=2, padding="same")),
        layers.LeakyReLU(negative_slope=0.2),

        layers.SpectralNormalization(layers.Conv2D(512, kernel_size=4, strides=2, padding="same")),
        layers.LeakyReLU(negative_slope=0.2),

        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()

latent_dim = 100
#reg = l1reg
dim = ndim
wnd = wnd2d

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        
        layers.Dense(4 * 4 * 1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Reshape((4, 4, 1024)),
        mouse.mConv2DTranspose(512, form='2d', input2d_width=32, output2d_width=32, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        mouse.mConv2DTranspose(256, form='2d', input2d_width=32, output2d_width=16, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        mouse.mConv2DTranspose(128, form='2d', input2d_width=16, output2d_width=16, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(randomseed)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = ops.shape(real_images)[0]

        for _ in range(ndrepeat):
            random_latent_vectors = keras.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator
            )

            # Decode them to fake images
            generated_images = self.generator(random_latent_vectors)

            # Combine them with real images
            combined_images = ops.concatenate([generated_images, real_images], axis=0)

            # Assemble labels discriminating real from fake images
            labels = ops.concatenate(
                [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
            )
            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            #looping here is rather inappropriate but it works and memory friendly.
            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.d_loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
        #for

        # Sample random points in the latent space
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Assemble labels that say "all real images"
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "d_lr": self.d_optimizer.learning_rate,
            "g_lr": self.g_optimizer.learning_rate,
        }

checkpoint_dir = './ckpt' + run_id
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

output_mode = 2
if (epochs == 1):
      output_mode = 1
#output_mode = 0: no console output (minimum output)
#output_mode = 1: show progress bar (jupyter notebook)
#output_mode = 2: one line per epoch (shell script)

def pos_neg_weight_stats(model):
    #print('Negative / positive weights and biases.')
    print('Type\t', 'nwpos/nw\t', 'nwneg/nw\t', 'nwpos\t', 'nwneg\t', 'nw\t', 'nzero\t', 'nbpos\t', 'nbneg\t', 'nb')
    eps = 0.0001
    swpos = 0
    swneg = 0
    sw = 0
    szero = 0
    sbpos = 0
    sbneg = 0
    sb = 0
    for layer in model.generator.layers:
      type = layer.__class__.__name__
      #print(type)
      if (type != 'Dense') and (type != 'mDense') and (type != 'Conv2DTranspose') and (type != 'mConv2DTranspose'):
          continue
      w = layer.get_weights()[0]
      b = layer.get_weights()[1]
      nwpos = np.count_nonzero(w > eps)
      nwneg = np.count_nonzero(w < -eps)
      nw = np.size(w)
      nzero = 0
      if (type == 'mDense') or (type == 'mConv2DTranspose'):
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


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(randomseed)

    def on_epoch_end(self, epoch, logs=None):
        #pos_neg_weight_stats(self.model)
        if (epoch != epochs-1):
            #if ((epoch+1) % 10): return
            if ((epoch+1) % (epochs/5)): return
            return

        pos_neg_weight_stats(self.model)

        #print('Ep=', epoch, 'checkpoint')
        #checkpoint.save(file_prefix = checkpoint_prefix)

        print('Ep=', epoch, 'images')
        random_latent_vectors = keras.random.normal(
            shape=(self.num_img * self.num_img, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        fig = plt.figure(figsize=(self.num_img, self.num_img))
        for i in range(generated_images.shape[0]):
            plt.subplot(self.num_img, self.num_img, i+1)
            plt.imshow(keras.utils.array_to_img(generated_images[i]))
            plt.axis('off')
        fig.savefig('image_{0}{1:03d}.png'.format(run_id, epoch))
        plt.close()

        print('Ep=', epoch, 'FID')
        ###FID
        fid_batch_size = 64
        fid_num_batches = 800
        cardinality = tf.data.experimental.cardinality(dataset).numpy()
        fd = fid.FrechetInceptionDistance(generator, (0,1)) 

        train_images = []
        icount = 0
        gan_fid = -1
        itotal = 0
        #for _ in range(fid_repeat):
        while itotal < fid_num_batches:
            icard = 0
            for x in dataset:
                #training images
                icard += 1
                train_images.extend(x.numpy())
                icount += (x.numpy()).shape[0]
                if (icount < fid_batch_size):
                    if (icard >= cardinality-1):
                        break
                    continue
                train_images = np.array(tf.image.resize(train_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                #gen images
                random_latent_vectors = keras.random.normal(shape=(icount, latent_dim), seed=self.seed_generator)
                generated_images = self.model.generator(random_latent_vectors)
                generated_images = np.array(tf.image.resize(generated_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                #FID accum
                itotal += 1
                #ibatch += 1
                fd(train_images , generated_images, batch_size=fid_batch_size, num_batches_real=1, num_batches_gen=1)
                train_images = []
                icount = 0
                #fd.reset(None)
                if (itotal >= fid_num_batches):
                    break;
                if (icard >= cardinality-1):
                    break
            #print ('icard=', icard, '/', cardinality, 'itotal=', itotal)
        gan_fid = fid.frechet_distance(fd.real_mean, fd.real_cov, fd.gen_mean, fd.gen_cov)
        print('Ep=', epoch, ' FID=', gan_fid, ' N=', itotal * fid_batch_size)


steps_per_epoch = ndrepeat * tf.data.experimental.cardinality(dataset).numpy()
d_boundaries = [steps_per_epoch * 5]
d_values = [0.000001, 0.00001]
d_learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(d_boundaries, d_values)

class D_LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_loss_cutoff):
        self.d_loss_cutoff = d_loss_cutoff
    def __call__(self, step):
        d_loss = gan.d_loss_metric.result()
        #bool1 = (step > steps_per_epoch * 5)
        sigmoid2 = 1.0 / (1.0 + tf.math.exp(-10.0 * (d_loss - self.d_loss_cutoff)))
        #d_lr = tf.cast(bool1, tf.float32) * sigmoid2 * 0.000009 + 0.000001
        d_lr = 0.0001 * (sigmoid2 * 0.9 + 0.1)

        return d_lr

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    #d_optimizer=keras.optimizers.Adam(learning_rate=D_LRSchedule(0.3), beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
)

#checkpoint.restore('../gan2c/training_checkpoints/ckpt-9')
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)], verbose=output_mode
)

checkpoint.save(file_prefix = checkpoint_prefix)



