#https://keras.io/examples/generative/dcgan_overriding_train_step/
#https://keras.io/examples/generative/conditional_gan/

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
epochs = int(os.environ.get('EPOCHS', default='125'))
wnd2d = float(os.environ.get('WND2D', default='0.4'))
run_id = os.environ.get('RUN_ID', default='1')
rseed = int(os.environ.get('RNDSEED', default='25692'))
print('RUN_ID=', run_id)
print('RNDSEED=', rseed)
print('EPOCHS=', epochs)
print('WND2D=', wnd2d)
randomseed = rseed
seed(randomseed)
tf.random.set_seed(randomseed)
np.random.seed(randomseed)
random.seed(a=randomseed, version=2)
os.environ["PYTHONHASHSEED"] = '0'
batch_size = 32
num_color_channels = 3
image_size = 64

def prepare_dataset(path):
    dataset = keras.utils.image_dataset_from_directory(
        path, label_mode=None, image_size=(image_size, image_size), batch_size=None
    )
    return dataset.map(lambda x: x / 255.0)

filepathlist = ["/opt/dlami/nvme/celeba_gan/img_align_celeba",
            "/opt/dlami/nvme/afhq/train/cat",  
            "/opt/dlami/nvme/525birds", 
            "/opt/dlami/nvme/CheesePicsSelec"]

num_per_set = 5000
dataset = None
num_classes = len(filepathlist)
#org_images = [0] * num_classes
images_shard = [0] * num_classes

idx = 0
for filepath in filepathlist:
    org_images = prepare_dataset(filepath)
    images_shard[idx] = org_images.shard(num_shards=org_images.__len__().numpy() // num_per_set, index=0).cache()
    labels = [idx] * images_shard[idx].__len__().numpy()
    labels = keras.utils.to_categorical(labels, num_classes)
    labels = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    if (idx == 0): 
        dataset = tf.data.Dataset.zip((images_shard[idx], labels))
    else: 
        dataset = dataset.concatenate(tf.data.Dataset.zip((images_shard[idx], labels)))
    print(dataset.__len__())
    idx += 1

dataset_s = dataset.shuffle(buffer_size=dataset.__len__().numpy()).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

latent_dim = 100
generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_color_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, discriminator_in_channels)),
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

wnd = wnd2d

generator = keras.Sequential(
    [
        keras.Input(shape=(generator_in_channels,)),
        
        layers.Dense(4 * 4 * 1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Reshape((4, 4, 1024)),
        mouse.mConv2DTranspose(512, form='2d', input2d_width=32, output2d_width=32, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        #layers.Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        mouse.mConv2DTranspose(256, form='2d', input2d_width=32, output2d_width=16, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        #layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        mouse.mConv2DTranspose(128, form='2d', input2d_width=16, output2d_width=16, window2d_width=wnd, kernel_size=5, strides=2, padding="same"),
        #layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"),
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

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = ops.repeat(
            #image_one_hot_labels, repeats=[image_size * image_size]
            image_one_hot_labels, repeats=[image_size * image_size], axis=0
        )
        image_one_hot_labels = ops.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space
        batch_size = ops.shape(real_images)[0]

        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = ops.concatenate(
            [generated_images, image_one_hot_labels], -1
        )
        real_image_and_labels = ops.concatenate([real_images, image_one_hot_labels], -1)
        combined_images = ops.concatenate(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.d_loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = ops.concatenate(
                [fake_images, image_one_hot_labels], -1
            )
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
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
    #for layer in model.discriminator.layers:
    for layer in model.generator.layers:
      type = layer.__class__.__name__
      #print(type)
      if (type != 'Dense') and (type != 'mDense') and (type != 'Conv2DTranspose') and (type != 'mConv2DTranspose') and (type != 'SpectralNormalization'):
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
            return
            #if ((epoch+1) % 10): return
            if ((epoch+1) % (epochs/5)): return

        pos_neg_weight_stats(self.model)

        #print('Ep=', epoch, 'checkpoint')
        #checkpoint.save(file_prefix = checkpoint_prefix)

        print('Ep=', epoch, 'images')
        random_latent_vectors = keras.random.normal(
            shape=(self.num_img * self.num_img, self.latent_dim), seed=self.seed_generator
        )
        for imgclass in range(num_classes):
            labels = [imgclass] * self.num_img * self.num_img
            one_hot_labels = keras.utils.to_categorical(labels, num_classes)
            random_vector_labels = ops.concatenate(
                [random_latent_vectors, one_hot_labels], axis=1
            )
            generated_images = self.model.generator(random_vector_labels)
            generated_images *= 255
            generated_images.numpy()
            fig = plt.figure(figsize=(self.num_img*2, self.num_img*2))
            for i in range(generated_images.shape[0]):
                plt.subplot(self.num_img, self.num_img, i+1)
                plt.imshow(keras.utils.array_to_img(generated_images[i]))
                plt.axis('off')
            fig.savefig('image_{1}{2:03d}_{0}.png'.format(imgclass, run_id, epoch))
            plt.close()
        #return

        print('Ep=', epoch, 'FID')
        ###FID
        fid_batch_size = 64
        fid_num_batches = 800
        #fid_num_batches = 80
        fid_repeat = 11

        for imgclass in range(num_classes):
            fd = fid.FrechetInceptionDistance(generator, (0,1)) 
            train_images = []
            icount = 0
            gan_fid = -1
            #itotal = 0
            org_images_batch = images_shard[imgclass].batch(batch_size, drop_remainder=True)
            ibatch = 0
            for _ in range(fid_repeat):
                #ibatch = 0
                for x in org_images_batch:
                    #training images
                    train_images.extend(x.numpy())
                    icount += (x.numpy()).shape[0]
                    if (icount < fid_batch_size):
                        continue
                    train_images = np.array(tf.image.resize(train_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                    #gen images
                    random_latent_vectors = keras.random.normal(shape=(icount, latent_dim), seed=self.seed_generator)
                    labels = [imgclass] * icount
                    one_hot_labels = keras.utils.to_categorical(labels, num_classes)
                    random_vector_labels = ops.concatenate(
                        [random_latent_vectors, one_hot_labels], axis=1
                    )
                    generated_images = self.model.generator(random_vector_labels)
                    generated_images = np.array(tf.image.resize(generated_images, [299, 299], method=tf.image.ResizeMethod.BILINEAR))
                    #FID accum
                    #itotal += 1
                    ibatch += 1
                    fd(train_images , generated_images, batch_size=fid_batch_size, num_batches_real=1, num_batches_gen=1)
                    train_images = []
                    icount = 0
                    #fd.reset(None)
                    if (ibatch >= fid_num_batches): break
                if (ibatch >= fid_num_batches): break
            gan_fid = fid.frechet_distance(fd.real_mean, fd.real_cov, fd.gen_mean, fd.gen_cov)
            print('RUN_ID=', run_id, 'Class=', imgclass, 'Ep=', epoch, 'FID=', gan_fid, 'N=', ibatch * fid_batch_size)
        #for imgclass

#steps_per_epoch = ndrepeat * tf.data.experimental.cardinality(dataset).numpy()

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5),
    d_loss_fn=keras.losses.BinaryCrossentropy(),
    g_loss_fn=keras.losses.BinaryCrossentropy(),
)

#checkpoint.restore('../gan2c/training_checkpoints/ckpt-9')
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

gan.fit(
    dataset_s, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)], verbose=output_mode
)

checkpoint.save(file_prefix = checkpoint_prefix)


