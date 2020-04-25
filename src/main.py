import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio
import os
import PIL
import time

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 128
# BATCH_SIZE = 1
NOISE_DIM = 100
EPOCHS = 10000

# IMAGE_DIR = './data/celeba'
# IMAGE_DIR = '/media/HDD/celeba-hq/images/celeba-hq/celeba-64'
IMAGE_DIR = '/media/HDD/celeba/align'

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)  # None is the batch size

    model.add(layers.Conv2DTranspose(
        512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                            input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    #
    #model.add(layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same'))
    #model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def get_loss(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    generated_images = generator(noise, training=False)

    real_output = discriminator(images, training=False)
    fake_output = discriminator(generated_images, training=True)
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(np.int))
        plt.axis('off')
    
    plt.savefig('./generated_images/image_at_epoch{:04d}.png'.format(epoch))
    plt.close()

def train(dataset, epochs, steps):
    num_examples_to_generate = 16
    progbar = tf.keras.utils.Progbar(steps)

    seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    #generator_optimizer = tf.keras.optimizers.Adam(0.002, beta_1=0.5, beta_2=0.999)
    #discriminator_optimizer = tf.keras.optimizers.Adam(0.002, beta_1=0.5, beta_2=0.999)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    for epoch in range(epochs):
        start = time.time()
        for step in range(steps):
            progbar.update(step + 1)
            image_batch, _ = dataset.next()
            train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
        
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print("*" * 20)
        print("Gen Loss & Disc Loss")
        gen_loss, disc_loss = get_loss(image_batch, generator, discriminator)
        tf.print(gen_loss)
        tf.print(disc_loss)
        print("*" * 20)

    generate_and_save_images(generator, epochs, seed)


def preprocess_input(img):
    img = img - 127.5
    img = img / 127.5
    return img


def main():
    num_images = len(os.listdir(os.path.join(IMAGE_DIR, 'images')))
    steps = num_images // BATCH_SIZE

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        IMAGE_DIR,
        target_size=(64, 64),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    train(train_generator, EPOCHS, steps)

if __name__ == "__main__":
    main()
