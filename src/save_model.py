import os
import tensorflow as tf
from tensorflow import keras

from main import make_generator_model, make_discriminator_model

def main():
  generator = make_generator_model()
  discriminator = make_discriminator_model()
  generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
  discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

  checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
  )

  checkpoint.restore(tf.train.latest_checkpoint('./training_checkpoints'))
  generator.save('./saved_model/generator.h5')

if __name__ == '__main__':
  main()
