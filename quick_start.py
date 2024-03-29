from __future__ import absolute_import, division, print_function, unicode_literals

from datacollct import load_image_train, load_image_test
from model import Generator, Discriminator
from model import generator_loss, discriminator_loss

import tensorflow as tf
import numpy as np
import time
import cv2
import os


PATH = os.path.join(os.getcwd(), 'data/')
WEIGHT_PATH = os.path.join(os.getcwd(), 'weights/')
IMAGE_PATH = os.path.join(os.getcwd(), 'generated_images/')
IS_LOAD = False 
LOAD_NUM = 200
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 200


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator()
discriminator = Discriminator()

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
# shuffling so that for every epoch a different image is generated
# to predict and display the progress of our model.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def train(dataset, epochs, weight_path = WEIGHT_PATH) :
    for epoch in range(epochs) :
        start = time.time()

        for input_image, target in dataset:
            train_step(input_image, target)

        # saving (checkpoint) the model every 10 epochs
        if (epoch + 1) % 10 == 0 :
            path_tmp = os.path.join(weight_path, str(epoch + 1))

            generator.save_weights(os.path.join(path_tmp, 'generator'))
            discriminator.save_weights(os.path.join(path_tmp, 'discriminator'))

            for inp, tar in test_dataset.take(1) :
                generate_images(inp, epoch + 1)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))    


def generate_images(inp, epoch_num) :
    prediction = generator(inp, training = False)
    img_tmp = np.array(prediction[0])
    img_tmp = cv2.cvtColor(img_tmp * 0.5 + 0.5, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(IMAGE_PATH, str(epoch_num)+".tiff"), img_tmp)
    cv2.imwrite(os.path.join(IMAGE_PATH, str(epoch_num)+"_inp.tiff"), inp[0].numpy())


def main() :
    if IS_LOAD :
        generator.load_weights(os.path.join(WEIGHT_PATH, str(LOAD_NUM) + '/generator'))
        discriminator.load_weights(os.path.join(WEIGHT_PATH, str(LOAD_NUM) + '/discriminator'))

    train(train_dataset, EPOCHS)

    print("Congratulations!!!")


if __name__ == '__main__' :
    main()
