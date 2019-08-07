import tensorflow as tf
import numpy as np
import cv2
import os


IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def main() :
    input_path = 'crack_data/real_image'
    output_path = 'crack_data/input_image/canny_data'
    
    file_list = os.listdir(input_path)
    num_data = np.array(file_list).shape[0]
    
    for idx in range(num_data) :

        path_tmp = os.path.join(input_path, file_list[idx])
        
        image_tmp = cv2.imread(path_tmp, cv2.IMREAD_GRAYSCALE)
        image_tmp = cv2.Canny(image_tmp, 50, 200)
        image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_GRAY2RGB)

        path_tmp = os.path.join(output_path, file_list[idx])
        
        cv2.imwrite(path_tmp, image_tmp)

    
if __name__ == '__main__' :
    main()