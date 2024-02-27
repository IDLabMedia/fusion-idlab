import tensorflow as tf
import os
from os.path import exists
import numpy as np
import cv2
from SpectralNormalization import SpectralNormalization
import random
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

# Set random seeds
random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)

INPUT_CHANNELS = 11
OUTPUT_CHANNELS = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Inspired by https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb
def downsample(filters, size, apply_batchnorm=True, is_discriminator=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    if is_discriminator:
        result.add(
            SpectralNormalization(
                tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False)
            )
        )
    else:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False)
        )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

    tar_duplicate = tf.keras.layers.concatenate([tar for _ in range(INPUT_CHANNELS)])

    x = tf.keras.layers.concatenate([inp, tar_duplicate])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, apply_batchnorm=False, is_discriminator=True)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4, is_discriminator=True)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4, is_discriminator=True)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv1 = SpectralNormalization(tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False))(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = SpectralNormalization(tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer))(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def normalize_for_model(heatmap):
    return (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) if np.max(heatmap) - np.min(heatmap) > 1e-1 else 1) * 2 - 1

def read_image_from_npz_file(output_file_path):
    image = np.load("%s.npz" % (output_file_path))['image']
    return image 

def load_heatmap_image(filename, width, height):
    try:    
        heatmap = read_image_from_npz_file(filename)
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = np.expand_dims(heatmap, axis=-1)
        return heatmap
    except Exception as e:
        print("Exception for %s, use zeros instead." % filename)
        print(e)
        heatmap = np.zeros((width, height))
        heatmap = np.expand_dims(heatmap, axis=-1)
        return heatmap
    
def load_combined_heatmaps(cat_net, comprint_noiseprint, adq1, blk, dct, cagi, comprint, noiseprint, original_image):
    heatmap1 = load_heatmap_image(cat_net, IMG_WIDTH, IMG_HEIGHT)
    heatmap2 = load_heatmap_image(comprint_noiseprint, IMG_WIDTH, IMG_HEIGHT)
    heatmap3 = load_heatmap_image(adq1, IMG_WIDTH, IMG_HEIGHT)
    heatmap4 = load_heatmap_image(blk, IMG_WIDTH, IMG_HEIGHT)
    heatmap5 = load_heatmap_image(dct, IMG_WIDTH, IMG_HEIGHT)
    heatmap6 = load_heatmap_image(cagi, IMG_WIDTH, IMG_HEIGHT)
    heatmap7 = load_heatmap_image(comprint, IMG_WIDTH, IMG_HEIGHT)
    heatmap8 = load_heatmap_image(noiseprint, IMG_WIDTH, IMG_HEIGHT)
    normalized_heatmaps = [normalize_for_model(heatmap1), normalize_for_model(heatmap2), normalize_for_model(heatmap3), normalize_for_model(heatmap4),
                           normalize_for_model(heatmap5), normalize_for_model(heatmap6), normalize_for_model(heatmap7), normalize_for_model(heatmap8)]

    original = cv2.imread(original_image)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized_original = cv2.resize(original, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    normalized_original = normalize_for_model(resized_original)
    normalized_heatmaps.append(normalized_original)
    
    combined_heatmaps = np.concatenate(normalized_heatmaps, axis=-1)
    
    return combined_heatmaps

# The images are paths to the npz files
def get_prediction(checkpoint_dir, cat_net_image, comprint_noiseprint_image, adq1_image, blk_image, dct_image, cagi_image, comprint_image, noiseprint_image, original_image):
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    
    combined_heatmaps_and_original = load_combined_heatmaps(cat_net_image, comprint_noiseprint_image, adq1_image, blk_image, dct_image, cagi_image, comprint_image, noiseprint_image, original_image)
    combined_heatmaps_and_original = combined_heatmaps_and_original[tf.newaxis, ...] # For batch dimension

    # Training=True is very important, otherwise it doesn't work!
    prediction = generator(combined_heatmaps_and_original, training=True) 
    prediction = prediction.numpy()
    prediction = prediction[0][:,:,0]

    return prediction