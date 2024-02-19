import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import h5py
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import random
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D, ReLU, concatenate, MaxPooling2D, GlobalMaxPooling2D, Dropout
import tensorflow.keras.backend as K

def create_dataframe(folder_path, label):
    file_list = os.listdir(folder_path)
    file_names = [os.path.join(os.path.basename(folder_path), file) for file in file_list]
    df = pd.DataFrame({'File': file_names, 'Label': label})
    return df

def print_split(train_reference_df, val_reference_df, test_reference_df):
    # Count the elements in the sets
    num_train_data_normal = sum(train_reference_df['Label'] == 0)
    num_train_data_covid   = sum(train_reference_df['Label'] == 1)
    num_train_data_pneumonia   = sum(train_reference_df['Label'] == 2)
    num_val_data_normal   = sum(val_reference_df['Label'] == 0)
    num_val_data_covid     = sum(val_reference_df['Label'] == 1)
    num_val_data_pneumonia   = sum(val_reference_df['Label'] == 2)
    num_test_data_normal   = sum(test_reference_df['Label'] == 0)
    num_test_data_covid     = sum(test_reference_df['Label'] == 1)
    num_test_data_pneumonia   = sum(test_reference_df['Label'] == 2)
    
    print('TRAIN SET')
    print('\tNormal X-ray: {} ({:.2f}%)'.format(num_train_data_normal, 100 * num_train_data_normal / len(train_reference_df)))
    print('\tCOVID-19 X-ray: {} ({:.2f}%)'.format(num_train_data_covid, 100 * num_train_data_covid / len(train_reference_df)))
    print('\tPneumonia X-ray: {} ({:.2f}%)'.format(num_train_data_pneumonia, 100 * num_train_data_pneumonia / len(train_reference_df)))
    print('VALIDATION SET')
    print('\tNormal ECG: {} ({:.2f}%)'.format(num_val_data_normal, 100 * num_val_data_normal / len(val_reference_df)))
    print('\tCOVID-19 X-ray: {} ({:.2f}%)'.format(num_val_data_covid, 100 * num_val_data_covid / len(val_reference_df)))
    print('\tPneumonia X-ray: {} ({:.2f}%)'.format(num_val_data_pneumonia, 100 * num_val_data_pneumonia / len(val_reference_df)))
    print('TEST SET')
    print('\tNormal ECG: {} ({:.2f}%)'.format(num_test_data_normal, 100 * num_test_data_normal / len(test_reference_df)))
    print('\tCOVID-19 X-ray: {} ({:.2f}%)'.format(num_test_data_covid, 100 * num_test_data_covid / len(test_reference_df)))
    print('\tPneumonia X-ray: {} ({:.2f}%)'.format(num_test_data_pneumonia, 100 * num_test_data_pneumonia / len(test_reference_df)))

def load_image(file_name, data_dir):
    if isinstance(data_dir, bytes):
        data_dir = data_dir.decode()
    if isinstance(file_name, bytes):
        file_name = file_name.decode()

    # Load the image
    file_path = os.path.join(data_dir, file_name)
    image = Image.open(file_path)

    # Convert RGBA to L (grayscale)
    image = image.convert('L')

    return image

def center_crop_and_resize(image, target_size=(224, 224), center_crop=True):

    if center_crop:
        # Get the original size of the image
        original_width, original_height = image.size
    
        # Calculate the crop box coordinates to achieve an even aspect ratio
        aspect_ratio = target_size[0] / target_size[1]
        crop_width = min(original_width, int(original_height * aspect_ratio))
        crop_left = max(0, (original_width - crop_width) // 2)
        crop_right = min(original_width, crop_left + crop_width)
    
        # Perform the center crop
        image = image.crop((crop_left, 0, crop_right, original_height))

    # Resize the cropped image to the target size
    processed_image = image.resize(target_size)

    return processed_image

def normalize_image(image, new_range=(0,1)):
    # Convert PIL image to NumPy array
    img_array = np.array(image)

    # Scale pixel values to the new range
    new_min, new_max = new_range
    scaled_array = (img_array / 255) * (new_max - new_min) + new_min

    # Clip values to ensure they are within the specified range
    scaled_array = np.clip(scaled_array, new_range[0], new_range[1])

    return scaled_array

def load_and_preprocess_data(file_name, data_dir):
    # Load data
    data = load_image(file_name, data_dir)
    # Baseline wander removal
    data = center_crop_and_resize(data, target_size=(256,256))
    # Normalize
    data = normalize_image(data, new_range=(0,1))
    return data.astype(np.float32)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y].T
    return Y

import numpy as np

def generate_subpatches(image, patch_size=(64, 64), overlap=(0, 0)):

    # Calculate overlap in pixels
    overlap_pixels = (int(patch_size[0] * overlap[0]), int(patch_size[1] * overlap[1]))

    # Calculate step size based on patch size and overlap
    step = (patch_size[0] - overlap_pixels[0], patch_size[1] - overlap_pixels[1])

    subpatches_list = []

    # Iterate over rows and columns of the image
    for i in range(0, image.shape[0] - patch_size[0] + 1, step[0]):
        for j in range(0, image.shape[1] - patch_size[1] + 1, step[1]):
            # Extract subpatch
            subpatch = image[i:i+patch_size[0], j:j+patch_size[1], ...]
            subpatches_list.append(subpatch)

    subpatches = np.array(subpatches_list)

    return subpatches

import matplotlib.pyplot as plt

def plot_subpatches(subpatches):
    
    num_subpatches = len(subpatches)

    num_rows = int(np.ceil(np.sqrt(num_subpatches)))
    num_cols = int(np.ceil(num_subpatches / num_rows))

    for i in range(num_subpatches):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(subpatches[i], cmap='gray')
        plt.axis('off')
        plt.title(f'Subpatch {i+1}')

    plt.show()

def build_deep_autoencoder_2(img_shape, code_size):

    # encoder
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.Input(img_shape))

    encoder.add(layers.Conv2D(32, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))
    
    encoder.add(layers.Conv2D(64, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Conv2D(256, (3, 3), activation='elu', padding='same'))
    encoder.add(layers.MaxPool2D((2, 2), padding='same'))

    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(code_size))

    # decoder
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.Input((code_size,)))

    decoder.add(layers.Dense(4 * 4 * 256, activation='elu'))
    decoder.add(layers.Reshape((4, 4, 256)))
    decoder.add(layers.Conv2DTranspose(128, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(layers.Conv2DTranspose(1, (3, 3), strides=2, activation=None, padding='same'))

    return encoder, decoder

def show_image(x):
    plt.imshow(x, cmap='gray')

def visualize(img, encoder, decoder):
    """
    Arguments:
    img -- original image
    encoder -- trained encoder network
    decoder -- trained decoder network
    """

    code = encoder.predict(img[np.newaxis, :])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]), cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()