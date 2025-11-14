import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

# Load InceptionV3 model without the top classification layer, with no average pooling leaving the shape to be 8x8x2048
#model = InceptionV3(include_top=False, weights='imagenet', pooling=None) vectorizeing 
model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

def extract_features(directory, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    filenames = sorted(os.listdir(directory))
    total_images = len(filenames)
    chunk_size = 79

    for idx in range(0, total_images, chunk_size):
        chunk_filenames = filenames[idx:idx+chunk_size]
        chunk_images = []

        for fname in chunk_filenames:
            path = os.path.join(directory, fname)
            try:
                img = image.load_img(path, target_size=(299, 299))  # for InceptionV3
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                chunk_images.append(x[0])
            except Exception as e:
                print(f"Error loading image {fname}: {e}")
                continue

        if not chunk_images:
            continue

        chunk_images = np.array(chunk_images)
        chunk_features = model.predict(chunk_images)

        # Save this batch's features
        features = {}
        for fname, feat in zip(chunk_filenames, chunk_features):
            img_id = os.path.splitext(fname)[0]
            features[img_id] = feat

        part_name = f"train_features_part_{idx//chunk_size:03}.pkl"
        save_path = os.path.join(save_dir, part_name)
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"Saved: {save_path}")

directory_val = "./train2014"
extract_features(directory_val, model, "./incep/train_features2")
