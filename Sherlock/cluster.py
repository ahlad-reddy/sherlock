from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix


def parse_args(): 
	desc = "Cluster a dataset and apply labels" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--model', type=str, help='Path to saved features', default="imagenet")
	parser.add_argument('--save_path', type=str, help='Path to save json file')
	parser.add_argument('--n_clusters', type=int, help='Number of Clusters', default=20)
	parser.add_argument('--n_datapoints', type=int, help='Number of datapoints to use', default=5000)
	parser.add_argument('--seed', type=int, help='Number of Clusters', default=2019)

	args = parser.parse_args()
	return args


def build_dataframe():
	data_dir = "data/raw/food-101/"
	data_file = os.path.join(data_dir, "meta/train.txt")
	labels_file = os.path.join(data_dir, "meta/classes.txt")

	df = pd.read_csv(data_file, names=["str_label", "id"], sep="/")
	df["image_path"] = df[['str_label', 'id']].apply(lambda x: os.path.join(data_dir, 'images/{}/{}.jpg'.format(x[0],x[1])), axis=1)
	df["assigned_label"] = ""
	return df


def build_model(weights="imagenet"):
	classes = 1000 if weights=="imagenet" else 4
	_model = MobileNetV2(include_top=False, weights=weights, input_shape=(224, 224, 3), classes=classes)
	model = tf.keras.Sequential([_model, tf.keras.layers.GlobalAveragePooling2D()])
	return model


def load_image(image_path, image_shape=(224, 224)):
	im = Image.open(image_path)
	w, h = im.size
	res = min(h, w)
	h0, w0 = (h-res)//2, (w-res)//2
	im = im.resize(image_shape, box=(w0, h0, w0+res, h0+res))
	im = im.convert('RGB')
	return im


def compute_features(model, image_path):
	im = load_image(image_path)
	im = np.expand_dims(np.array(im) / 255.0, axis=0)
	return np.squeeze(model.predict(im))


def main(): 
	args = parse_args()

	df = build_dataframe()
	df = df.sample(n=args.n_datapoints, random_state=args.seed)
	model = build_model(args.model)

	df["feature_vector"] = df[["image_path"]].apply(lambda x: compute_features(model, x[0]), axis=1)
	kmeans = KMeans(n_clusters=args.n_clusters)
	distances = kmeans.fit_transform(list(df["feature_vector"]))
	df["distance"] = distances.min(axis=1)
	df["cluster"] = kmeans.labels_
	df.to_json(args.save_path)



if __name__ == '__main__':
	main()

