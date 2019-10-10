from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from model import build_model

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix


def parse_args(): 
	desc = "Cluster a dataset and apply labels" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--model', type=str, help='Path to saved features', default="imagenet")
	parser.add_argument('--save_path', type=str, help='Path to save json file')
	parser.add_argument('--n_clusters', type=int, help='Number of Clusters', default=20)
	parser.add_argument('--n_datapoints', type=int, help='Number of datapoints to use', default=5000)
	parser.add_argument('--seed', type=int, help='Random Seed', default=2019)

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


def load_image(image_path, image_shape):
	im = Image.open(image_path)
	w, h = im.size
	res = min(h, w)
	h0, w0 = (h-res)//2, (w-res)//2
	im = im.resize(image_shape, box=(w0, h0, w0+res, h0+res))
	im = im.convert('RGB')
	return im


def compute_features(model, image_path, image_shape):
	im = load_image(image_path, image_shape)
	im = np.expand_dims(np.array(im) / 255.0, axis=0)
	return np.squeeze(model.predict(im))


def main(): 
	args = parse_args()

	df = build_dataframe()
	df = df.sample(n=args.n_datapoints, random_state=args.seed)
	model = build_model(classes=None, input_shape=(args.res, args.res, 3), base_weights=args.model)
	image_shape = (args.res, args.res)

	df["feature_vector"] = df[["image_path"]].apply(lambda x: compute_features(model, x[0], image_shape), axis=1)
	kmeans = KMeans(n_clusters=args.n_clusters)
	distances = kmeans.fit_transform(list(df["feature_vector"]))
	df["distance"] = distances.min(axis=1)
	df["cluster"] = kmeans.labels_
	df.to_json(args.save_path)


if __name__ == '__main__':
	main()

