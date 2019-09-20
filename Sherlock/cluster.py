from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import confusion_matrix


def parse_args(): 
	desc = "Cluster a dataset and apply labels" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)

	args = parser.parse_args()
	return args


def build_dataset(name, res, batch_size):
	ds, info = tfds.load(name=name, split='train', with_info=True)

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 255.0
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds = ds.map(preprocess_input).batch(batch_size)
	return ds, info


def cluster(model, ds, n_clusters, batch_size):
	kmeans = KMeans(n_clusters=n_clusters, max_iter=3000)

	images = None
	features = None
	labels = None

	progress = st.progress(0)
	i = 0
	for image_batch, label_batch in ds:
		if type(images) != type(None):
			images = np.concatenate((images, image_batch.numpy()))
		else:
			images = image_batch.numpy()

		f = model(image_batch)
		if type(features) != type(None):
			features = np.concatenate((features, f.numpy()))
		else:
			features = f.numpy()

		if type(labels) != type(None):
			labels = np.concatenate((labels, np.squeeze(label_batch.numpy())))
		else:
			labels = np.squeeze(label_batch.numpy())

		i+=1
		progress.progress(i*32*100//2100)

	kmeans.fit(features)

	n = len(labels)
	pred = kmeans.labels_
	true = labels
	cm = confusion_matrix(true, pred)
	sn.heatmap(cm, annot=True, square=True, cbar=False)
	plt.xlim((0, n_clusters))
	plt.ylim((0, n_clusters))
	st.pyplot()


def main(): 
	args = parse_args()

	ds, info = build_dataset(name=args.dataset, res=args.res, batch_size=args.batch_size)

	model = keras.models.Sequential([
		MobileNetV2(input_shape=(args.res, args.res, 3), include_top=False, weights='imagenet'),
		keras.layers.GlobalAveragePooling2D()
		])

	cluster(model, ds, info.features['label'].num_classes, args.batch_size)


if __name__ == '__main__':
	main()

