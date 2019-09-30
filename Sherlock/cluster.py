from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse
import json
from tqdm import tqdm

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
tf.enable_eager_execution()
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

	parser.add_argument('--load_features', type=str, help='Path to saved features', default="logdir/yelp_photos_023/imagenet_fv.npy")
	parser.add_argument('--n_clusters', type=int, help='Number of Clusters', default=20)

	args = parser.parse_args()
	return args


def cluster(features, labels, n_clusters, n_classes):
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(features)

	yticklabels = ['inside', 'outside', 'food', 'drink', 'menu']

	pred = kmeans.labels_
	true = labels
	n_clusters = max(pred)
	cm = confusion_matrix(true, pred)
	fig, ax = plt.subplots(figsize=(10, 10))
	sn.heatmap(cm, annot=False, cbar=True, ax=ax, yticklabels=yticklabels)
	plt.xlim((0, n_clusters+1))
	plt.ylim((0, n_classes))
	st.pyplot()


def main(): 
	args = parse_args()

	array = np.load(args.load_features)
	labels, features = array[:, 0], array[:, 1:]

	cluster(features, labels, args.n_clusters, 5)


if __name__ == '__main__':
	main()
