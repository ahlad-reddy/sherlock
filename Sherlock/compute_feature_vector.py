from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
from tqdm import tqdm

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
tf.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import streamlit as st

from data import build_missouri_dataset, build_yelp_dataset


def parse_args(): 
	desc = "Unsupervised training using image rotations on missouri camera traps dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--load_model', type=str, help='Path to model checkpoint', default="imagenet")
	parser.add_argument('--save_path', type=str, help='Path to saved features', required=True)
	parser.add_argument('--res', type=int, help='Image Resolution', default=96)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)

	args = parser.parse_args()
	return args


def compute_feature_vector(model, ds, save_path):
	features = None
	labels = None

	for image_batch, label_batch in tqdm(ds):
		f = np.array(model(image_batch))
		label_batch = np.array(label_batch)
		if type(features) != type(None):
			features = np.concatenate((features, f))
		else:
			features = f

		if type(labels) != type(None):
			labels = np.concatenate((labels, np.squeeze(label_batch)))
		else:
			labels = np.squeeze(label_batch)

	out = np.concatenate([np.expand_dims(labels, axis=1), features], axis=1)
	np.save(save_path, out)


def main(): 
	args = parse_args()

	ds = build_yelp_dataset(splits=['train1'], image_shape=(args.res, args.res), rotate=False, batch_size=args.batch_size)

	if args.load_model == 'imagenet':
		model = keras.models.Sequential([
			MobileNetV2(input_shape=(args.res, args.res, 3), include_top=False, weights='imagenet'),
			keras.layers.GlobalAveragePooling2D()
		])
	else:
		base_model = MobileNetV2(weights=args.load_model, classes=4)
		model = keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer('global_average_pooling2d').output)

	compute_feature_vector(model, ds, args.save_path)

	


if __name__ == '__main__':
	main()