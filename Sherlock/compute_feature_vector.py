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
from PIL import Image
import pandas as pd

from data import build_missouri_dataset, build_yelp_dataset


def parse_args(): 
	desc = "Unsupervised training using image rotations on missouri camera traps dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--load_model', type=str, help='Path to model checkpoint', default="imagenet")
	parser.add_argument('--save_path', type=str, help='Path to saved features', required=True)
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)

	args = parser.parse_args()
	return args

def get_image(image_path, image_shape):
	im = Image.open(image_path)
	w, h = im.size
	res = min(h, w)
	h0, w0 = (h-res)//2, (w-res)//2
	im = im.resize(image_shape, box=(w0, h0, w0+res, h0+res))
	im = np.array(im) / 255.0
	return np.expand_dims(im, axis=0)


def compute_feature_vector(model, df, image_shape, save_path):
	for i in tqdm(range(len(df))):
		image = get_image(df.iloc[i]["image_path"], image_shape)
		df.at[i, "feature_vector"] = model.predict(image)[0]

	df.to_json(save_path)


def main(): 
	args = parse_args()

	# ds = build_yelp_dataset(splits=['train1'], image_shape=(args.res, args.res), rotate=False, batch_size=args.batch_size, epochs=1)
	data_file = "data/preprocessed/yelp_photos_train1.json"
	df = pd.read_json(data_file)

	if args.load_model == 'imagenet':
		model = keras.models.Sequential([
			MobileNetV2(input_shape=(args.res, args.res, 3), include_top=False, weights='imagenet'),
			keras.layers.GlobalAveragePooling2D()
		])
	else:
		base_model = MobileNetV2(weights=args.load_model, classes=4)
		model = keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer('global_average_pooling2d').output)

	compute_feature_vector(model, df, (args.res, args.res), args.save_path)

	


if __name__ == '__main__':
	main()