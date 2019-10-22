from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
import tensorflow as tf
from math import ceil


def build_dataset(ds, image_shape, rotate, batch_size, epochs):
	def _preprocess_input(data):
		image = tf.io.read_file(data["image_path"])
		image = tf.io.decode_jpeg(image)

		shape = tf.shape(image)
		h, w, d = shape[0], shape[1], shape[2]
		image = tf.cond(tf.equal(d, 1), lambda: tf.image.grayscale_to_rgb(image), lambda: image)

		res = tf.minimum(h, w)
		h0, w0 = (h-res)//2, (w-res)//2
		image = tf.image.crop_to_bounding_box(image, h0, w0, res, res)

		image = tf.image.resize(image, image_shape)
		image = tf.dtypes.cast(image, tf.float32)
		image = image / 255.0		

		if rotate:
			image = tf.concat([tf.image.rot90(image, k=i) for i in range(4)], axis=0)
			label = [0, 1, 2, 3]
		else: 
			label = data["label"]

		return image, label

	def _rebatch(image, label):
		image = tf.reshape(image, (-1, *image_shape, 3))
		label = tf.reshape(label, (-1,))
		return image, label

	ds = ds.map(_preprocess_input)
	ds = ds.shuffle(2048).batch(batch_size).repeat(epochs).prefetch(tf.data.experimental.AUTOTUNE)
	if rotate:
		ds = ds.map(_rebatch)
	return ds


def build_yelp_dataset(split="train", image_shape=(224, 224), rotate=False, batch_size=16, epochs=None, take_per_class=None):
	data_file = "data/preprocessed/yelp_photos_{}.json".format(split)

	df = pd.read_json(data_file)
	df = df.sample(frac=1).reset_index(drop=True)

	if take_per_class:
		df = df.groupby("label").head(take_per_class)
	
	n_classes = df["label"].max()+1
	length = ceil(len(df)/batch_size)
	info = {"length": length, "classes": n_classes}

	ds = tf.data.Dataset.from_tensor_slices(dict(df[["image_path", "label"]]))
	ds = build_dataset(ds, image_shape, rotate, batch_size, epochs)

	return ds, info

