from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
import tensorflow as tf
from math import ceil


def build_dataset(ds, image_shape, rotate, batch_size, epochs, shuffle, take):
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
	ds = ds.take(take).batch(batch_size).repeat(epochs).prefetch(tf.data.experimental.AUTOTUNE)
	if rotate:
		ds = ds.map(_rebatch)
	return ds


def build_food101_dataset(split="train", image_shape=(224, 224), rotate=False, batch_size=16, epochs=None, shuffle=True, take=1):
	data_dir = "data/raw/food-101/"
	data_file = os.path.join(data_dir, "meta/{}.txt".format(split))
	labels_file = os.path.join(data_dir, "meta/classes.txt")

	df = pd.read_csv(data_file.format(split), names=["str_label", "id"], sep="/")
	df["image_path"] = df[['str_label', 'id']].apply(lambda x: os.path.join(data_dir, 'images/{}/{}.jpg'.format(x[0],x[1])), axis=1)

	classes = pd.read_csv(labels_file, header=None)
	classes = { classes.iloc[i][0]: i for i in range(len(classes)) }
	df["label"] = df[["str_label"]].apply(lambda x: classes[x[0]], axis=1)

	assert 0 < take <= 1
	take = round(len(df)*take)
	info["length"] = ceil(take/batch_size)

	if shuffle:
		df = df.sample(frac=1).reset_index(drop=True)

	ds = tf.data.Dataset.from_tensor_slices(dict(df[["image_path", "label"]]))
	ds = build_dataset(ds, image_shape, rotate, batch_size, epochs, shuffle, take)
	return ds, info


def build_yelp_dataset(splits=["train1"], image_shape=(224, 224), rotate=False, batch_size=16, epochs=None, shuffle=True):
	record_file = "data/preprocessed/yelp_photos_{}.tfrecords"
	ds = tf.data.TFRecordDataset([record_file.format(s) for s in splits])

	image_feature_description = {
	    'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64),
		'photo_id': tf.io.FixedLenFeature([], tf.string)
	}
	def _parse_image_function(example_proto):
		# Parse the input tf.Example proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, image_feature_description)

	ds = ds.map(_parse_image_function)
	ds = build_dataset(ds, image_shape, rotate, batch_size, epochs, shuffle, take)

	return ds, info

