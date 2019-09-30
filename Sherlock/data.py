from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import tensorflow as tf


def build_yelp_dataset(splits=["train1"], image_shape=(224, 224), rotate=False, batch_size=16):
	record_file = "data/preprocessed/yelp_photos_{}.tfrecords"
	ds = tf.data.TFRecordDataset([record_file.format(s) for s in splits])

	image_feature_description = {
	    'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64)
	}
	def _parse_image_function(example_proto):
		# Parse the input tf.Example proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, image_feature_description)

	def _preprocess_input(data):
		image = tf.io.decode_jpeg(data["image"])
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

	ds = ds.map(_parse_image_function)
	ds = ds.map(_preprocess_input)
	ds = ds.shuffle(2048).batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
	if rotate:
		ds = ds.map(_rebatch)

	return ds


def build_missouri_dataset(split='train' , image_shape=(224, 224), rgb=True, rotate=False, batch_size=16):
	assert split in ['train', 'val', 'test', 'full']
	record_file = 'data/preprocessed/missouri_camera_traps_set1_{}.tfrecords'.format(split)
	ds = tf.data.TFRecordDataset(record_file)

	image_feature_description = {
	    'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64)
	}
	def _parse_image_function(example_proto):
		# Parse the input tf.Example proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, image_feature_description)

	def _preprocess_input(data):
		image = tf.io.decode_jpeg(data["image"])
		image = tf.image.resize(image, image_shape)

		if rgb == False: 
			image = tf.image.rgb_to_grayscale(image)
		image = tf.dtypes.cast(image, tf.float32)
		image = image / 255.0

		if rotate:
			image = tf.concat([tf.image.rot90(tf.expand_dims(image, axis=0), k=i) for i in range(4)], axis=0)
			label = [0, 1, 2, 3]
		else: 
			label = data["label"]

		return image, label

	def rebatch(image, label):
		image = tf.reshape(image, (-1, *image_shape, 3 if rgb else 1))
		label = tf.reshape(label, (-1,))
		return image, label

	ds = ds.map(_parse_image_function)
	ds = ds.map(_preprocess_input)
	ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	if rotate:
		ds = ds.map(_rebatch)
	return ds

