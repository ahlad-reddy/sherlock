import os
import glob
import json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import streamlit as st


def build_missouri_dataset(image_shape=(224, 224), rgb=True, rotate=False, batch_size=16):
	record_file = 'data/preprocessed/missouri_camera_traps_set1.tfrecords'
	ds = tf.data.TFRecordDataset(record_file)

	image_feature_description = {
	    'image': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64)
	}
	def _parse_image_function(example_proto):
		# Parse the input tf.Example proto using the dictionary above.
		return tf.io.parse_single_example(example_proto, image_feature_description)

	def preprocess_input(data):
		image = tf.io.decode_jpeg(data["image"])
		image = tf.image.resize(image, image_shape)

		if rgb == False: 
			image = tf.image.rgb_to_grayscale(image)
		image = tf.dtypes.cast(image, tf.float32)
		image = image / 255.0

		if rotate:
			label = tf.random.uniform((1,), minval=0, maxval=4, dtype=tf.dtypes.int32)[0]
			image = tf.image.rot90(image, k=label)
		else: 
			label = data["label"]

		return image, label

	ds = ds.map(_parse_image_function)
	ds = ds.map(preprocess_input)
	ds = ds.shuffle(128).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds
