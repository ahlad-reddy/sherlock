import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np


def build_dataset(ds_name, split, res, batch_size):
	ds, info = tfds.load(name=ds_name, split=split, with_info=True)
	buffer_size = info.splits[split].num_examples if info.splits[split].num_examples <= 10000 else 10000

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds = ds.map(preprocess_input)
	ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds, info


def build_rotation_dataset(ds_name, split, res, batch_size):
	ds, info = tfds.load(name=ds_name, split=split, with_info=True)
	buffer_size = info.splits[split].num_examples if info.splits[split].num_examples <= 10000 else 10000

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = np.random.randint(4)
		image = tf.image.rot90(image, k=label)
		return image, label

	ds = ds.map(preprocess_input)
	ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds, info