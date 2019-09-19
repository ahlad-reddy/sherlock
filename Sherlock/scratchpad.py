import tensorflow as tf 
import tensorflow_datasets as tfds 
import time

import streamlit as st


def build_dataset(name, split, res, batch_size):
	ds, info = tfds.load(name=name, split=split, with_info=True)

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 255.0
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds = ds.map(preprocess_input)
	ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds, info


ds, ds_test, _ = build_dataset('uc_merced', [tfds.Split.TRAIN.subsplit(tfds.percent[:10]), tfds.Split.TRAIN.subsplit(tfds.percent[10:])], 224, 1)

i = st.empty()
l = st.empty()

for image, label in ds:
	i.image(image.numpy(), width=512)
	l.text(label)
	time.sleep(1)

