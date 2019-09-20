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


def parse_args(): 
	desc = "Unsupervised training using image rotations" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--split', type=int, help='Train/Test Percentage Split', default=90)
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-3)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


def build_dataset(name, split, res, batch_size):
	train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:split])
	test_split = tfds.Split.TRAIN.subsplit(tfds.percent[split:])
	ds_train, info = tfds.load(name=name, split=train_split, with_info=True)
	ds_test, info = tfds.load(name=name, split=test_split, with_info=True)

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = tf.random.uniform((1,), minval=0, maxval=4, dtype=tf.dtypes.int32)
		image = tf.image.rot90(image, k=label[0])
		return image, label

	ds_train = ds_train.map(preprocess_input).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	ds_test = ds_test.map(preprocess_input).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds_train, ds_test, info


class RotationClassifier(keras.Model):
	def __init__(self, image_shape, weights=None):
		super(RotationClassifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights=weights)
		self.pool = keras.layers.GlobalAveragePooling2D()
		self.dense = keras.layers.Dense(4)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		logits = self.dense(x)
		return logits


def train(model, ds_train, ds_test, lr, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=lr)
	loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	accuracy = []

	st.subheader("Training model...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in ds_train:

			with tf.GradientTape() as tape:
				logits = model(image_batch)
				loss = loss_fn(y_true=label_batch, y_pred=logits)
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))


		acc = evaluate(model, ds_test)
		accuracy.append(acc)
		chart.line_chart(accuracy)
		progress.progress(epoch*100//epochs)


def evaluate(model, dataset):
	st.subheader("Evaluating classifier...")
	acc = keras.metrics.Accuracy()

	for image_batch, label_batch in dataset:
		logits = model(image_batch)
		y_pred = np.argmax(logits, axis=-1)
		acc.update_state(y_true=label_batch, y_pred=y_pred)
	
	st.write('Accuracy: {}'.format(acc.result().numpy()))
	return acc.result().numpy()


def main():
	args = parse_args()

	ds_train, ds_test, info = build_dataset(name=args.dataset, split=args.split, res=args.res, batch_size=args.batch_size)
	model = RotationClassifier(image_shape=(args.res, args.res, 3))

	train(model, ds_train, ds_test, args.lr, args.epochs)

	if args.save:
		keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))
		logdir = make_logdir('rotation_network-{}'.format(args.dataset))


if __name__ == '__main__':
    main()

