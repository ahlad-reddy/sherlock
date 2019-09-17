from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2

import numpy as np
import streamlit as st

import argparse


def parse_args():
	desc = "Training a classifier to predict whether an image will be classified correctly or incorrectly"  
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--ds_name', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--res', type=int, help='Image Resolution', default=256)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)

	args = parser.parse_args()
	return args


def build_dataset(ds_name, split, res, batch_size):
	ds = tfds.load(name=ds_name, split=split)

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds = ds.map(preprocess_input)
	ds = ds.shuffle(2048).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds


class HardImageClassifier(keras.Model):
	def __init__(self, image_shape):
		super(HardImageClassifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = tf.keras.layers.GlobalAveragePooling2D()
		self.dense_1 = keras.layers.Dense(21)
		self.dense_2 = keras.layers.Dense(1)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		class_logits = self.dense_1(x)
		hard_logits = self.dense_2(x)
		return class_logits, hard_logits


def train_class(model, dataset, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=1e-3)
	class_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	class_losses = []

	st.subheader("Training label classifier...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in dataset:

			with tf.GradientTape() as tape:
				class_logits, hard_logits = model(image_batch)
				class_loss = class_loss_fn(y_true=label_batch, y_pred=class_logits)
			grads = tape.gradient(class_loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			class_losses.append(class_loss)
			chart.line_chart(class_losses)

		progress.progress(epoch*100//epochs)


def evaluate_class(model, dataset):
	st.subheader("Evaluating classifier...")
	acc = keras.metrics.Accuracy()

	for image_batch, label_batch in dataset:
		class_logits, hard_logits = model(image_batch)
		y_pred = np.argmax(class_logits, axis=-1)
		acc.update_state(y_true=label_batch, y_pred=y_pred)
	
	st.write('Classifier Accuracy: {}'.format(acc.result().numpy()))


def train_hard(model, dataset, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=1e-3)
	hard_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
	hard_losses = []

	st.subheader("Training hard image predictor...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in dataset:

			with tf.GradientTape() as tape:
				class_logits, hard_logits = model(image_batch)
				hard_labels = label_batch.numpy() != np.argmax(class_logits, axis=-1)
				hard_loss = hard_loss_fn(y_true=hard_labels, y_pred=tf.squeeze(hard_logits))
			grads = tape.gradient(hard_loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			hard_losses.append(hard_loss)
			chart.line_chart(hard_losses)

		progress.progress(epoch*100//epochs)


def evaluate_hard(model, dataset):
	st.subheader("Evaluating hard image predictor...")
	acc = keras.metrics.Accuracy()
	rec = keras.metrics.Recall()
	pre = keras.metrics.Precision()

	for image_batch, label_batch in dataset:
		class_logits, hard_logits = model(image_batch)
		y_pred = tf.round(tf.nn.sigmoid(tf.squeeze(hard_logits)))
		y_true = label_batch.numpy() != np.argmax(class_logits, axis=-1)
		acc.update_state(y_true=y_true, y_pred=y_pred)
		rec.update_state(y_true=y_true, y_pred=y_pred)
		pre.update_state(y_true=y_true, y_pred=y_pred)

	st.write('Prediction Accuracy: {}'.format(acc.result().numpy()))
	st.write('Prediction Recall: {}'.format(rec.result().numpy()))
	st.write('Prediction Precision: {}'.format(pre.result().numpy()))


def main():
    args = parse_args()

    ds = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
    model = HardImageClassifier(image_shape=(args.res, args.res, 3))

    train_class(model, ds, args.epochs)
    evaluate_class(model, ds)

    train_hard(model, ds, args.epochs)
    evaluate_hard(model, ds)


if __name__ == '__main__':
    main()

