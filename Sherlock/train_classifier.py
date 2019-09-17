from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2

import numpy as np
import streamlit as st

from utils import parse_args, build_dataset, make_logdir
import os


class Classifier(keras.Model):
	def __init__(self, image_shape, num_classes):
		super(Classifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = keras.layers.GlobalAveragePooling2D()
		self.dense = keras.layers.Dense(num_classes)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		logits = self.dense(x)
		return logits


def train(model, dataset, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=1e-3)
	loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	losses = []

	st.subheader("Training classifier...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in dataset:

			with tf.GradientTape() as tape:
				logits = model(image_batch)
				loss = loss_fn(y_true=label_batch, y_pred=logits)
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			losses.append(loss)
			chart.line_chart(losses)

		progress.progress(epoch*100//epochs)


def evaluate(model, dataset):
	st.subheader("Evaluating classifier...")
	acc = keras.metrics.Accuracy()

	for image_batch, label_batch in dataset:
		logits = model(image_batch)
		y_pred = np.argmax(logits, axis=-1)
		acc.update_state(y_true=label_batch, y_pred=y_pred)
	
	st.write('Classifier Accuracy: {}'.format(acc.result().numpy()))


def main():
	desc = "Using transfer learning to train a classifier" 
	args = parse_args(desc)

	logdir = make_logdir('train_classifier-{}'.format(args.ds_name))

	ds, info = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
	model = Classifier(image_shape=(args.res, args.res, 3), num_classes=info.features['label'].num_classes)

	train(model, ds, args.epochs)
	evaluate(model, ds)
	keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))


if __name__ == '__main__':
    main()

