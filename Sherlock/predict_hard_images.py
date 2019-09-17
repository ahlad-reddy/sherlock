from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2

import numpy as np
import streamlit as st

from utils import parse_args, build_dataset, make_logdir


class HardImageClassifier(keras.Model):
	def __init__(self, image_shape, num_classes):
		super(HardImageClassifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = tf.keras.layers.GlobalAveragePooling2D()
		self.dense_1 = keras.layers.Dense(num_classes)
		self.dense_2 = keras.layers.Dense(1)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		class_logits = self.dense_1(x)
		hard_logits = self.dense_2(x)
		return class_logits, hard_logits


def train(model, dataset, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=1e-3)
	loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
	losses = []

	st.subheader("Training hard image predictor...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in dataset:

			with tf.GradientTape() as tape:
				class_logits, hard_logits = model(image_batch)
				hard_labels = label_batch.numpy() != np.argmax(class_logits, axis=-1)
				loss = loss_fn(y_true=hard_labels, y_pred=tf.squeeze(hard_logits))
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			losses.append(loss)
			chart.line_chart(losses)

		progress.progress(epoch*100//epochs)


def evaluate(model, dataset):
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
	desc = "Training a classifier to predict whether an image will be classified correctly or incorrectly"  
	args = parse_args(desc)

    # logdir = make_logdir('predict_hard_images-{}'.format(args.ds_name))

	ds, info = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
	model = HardImageClassifier(image_shape=(args.res, args.res, 3), num_classes=info.features['label'].num_classes)

	train(model, ds, args.epochs)
	evaluate(model, ds)
	# keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))


if __name__ == '__main__':
    main()

