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


def parse_args(): 
	desc = "Train a classifier with confidence score" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--dataset', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--split', type=int, help='Train/Test Percentage Split', default=20)
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--lr', type=float, help='Image Resolution', default=1e-3)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)
	parser.add_argument('--fine_tune', type=int, help='Layer to begin fine tuning', default=100)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


def make_logdir(name):
	if not os.path.exists('logdir'): os.mkdir('logdir')
	logdir = 'logdir/{}_{:03d}'.format(name, len(glob.glob('logdir/*')))
	os.mkdir(logdir)
	print('Saving to results to {}'.format(logdir))
	return logdir


def build_dataset(name, split, res, batch_size):
	train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:split])
	test_split = tfds.Split.TRAIN.subsplit(tfds.percent[split:])
	ds_train, info = tfds.load(name=name, split=train_split, with_info=True)
	ds_test, info = tfds.load(name=name, split=test_split, with_info=True)

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds_train = ds_train.map(preprocess_input).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	ds_test = ds_test.map(preprocess_input).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds_train, ds_test, info


class ConfidenceClassifier(keras.Model):
	def __init__(self, image_shape, num_classes):
		super(ConfidenceClassifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = keras.layers.GlobalAveragePooling2D()
		self.dense = keras.layers.Dense(num_classes+1)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		logits = self.dense(x)
		class_logits = logits[:, :-1]
		confi_logits = logits[:, -1]
		return class_logits, confi_logits

	def set_fine_tuning_layers(self, fine_tune_at=-1):
			if fine_tune_at == -1:
				self.base_model.trainable = False
			else:
				self.base_model.trainable = True
				for layer in self.base_model.layers[:fine_tune_at]:
					layer.trainable = False


def train(model, ds_train, ds_test, lr, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=lr)
	class_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	confi_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
	losses = []

	st.subheader("Training model...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		for image_batch, label_batch in ds_train:

			with tf.GradientTape() as tape:
				class_logits, confi_logits = model(image_batch)
				confi_labels = label_batch.numpy() == np.argmax(class_logits, axis=-1)

				class_loss = class_loss_fn(y_true=label_batch, y_pred=class_logits)
				confi_loss = confi_loss_fn(y_true=confi_labels, y_pred=tf.squeeze(confi_logits))
				total_loss = class_loss + confi_loss
			grads = tape.gradient(total_loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			losses.append((class_loss.numpy(), confi_loss.numpy(), total_loss.numpy()))
			chart.line_chart(losses)

		evaluate(model, ds_train)
		evaluate(model, ds_test)
		progress.progress(epoch*100//epochs)


def evaluate(model, dataset):
	st.subheader("Evaluating model...")
	cat = keras.metrics.CategoricalAccuracy()
	acc = keras.metrics.Accuracy()
	rec = keras.metrics.Recall()
	pre = keras.metrics.Precision()

	for image_batch, label_batch in dataset:
		class_logits, confi_logits = model(image_batch)
		y_pred = tf.round(tf.nn.sigmoid(tf.squeeze(confi_logits)))
		y_true = label_batch.numpy() == np.argmax(class_logits, axis=-1)

		cat.update_state(y_true=label_batch, y_pred=np.argmax(class_logits, axis=-1))
		acc.update_state(y_true=y_true, y_pred=y_pred)
		rec.update_state(y_true=y_true, y_pred=y_pred)
		pre.update_state(y_true=y_true, y_pred=y_pred)


	st.write('Classifier Accuracy: {}'.format(cat.result().numpy()))
	st.write('Confidence Accuracy: {}'.format(acc.result().numpy()))
	st.write('Confidence Recall: {}'.format(rec.result().numpy()))
	st.write('Confidence Precision: {}'.format(pre.result().numpy()))


def main(): 
	args = parse_args()

	ds_train, ds_test, info = build_dataset(name=args.dataset, split=args.split, res=args.res, batch_size=args.batch_size)

	model = ConfidenceClassifier(image_shape=(args.res, args.res, 3), num_classes=info.features['label'].num_classes)

	train(model, ds_train, ds_test, args.lr, args.epochs)

	model.set_fine_tuning_layers(args.fine_tune)
	train(model, ds_train, ds_test, args.lr/10, args.epochs)

	if args.save:
		keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))
		logdir = make_logdir('confidence_classifier-{}'.format(args.ds_name))


if __name__ == '__main__':
	main()

