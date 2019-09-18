from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import streamlit as st

from utils import make_logdir
from data import build_dataset
from models import Classifier, build_convnet

import os
import argparse


def parse_args(): 
	desc = "Using transfer learning to train a classifier" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--ds_name', type=str, help='Dataset name passed to tfds', default="downsampled_imagenet/64x64")
	parser.add_argument('--res', type=int, help='Image Resolution', default=64)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=64)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


def train(model, dataset, epochs):
	optimizer = keras.optimizers.Adam(learning_rate=1e-3)
	loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	losses = []

	st.subheader("Training classifier...")
	progress = st.progress(0)
	chart = st.empty()

	for epoch in range(1, epochs+1):
		print('Starting Epoch {}'.format(epoch))
		for image_batch, label_batch in dataset:

			with tf.GradientTape() as tape:
				logits = model(image_batch)
				loss = loss_fn(y_true=label_batch, y_pred=logits)
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			losses.append(loss)
			chart.line_chart(losses)

		progress.progress(epoch*100//epochs)
		evaluate(model, dataset)


def evaluate(model, dataset):
	st.subheader("Evaluating classifier...")
	acc = keras.metrics.Accuracy()

	for image_batch, label_batch in dataset:
		logits = model(image_batch)
		y_pred = np.argmax(logits, axis=-1)
		acc.update_state(y_true=label_batch, y_pred=y_pred)
	
	st.write('Classifier Accuracy: {}'.format(acc.result().numpy()))
	print('Classifier Accuracy: {}'.format(acc.result().numpy()))


def main():
	args = parse_args()

	if args.save:
		logdir = make_logdir('train_classifier-{}'.format(args.ds_name))

	ds_train, info = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
	ds_test, _ = build_dataset(ds_name=args.ds_name, split='test', res=args.res, batch_size=args.batch_size)
	model = build_convnet(image_shape=(args.res, args.res, 3), num_classes=info.features['label'].num_classes)

	train(model, ds_train, args.epochs)
	evaluate(model, ds_test)

	if args.save:
		keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))


if __name__ == '__main__':
    main()

