from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow import keras

import numpy as np
import streamlit as st

from utils import make_logdir
from data import build_rotation_dataset as build_dataset
from models import Classifier

import os
import argparse


def parse_args(): 
	desc = "Training a vision model with unsupervised learning, predicting image rotations" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--ds_name', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=10)
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
	args = parse_args()

	logdir = make_logdir('predict_image_rotations-{}'.format(args.ds_name))

	ds, info = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
	model = Classifier(image_shape=(args.res, args.res, 3), num_classes=4)

	train(model, ds, args.epochs)
	evaluate(model, ds)
	keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))


if __name__ == '__main__':
    main()

