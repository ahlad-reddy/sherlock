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


def parse_args(desc): 
	desc = "Training a classifier to predict whether an image will be classified correctly or incorrectly" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--ds_name', type=str, help='Dataset name passed to tfds', default="uc_merced")
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


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
	args = parse_args()

	if args.save:
		logdir = make_logdir('predict_hard_images-{}'.format(args.ds_name))

	ds, _ = build_dataset(ds_name=args.ds_name, split='train', res=args.res, batch_size=args.batch_size)
	model = HardImageClassifier(image_shape=(args.res, args.res, 3), num_classes=info.features['label'].num_classes)

	train(model, ds, args.epochs)
	evaluate(model, ds)
	model.set_fine_tuning_layers(100)
	train(model, ds, args.epochs)
	evaluate(model, ds)

	if args.save:
		keras.experimental.export_saved_model(model, os.path.join(logdir, 'model'))


if __name__ == '__main__':
    main()