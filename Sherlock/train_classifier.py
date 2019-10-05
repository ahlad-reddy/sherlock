from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2

import numpy as np
from data import build_missouri_dataset, build_yelp_dataset, build_food101_dataset
from math import ceil


def parse_args(): 
	desc = "Transfer learning on classification task" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-3)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)
	parser.add_argument('--model', type=str, help='Path to pretrained model or "imagenet"', default=None)
	parser.add_argument('--take', type=float, help='Fraction of dataset used to train', default=0.1)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	callbacks = None
	if args.save:
		logdir = 'logdir/{}_{:03d}'.format("food101_classifier", len(glob.glob('logdir/*')))
		print('Saving to {}'.format(logdir))
		callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(logdir, 'mobilenetv2.h5')), 
					 keras.callbacks.TensorBoard(log_dir=logdir, update_freq=50)]

	ds_train, train_len = build_food101_dataset(split='train', image_shape=(args.res, args.res), rotate=False, batch_size=args.batch_size, take=args.take)
	ds_test, test_len = build_food101_dataset(split='test', image_shape=(args.res, args.res), rotate=False, batch_size=args.batch_size)

	if args.model in ['imagenet', None]:
		base_model = MobileNetV2(input_shape=(args.res, args.res, 3), weights=args.model, classes=1000)
	else:
		base_model = MobileNetV2(input_shape=(args.res, args.res, 3), weights=args.model, classes=4)
	# base_model.trainable = False
	
	_model = keras.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
	model = keras.Sequential([_model, keras.layers.Dense(101, activation='softmax')])

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	model.fit(ds_train, 
			  epochs=args.epochs, 
			  steps_per_epoch=train_len,
			  validation_data=ds_test,
			  validation_steps=test_len,
			  callbacks=callbacks)

	# for layer in model.layers[0].layers[100:]:
	# 	layer.trainable = True

	# model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr//10),
	# 		  loss='sparse_categorical_crossentropy',
	# 		  metrics=['accuracy'])

	# model.fit(ds_train, 
	# 		  epochs=args.epochs, 
	# 		  steps_per_epoch=2000,
	# 		  validation_data=ds_val,
	# 		  validation_steps=2500,
	# 		  callbacks=callbacks)

	# model.evaluate(ds_test, callbacks=callbacks, steps=test_len)


if __name__ == '__main__':
	main()
