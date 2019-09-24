from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse
import time

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from data import build_missouri_dataset


def parse_args(): 
	desc = "Unsupervised training using image rotations on missouri camera traps dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--lr', type=float, help='Learning Rate', default=1e-3)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=20)
	parser.add_argument('--save', help='Save model', action='store_true')

	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	callbacks = None
	if args.save:
		logdir = 'logdir/19_09_24_missouri_camera_traps'
		callbacks = [keras.callbacks.TensorBoard(log_dir=logdir)]

	ds = build_missouri_dataset(image_shape=(args.res, args.res), rgb=True, rotate=True, batch_size=args.batch_size)

	model = MobileNetV2(weights=None, classes=4)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	model.fit(ds, epochs=args.epochs,
				  callbacks=callbacks)

	if args.save:
		model.save(os.path.join(logdir, 'mobilenetv2.h5'))


if __name__ == '__main__':
	main()

