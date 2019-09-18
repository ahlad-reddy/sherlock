import argparse
import os
import glob

import tensorflow as tf
import tensorflow_datasets as tfds


def parse_args(desc): 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--ds_name', type=str, help='Dataset name passed to tfds', default="imagenet2012")
	parser.add_argument('--res', type=int, help='Image Resolution', default=224)
	parser.add_argument('--batch_size', type=int, help='Batch Size', default=16)
	parser.add_argument('--epochs', type=int, help='Training Epochs', default=1)

	args = parser.parse_args()
	return args


def build_dataset(ds_name, split, res, batch_size):
	ds, info = tfds.load(name=ds_name, split=split, with_info=True)
	buffer_size = info.splits[split].num_examples if info.splits[split].num_examples <= 10000 else 10000

	def preprocess_input(data):
		image = tf.dtypes.cast(data["image"], tf.float32)
		image = image / 127.5 - 1
		image = tf.image.resize_with_crop_or_pad(image, res, res)
		label = data["label"]
		return image, label

	ds = ds.map(preprocess_input)
	ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	return ds, info



def make_logdir(name):
    if not os.path.exists('logdir'): os.mkdir('logdir')
    logdir = 'logdir/{}_{:03d}'.format(name, len(glob.glob('logdir/*')))
    os.mkdir(logdir)
    print('Saving to results to {}'.format(logdir))
    return logdir

