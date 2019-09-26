import os
import json
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow.io import TFRecordWriter
tf.enable_eager_execution()
import numpy as np
np.random.seed(1000)


YELP_CLASSES = {
	"inside" : 0,
	"outside": 1,
	"food"   : 2,
	"drink"  : 3,
	"menu"   : 4
}

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

meta_file = "data/raw/photo.json"
image_dir = "data/raw/photos"
record_file = "data/preprocessed/yelp_photos_{}.tfrecords"
data = open(meta_file).readlines()
n = len(data)
split_idx = n * np.array([0, 0.16, 0.32, 0.48, 0.64, 0.80, 1.0])
split_idx = split_idx.astype('int32')
idx = np.arange(n)
np.random.shuffle(idx)

for i, split in enumerate(tqdm(['train1', 'train2', 'train3', 'train4', 'train5', 'test'])):
	with TFRecordWriter(record_file.format(split)) as writer:
		for j in tqdm(idx[split_idx[i]:split_idx[i+1]]):
			datapoint = json.loads(data[j])
			image_file = os.path.join(image_dir, datapoint["photo_id"] + '.jpg')
			image = tf.io.decode_jpeg(tf.io.read_file(image_file, 'rb'))

			h, w, _ = image.shape
			res = min(h, w)
			h0, w0 = (h-res)//2, (w-res)//2
			image = tf.image.crop_to_bounding_box(image, h0, w0, res, res)
			image_string = tf.io.encode_jpeg(image)
			label = YELP_CLASSES[datapoint["label"]]

			feature = {
				'image': _bytes_feature(image_string.numpy()),
				'label': _int64_feature(label)
			}
			tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(tf_example.SerializeToString())

