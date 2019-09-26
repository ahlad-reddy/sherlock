import os
import glob
import json
import tensorflow as tf
from tensorflow.io import TFRecordWriter as Writer
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.enable_eager_execution()
from tqdm import tqdm
import numpy as np
np.random.seed(1000)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

val_test_split = np.array([0.7, 0.85])
train_seqs, val_seqs, test_seqs = [], [], []
for folder in glob.glob("data/raw/Set1/*/"):
	sequences = [os.path.basename(seq) for seq in glob.glob(os.path.join(folder, '*'))]
	np.random.shuffle(sequences)
	splits = (len(sequences) * val_test_split).astype('uint8')

	train_seqs+=sequences[:splits[0]]
	val_seqs+=sequences[splits[0]:splits[1]]
	test_seqs+=sequences[splits[1]:]


CROPPED_DIM = (1010, 1920)
metadata = json.load(open('data/raw/missouri_camera_traps_set1.json'))
record_file = 'data/preprocessed/missouri_camera_traps_set1_{}.tfrecords'


i = 0

with Writer(record_file.format("train")) as train_writer, Writer(record_file.format("val")) as val_writer, Writer(record_file.format("test")) as test_writer:
	for meta in tqdm(metadata["images"]):
		file = os.path.join('data/raw', meta["file_name"].replace('\\', '/'))
		try:
			contents = tf.io.read_file(file, 'rb')
			image = tf.io.decode_image(contents)
			x0, y0 = (image.shape[0]-CROPPED_DIM[0])//2, (image.shape[1]-CROPPED_DIM[1])//2
			image = image[x0:x0+CROPPED_DIM[0], y0:y0+CROPPED_DIM[1], :]
			image_string = tf.io.encode_jpeg(image)

			while metadata["annotations"][i]["image_id"] != meta["id"]:
				i += 1
			label = metadata["annotations"][i]["category_id"]

			feature = {
				'image': _bytes_feature(image_string),
				'label': _int64_feature(label)
			}

			tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

			if meta["seq_id"] in train_seqs:
				w = train_writer
			elif meta["seq_id"] in val_seqs:
				w = val_writer
			elif meta["seq_id"] in test_seqs:
				w = test_writer
			else:
				print('seq_id not in seq lists')
				raise
			w.write(tf_example.SerializeToString())

		except tf.errors.InvalidArgumentError as err:
			print('Corrupted file, unable to process - {}'.format(file))


