import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2


def build_model(classes, input_shape, base_weights, full_weights=None):
	base_model = MobileNetV2(include_top=False, weights=base_weights, input_shape=input_shape)
	model = keras.Sequential()
	model.add(base_model)
	model.add(keras.layers.GlobalAveragePooling2D())
	if classes != None:
		model.add(keras.layers.Dense(classes, activation='softmax'))
	if full_weights:
		model.load_weights(full_weights)
	return model
