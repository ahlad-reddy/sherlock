import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2


def build_rotation_model(weights):
	_weights = "imagenet" if weights == 'imagenet' else None
	_model = MobileNetV2(include_top=False, weights=_weights)
	model = keras.Sequential([
		_model, 
		keras.layers.GlobalAveragePooling2D(),
		keras.layers.Dense(4, activation='softmax')
	])
	if weights not in ["imagenet", None]:
		model.load_weights(weights)
	return model


def build_classifier_model(weights, transfer=True, classes):
	_model = build_rotation_model(weights if transfer else None)
	model = keras.Sequential([
		keras.Model(inputs=_model.inputs, outputs=_model.layers[-2].output),
		keras.layers.Dense(classes, activation='softmax')
	])
	if transfer==False:
		model.load_weights(weights)
	return model