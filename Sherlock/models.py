import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2


def build_convnet(image_shape, num_classes):
	model = keras.models.Sequential([
		Conv2D(16, 3, padding='same', activation='relu', input_shape=image_shape),
		MaxPooling2D(),
		Dropout(0.5),
		Conv2D(32, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Conv2D(64, 3, padding='same', activation='relu'),
		MaxPooling2D(),
		Dropout(0.5),
		Flatten(),
		Dense(256, activation='relu'),
		Dense(num_classes)
	])
	return model


class Classifier(keras.Model):
	def __init__(self, image_shape, num_classes):
		super(Classifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = keras.layers.GlobalAveragePooling2D()
		self.dense = keras.layers.Dense(num_classes)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		logits = self.dense(x)
		return logits

	def set_fine_tuning_layers(self, fine_tune_at=-1):
			if fine_tune_at == -1:
				self.base_model.trainable = False
			else:
				self.base_model.trainable = True
				for layer in self.base_model.layers[:fine_tune_at]:
					layer.trainable = False


class MixedClassifier(keras.Model):
	def __init__(self, image_shape, num_classes):
		super(MixedClassifier, self).__init__()
		self.base_model = MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')
		self.base_model.trainable = False
		self.pool = tf.keras.layers.GlobalAveragePooling2D()
		self.dense_1 = keras.layers.Dense(num_classes)
		self.dense_2 = keras.layers.Dense(1)

	def call(self, x):
		x = self.base_model(x)
		x = self.pool(x)
		class_logits = self.dense_1(x)
		hard_logits = self.dense_2(x)
		return class_logits, hard_logits