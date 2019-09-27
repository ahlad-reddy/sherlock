import time
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import streamlit as st
from data import build_yelp_dataset
import matplotlib.pyplot as plt

ds = build_yelp_dataset(rotate=True, batch_size=16)

im_plot = st.empty()
im_label = st.empty()

for im_batch, l_batch in ds:
	for i in range(4*16):
		im_plot.image(np.array(im_batch[i, :, :, :]))
		im_label.text(np.array(l_batch[i]))
		time.sleep(0.1)

