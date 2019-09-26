import time
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import streamlit as st
from data import build_yelp_dataset

ds = build_yelp_dataset(rotate=True, batch_size=1, take=100)

for i, l in ds:
	st.image(np.array(i))
	st.text(l)
	break

