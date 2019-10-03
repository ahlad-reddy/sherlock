# import time
import tensorflow as tf
tf.enable_eager_execution()
import glob
import numpy as np
import streamlit as st
import pandas as pd
import json
from data import build_yelp_dataset, build_dataset
from cluster import cluster
from PIL import Image
import time
# import matplotlib.pyplot as plt

st.title("Visualizing and Labeling Clusters")


features_options = glob.glob('logdir/*/*.json')
features_select = st.selectbox("Path to saved features", options=features_options)
n_clusters = st.slider("Number of clusters to generate", min_value=2, max_value=100, value=10)

if st.button('Apply K-Means Clustering'):
	st.write('doing stuff')
	df = pd.read_json(features_select)

	st.write('done stuff')
	st.write(features[0])
	st.write(labels[0])



# for im_batch, l_batch in ds:
# 	for i in range(4*16):
# 		im_plot.image(np.array(im_batch[i, :, :, :]))
# 		im_label.text(np.array(l_batch[i]))
# 		time.sleep(0.1)

# im = Image.open('data/raw/photos/Ac00hU6jxNijxYwVI7xC4Q.jpg')
# im = np.array(im)
# st.image(im)
# image_old = st.empty()
# image_ph = st.empty()
# data_file = "data/preprocessed/yelp_photos_train1.json"
# df = pd.read_json(data_file)

# # ds = build_dataset(rotate=True, batch_size=1)
# for i in range(4):
# 	im = Image.open(df.iloc[i]["image_path"])
# 	image_old.image(np.array(im) / 255)

# 	w, h = im.size
# 	res = min(h, w)
# 	h0, w0 = (h-res)//2, (w-res)//2
# 	im = im.resize((224, 224), box=(w0, h0, w0+res, h0+res))

# 	im = np.array(im) / 255.0
# 	image_ph.image(im)
# 	time.sleep(1)

