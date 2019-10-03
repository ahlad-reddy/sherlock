import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import glob
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

import seaborn as sn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@st.cache(show_spinner=False)
def build_dataframe(json_file):
	df = pd.read_json(json_file)
	return df


def load_image(image_path, image_shape=(224, 224)):
	im = Image.open(image_path)
	w, h = im.size
	res = min(h, w)
	h0, w0 = (h-res)//2, (w-res)//2
	im = im.resize(image_shape, box=(w0, h0, w0+res, h0+res))
	return im

@st.cache
def show_image_batch(df, cluster_choice):
	image_batch = df[df["cluster"]==cluster_choice].sort_values(by="distance")
	st.image([load_image(image_batch.iloc[i]["image_path"]) for i in range(min(len(image_batch), 6))])


st.title("Visualizing and Labeling Clusters")
json_options = glob.glob('data/preprocessed/*.json')
json_select = st.sidebar.selectbox(label='JSON file to convert to dataframe', options=json_options)
df = build_dataframe(json_select)
cluster_choice = st.slider("Current Cluster", min_value=0, max_value=int(df["cluster"].max()), value=0)

show_image_batch(df, cluster_choice)


