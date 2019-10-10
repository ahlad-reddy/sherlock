import glob
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

@st.cache(show_spinner=False, ignore_hash=True)
def build_dataframe(json_file):
	df = pd.read_json(json_file)
	return df


def load_image(image_path, image_shape=(224, 224)):
	im = Image.open(image_path)
	w, h = im.size
	res = min(h, w)
	h0, w0 = (h-res)//2, (w-res)//2
	im = im.resize(image_shape, box=(w0, h0, w0+res, h0+res))
	im = im.convert('RGB')
	return im


def show_image_batch(df, cluster_choice):
	image_batch = df[df["cluster"]==cluster_choice].sort_values(by="distance")[:6]
	st.image([load_image(row["image_path"]) for i, row in image_batch.iterrows()], width=219)
	return image_batch


def main():
	st.sidebar.title("Visualizing and Labeling Clusters")
	json_options = glob.glob('data/preprocessed/food101*.json')
	json_select = st.sidebar.selectbox(label='JSON file to convert to dataframe', options=json_options)
	df = build_dataframe(json_select)

	cluster_choice = st.slider("Select Cluster", min_value=0, max_value=int(df["cluster"].max()), value=0)
	cluster_subheader = st.empty()
	image_batch = show_image_batch(df, cluster_choice)

	cluster_label = df.loc[image_batch.index[0], "assigned_label"] or "None"
	cluster_subheader.subheader("Cluster {}, Label: {}".format(cluster_choice, cluster_label.replace('_', ' ').capitalize()))

	label = st.sidebar.text_input('Input Label')
	if st.sidebar.button("Assign"):
		df.loc[image_batch.index, "assigned_label"] = label.lower().replace(' ', '_')
		cluster_subheader.subheader("Cluster {}, Label: {}".format(cluster_choice, label.replace('_', ' ').capitalize()))
		df.to_json(json_select)


if __name__=="__main__":
	main()


