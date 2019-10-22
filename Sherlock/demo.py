import glob
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def main():
	st.sidebar.title("Sherlock")
	app_mode = st.sidebar.selectbox("Choose the app mode", 
		["Instructions", "Dataset View", "Cluster View"])

	json_options = glob.glob('data/processed/*.json')
	json_select = st.sidebar.selectbox('Select the precomputed JSON file', json_options)

	if app_mode == "Instructions":
		instructions_ui()
	elif json_select:
		if app_mode == "Dataset View":
			dataset_ui(json_select)
		elif app_mode == "Cluster View":
			cluster_ui(json_select)
	else:
		st.sidebar.error("Follow the instructions to compute the JSON file")


def instructions_ui():
	st.markdown(open('Sherlock/instructions.md').read())


def dataset_ui(json_select):
	df = build_dataframe(json_select)
	n_images = st.sidebar.slider("Number of images to show", min_value=25, max_value=100, step=25)
	for cluster in range(df["cluster"].max()+1):
		df_subset = df[df["cluster"]==cluster].sort_values(by="distance")
		label = df_subset.iloc[0].assigned_label or "None"
		st.subheader("Cluster {}, Count: {}, Label: {}".format(cluster, len(df_subset), label))
		st.image([load_image(row["image_path"]) for i, row in df_subset[:n_images].iterrows()], width=55)


def cluster_ui(json_select):
	df = build_dataframe(json_select)

	cluster_choice = st.sidebar.slider("Select Cluster", min_value=0, max_value=int(df["cluster"].max()), value=0)
	df_subset = df[df["cluster"]==cluster_choice].sort_values(by="distance")

	cluster_subheader = st.empty()
	cluster_label = df.loc[df_subset.index[0], "assigned_label"] or "None"
	cluster_subheader.subheader("Cluster {}, Label: {}".format(cluster_choice, cluster_label.replace('_', ' ').capitalize()))

	st.image([load_image(row["image_path"]) for i, row in df_subset[:6].iterrows()], width=219)

	label = st.text_input('Input Label for Cluster {}'.format(cluster_choice))
	if label != '':
		df.loc[df_subset.index, "assigned_label"] = label.lower().replace(' ', '_')
		cluster_subheader.subheader("Cluster {}, Label: {}".format(cluster_choice, label.replace('_', ' ').capitalize()))
		df.to_json(json_select)


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


if __name__=="__main__":
	main()

