# Sherlock: Image Labeling Interface

This application allows you to efficiently label images for your machine learning project. 
To get started, make sure you have precomputed features and clusters for your data. 
Features and clusters will be saved in a JSON file that can then be loaded by this application. 
To run the script enter (with your desired options):

	python Sherlock/cluster.py \
		--res 224 \
		--model path/to/model \
		--save_path data/processed/name_of_file.json \
		--n_clusters 200 \
		--n_datapoints 5000 \
		--seed 2019

When your data has been precomputed, select the saved JSON file from the sidebar on the left.
There are two view modes in this app:

1. Dataset View: Visualizes all the clusters in your dataset, their currently assigned labels, and example images.
2. Cluster View: Shows images from a single batch and allows you to apply a label.

Any labels you assign will be saved in the JSON document.