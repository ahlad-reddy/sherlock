# Sherlock: Semi-supervised Image Labeling
Sherlock is a set of tools that leverage unsupervised learning techniques to make image labeling more efficient. Currently it contains an implementation pipeline of [RotNet](https://arxiv.org/pdf/1803.07728.pdf) as well as an interface that utilizes K-Means to cluster and visualize images for batch labeling of data. The experiments for this project were done primarily with the Yelp photos dataset (installation instructions below).

The presentation slides to the project are available [here](bit.ly/sherlock-ml).

## Prerequisites

- Python 3.6
- NVIDIA GPU (Recommended)

#### Dependencies

- Tensorflow 1.14
- Numpy
- Pandas
- Scikit-learn
- Pillow
- Streamlit

#### Installation
To get started, first clone the repository and install the dependencies. It is recommended to use a virtual environment, or an AWS instance built with the Deep Learning AMI.
```shell
git clone https://github.com/ahlad-reddy/sherlock
cd sherlock
pip install -r requirements.txt
```

## Downloading Data

The Yelp Photos dataset can be downloaded from this [link](https://www.yelp.com/dataset/download) after accepting the terms of use. 

## Rotation Pre-Training
To train a model on the unsupervised rotation training, run the following:

```shell
python Sherlock/rotation_network.py \
	--res 224 \
	--lr 0.0001 \
	--batch_size 16 \
	--epochs 10 \
	--model None \
	--save
```

## Labeling Interface
To run the labeling interface, features and clusters must first be precomputed. Features and clusters will be saved in a JSON file that can then be loaded by the Streamlit application. To run the script enter:

```shell
python Sherlock/cluster.py \
	--res 224 \
	--model path/to/model \
	--save_path save/directory \
	--n_clusters 200 \
	--n_datapoints 5000 \
	--seed 2019
```

Once the clusters have been calculated, you can run the Streamlit app.

```shell
streamlit run Sherlock/demo.py
```

Additional instructions to use the interface can be found in the application.
