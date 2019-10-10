# Sherlock: Semi-supervised Image Labeling
Sherlock is a set of tools that leverage unsupervised learning techniques to make image labeling more efficient. Currently it contains an implementation pipeline of [RotNet](https://arxiv.org/pdf/1803.07728.pdf) as well as an interface that utilizes K-Means to cluster and visualize images for batch labeling of data. 

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

To download the Food 101 dataset, use the following command.

```shell
wget -c -P data/raw/ http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -zxvf data/raw/food-101.tar.gz
```

## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
