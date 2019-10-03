import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
np.random.seed(1000)

meta_file = "data/raw/photo.json"
image_dir = "data/raw/photos"
save_path = "data/preprocessed/yelp_photos_{}.json"

metadata = open(meta_file).readlines()
n = len(metadata)
split_idx = n * np.array([0, 0.16, 0.32, 0.48, 0.64, 0.80, 1.0])
split_idx = split_idx.astype('int32')
idx = np.arange(n)
np.random.shuffle(idx)

for i, split in enumerate(tqdm(['train1', 'train2', 'train3', 'train4', 'train5', 'test'])):

	json_data = []
	for j in tqdm(idx[split_idx[i]:split_idx[i+1]]):
		data = json.loads(metadata[j])
		image_path = os.path.join(image_dir, data["photo_id"] + '.jpg')
		image = Image.open(image_path)
		w, h = image.size

		json_data.append({
			'image_path': image_path, 
			"height": h, 
			"width": w, 
			"feature_vector": [], 
			"label": None
		})

	with open(save_path.format(split), mode='w') as json_file:
		json.dump(json_data, json_file)

