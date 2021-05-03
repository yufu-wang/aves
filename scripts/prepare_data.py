import os
import argparse
from glob import glob
import numpy as np
from PIL import Image

import _init_paths
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./', help='Downloaded CUB dataset')
args = parser.parse_args()

root = args.root
img_dir = root + 'CUB_200_2011/CUB_200_2011/images'

for i, category in enumerate(Cub_category):
	input_folder = img_dir + '/' + category
	output_folder = 'data/CUB/images/' +  category 
	boxes = np.load('data/CUB/boxes/' + category + '.npy')
	
	os.makedirs(output_folder, exist_ok=True)

	species = Species[i]
	samples = Species_meta[species]['samples']

	imgfiles = sorted(glob(input_folder + '/*.jpg'))
	imgfiles = [imgfiles[k-1] for k in samples]


	for imgfile, box, sample in zip(imgfiles, boxes, samples):
		newfile = output_folder + '/{:04d}.png'.format(sample)

		img = Image.open(imgfile).crop([box[0], box[1], box[0]+box[2], box[1]+box[3]])
		img = img.resize([256, 256])
		img.save(newfile)
		



