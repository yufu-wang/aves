import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import _init_paths
from models import mesh_regressor, AVES
from optimization import base_renderer, AVES_Fitter
from utils.renderer import Renderer
from utils.cub_dataset import CUB_Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='Device to use')
parser.add_argument('--species_id', default=2, type=int, help='Species to run reconstruction')
args = parser.parse_args()
device = args.device
species_id = args.species_id

# dataset
root = 'data/CUB'
dataset = CUB_Dataset(root, species_id=species_id)
images = dataset.images
keypoints = dataset.keypoints
segmentations = dataset.segmentations

# objects
size = 256
focal = 2167
aves = AVES(device=device, high_res=True)
renderer = Renderer(focal, (size/2, size/2), img_w=size, img_h=size, faces=aves.dd['F'])


silhouette_renderer = base_renderer(size=256, focal=2167, device=device)
regressor = mesh_regressor(device=device)
avesfit = AVES_Fitter(model=aves, prior_weight=10, mask_weight=1, beta_weight=150, 
                       global_iters=180, pose_iters=300, mask_iters=100,
                       renderer=silhouette_renderer, device=device)

# Regression to initialize
print('Reconstructing', dataset.species, 'using AVES')
print('Initializing ...')
with torch.no_grad():
	k = torch.tensor(keypoints).float().to(device)
	k[:, [9,15], :] = 0
	k = k.reshape(-1, 54)
	pose, bone, tran = regressor(k)
	pose = regressor.postprocess(pose)


# Optimize alignment
print('Optimizing AVES ...')
masks = torch.tensor(segmentations).clone().float().to(device)
kpts = torch.tensor(keypoints).clone().float().to(device)

pose_op, bone_op, tran_op, beta, model_mesh, model_kpts = avesfit(pose, bone, tran, 
                                      focal_length=2167, camera_center=128,
                                      keypoints=kpts, masks=masks.squeeze(1), favor_mask=True)



# Render and save all results
print('Saving results ...')
output_dir = 'output_aves_{}'.format(species_id)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)


for i in range(len(images)):
	img = images[i]
	img_aves, _ = renderer(model_mesh[i].detach().cpu().numpy(), np.eye(3), [0,0,0], img)
	img_out = np.hstack([img, img_aves])
	plt.imsave(output_dir + '/{:04d}.png'.format(i), img_out.astype(np.uint8))




