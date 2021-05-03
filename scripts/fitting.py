import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import _init_paths
from models import mesh_regressor, bird_model
from optimization import Pose_Fitter, Shape_Fitter, base_renderer
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
bird = bird_model(device=device)
renderer = Renderer(focal, (size/2, size/2), img_w=size, img_h=size, faces=bird.dd['F'])

rigidity = torch.ones([1, bird.V.shape[1]]).float().to(device)
rigidity[:, bird.dd['legs']] = 5
rigidity[:, bird.dd['feet']] = 5
rigidity[:, [3879, 3904]] = 8

silhouette_renderer = base_renderer(size=256, focal=2167, device=device)
regressor = mesh_regressor(device=device)
posefit = Pose_Fitter(model=bird, num_iters=300, part_iters=200,
                        prior_weight=10, mask_weight=1, use_mask=True, 
                        renderer=silhouette_renderer, device=device)
shapefit = Shape_Fitter(model=bird, num_iters=50, num_refine=100, kpts_weight=1, mask_weight=1, 
                         edge_w=2e4, lap_w=1e4, arap_w=0.05, ortho_w=2e2, sym_w=1e3,
                         step_size=0.025, renderer=silhouette_renderer, device=device, rigidity=rigidity)



# Regression to initialize
print('Reconstructing', dataset.species, 'from template')
print('Initializing ...')
with torch.no_grad():
	k = torch.tensor(keypoints).float().to(device)
	k[:, [9,15], :] = 0
	k = k.reshape(-1, 54)
	pose, bone, tran = regressor(k)
	pose = regressor.postprocess(pose)


# Optimize alignment
print('Optimizing alignment ...')
masks = torch.tensor(segmentations).clone().float().to(device)
kpts = torch.tensor(keypoints).clone().float().to(device)

pose_op, bone_op, tran_op, part_scales, record, model_mesh = posefit(pose, bone, tran, 
                                      focal_length=2167, camera_center=128,
                                      keypoints=kpts, masks=masks.squeeze(1), part_scales=None)




# Optimize deformation: mean
print('Optimizing mean deformation ...')
part_scales = part_scales.clone().float().to(device)
kpts[:, [2,3,4,5,6,7], :] = 0

pose_d, bone_d, tran_d, dv, losses, model_mesh_d, model_kpts_d = shapefit.stage_1(pose_op, bone_op, tran_op, parts=part_scales, 
                                      focal_length=2167, camera_center=128,
                                      keypoints=kpts, masks=masks.squeeze(1), lap_method='uniform')


# Optimize deformation: variation
print('Optimizing individual deformation ...')
n_basis = dataset.meta['num_basis']
pose_di, bone_di, tran_di, alpha, dvi, losses_i, model_mesh_di,  model_kpts_di \
 = shapefit.stage_2(pose_op, bone_op, tran_op, parts=part_scales, dv=dv,
                      focal_length=2167, camera_center=128, n_basis=n_basis,
                      keypoints=kpts, masks=masks.squeeze(1), lap_method='uniform')



# Render and save all results
print('Saving results ...')
output_dir = 'output_{}'.format(species_id)
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for i in range(len(images)):
	img = images[i]
	img_align, _ = renderer(model_mesh[i].detach().cpu().numpy(), np.eye(3), [0,0,0], img)
	img_deform, depth = renderer(model_mesh_di[i].detach().cpu(), np.eye(3), [0,0,0], img)
	img_out = np.hstack([img, img_align, img_deform])
	plt.imsave(output_dir + '/{:04d}.png'.format(i), img_out.astype(np.uint8))


