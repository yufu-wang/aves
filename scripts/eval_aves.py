import os
import argparse
import numpy as np
import torch
import logging

import _init_paths
import constants
from models import mesh_regressor, AVES
from optimization import base_renderer, AVES_Fitter
from utils.renderer import Renderer
from utils.cub_dataset import CUB_Dataset
from utils.geometry import perspective_projection
from utils.evaluation import evaluate_iou, evaluate_pck, average_meter


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='Device to use')
args = parser.parse_args()

# dataset root
root = 'data/CUB'
device = args.device
logging.basicConfig(level=logging.INFO, format='%(message)s', 
				    handlers=[
			        logging.FileHandler("eval.log", mode='w'),
			        logging.StreamHandler()]
			        )


# objects
size = 256
focal = 2167
aves = AVES(device=device, high_res=False)
renderer = Renderer(focal, (size/2, size/2), img_w=size, img_h=size, faces=aves.dd['F'])

silhouette_renderer = base_renderer(size=256, focal=2167, device=device)
regressor = mesh_regressor(device=device)
avesfit = AVES_Fitter(model=aves, prior_weight=10, mask_weight=1, beta_weight=150, 
                       global_iters=180, pose_iters=300, mask_iters=100,
                       renderer=silhouette_renderer, device=device)


# Reconstructing each species
num_species = len(constants.Species)
PCK05 = average_meter()
PCK10 = average_meter()
IOU = average_meter()

for i in range(num_species):
	dataset = CUB_Dataset(root, species_id=i)
	images = dataset.images
	keypoints = dataset.keypoints
	segmentations = dataset.segmentations


	# Regression to initialize
	logging.info('Reconstructing ' + dataset.species + ' using AVES')
	with torch.no_grad():
		k = torch.tensor(keypoints).float().to(device)
		k[:, [9,15], :] = 0
		k = k.reshape(-1, 54)
		pose, bone, tran = regressor(k)
		pose = regressor.postprocess(pose)


	# Optimize alignment
	masks = torch.tensor(segmentations).clone().float().to(device)
	kpts = torch.tensor(keypoints).clone().float().to(device)

	pose_op, bone_op, tran_op, beta, model_mesh, model_kpts = avesfit(pose, bone, tran, 
	                                      focal_length=2167, camera_center=128,
	                                      keypoints=kpts, masks=masks.squeeze(1), favor_mask=True)



	# Render and save all results
	proj_masks = []
	obj_sizes = []
	for j in range(len(images)):
		img = images[j]
		img_aves, depth = renderer(model_mesh[j].detach().cpu().numpy(), np.eye(3), [0,0,0], img)

		proj_mask = torch.tensor(depth) > 0
		proj_masks.append(proj_mask)

		mask = masks[j].cpu()
		ind = torch.nonzero(mask>1e-3)
		h = ind[:,0].max() - ind[:,0].min()
		w = ind[:,1].max() - ind[:,1].min()
		obj_sizes.append(max(h, w))



	# Evaluate fitting quality
	proj_masks = torch.stack(proj_masks)
	iou = evaluate_iou(proj_masks, masks.cpu())

	kpts_gt = torch.tensor(keypoints).float().cpu()
	kpts_2d = perspective_projection(model_kpts, None, None, 2167, 128)
	pck05, pck10 = evaluate_pck(kpts_2d, kpts_gt, size=obj_sizes)

	species_pck05 = np.mean(pck05)
	species_pck10 = np.mean(pck10)
	species_iou = np.mean(iou)

	PCK05.collect(species_pck05)
	PCK10.collect(species_pck10)
	IOU.collect(species_iou)

	logging.info('%s %.4f', 'PCK05:', species_pck05)
	logging.info('%s %.4f', 'PCK10:', species_pck10)
	logging.info('%s %.4f', 'IOU:', species_iou)


logging.info('%s %.4f', 'Over all PCK05:', PCK05.report())
logging.info('%s %.4f', 'Over all PCK10:', PCK10.report())
logging.info('%s %.4f', 'Over all IOU:', IOU.report())




