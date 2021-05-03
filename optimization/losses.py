import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle

from utils.geometry import perspective_projection
import constants

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

        
def camera_fitting_loss(model_keypoints, rotation, camera_t, focal_length, camera_center, 
                        keypoints_2d, keypoints_conf, distortion=None):


    # Project model keypoints
    projected_keypoints = perspective_projection(model_keypoints, rotation, camera_t,
                                                focal_length, camera_center, distortion)
    
    # Disable Bill tip, Tail tip, Wing Tips, and Feet
    keypoints_conf = keypoints_conf.detach().clone()
    keypoints_conf[:, [9,15,16,17,10,12]] = 0

    # Weighted robust reprojection loss
    sigma = 50
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf ** 2) * reprojection_error.sum(dim=-1)

    total_loss = reprojection_loss.sum(dim=-1)

    return total_loss.sum()


def body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                    body_pose, bone_length, pose_init=None, bone_init=None, sigma=100,
                    prior_weight=1, use_wing=False, use_init=False):

    device = body_pose.device
    
    # Project model keypoints
    projected_keypoints = perspective_projection(model_keypoints, None, None,
                                                 focal_length, camera_center)

    # Disable Wing Tips
    keypoints_conf = keypoints_conf.detach().clone()
    if use_wing==False:
        keypoints_conf[:, [16,17]] = 0

    # Weighted robust reprojection loss
    reprojection_error = gmof(projected_keypoints - keypoints_2d, sigma)
    reprojection_loss = (keypoints_conf ** 2) * reprojection_error.sum(dim=-1)


    # For now
    if use_init:
        init_loss = (body_pose - pose_init).abs().sum() + (bone_length - bone_init).abs().sum()
        init_loss = init_loss * prior_weight

        total_loss = reprojection_loss.sum(dim=-1) + init_loss.sum()
    else:    
        total_loss = reprojection_loss.sum(dim=-1) 

    
    return total_loss.sum()


def mask_loss(proj_masks, masks, mask_weight):

    # L1 mask loss
    total_loss = F.smooth_l1_loss(proj_masks, masks, reduction='none').sum(dim=[1,2])
    total_loss = mask_weight * total_loss
    
    return total_loss.sum()


def prior_loss(p, mean, cov_in, weight):
    # Squared Mahalanobis distance
    pm = p - mean

    dis = pm @ cov_in @ pm.t()
    dis = weight * torch.diag(dis).sum()

    return dis

def symmetric_loss(dxyz, body_lr):
    dv_l = dxyz[:, body_lr[:,0], :]
    dv_r = dxyz[:, body_lr[:,1], :]
    dv_r[:,:,0] *= -1

    loss = (dv_l - dv_r).norm(dim=2, p=2).mean()

    return loss



