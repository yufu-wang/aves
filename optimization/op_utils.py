import cv2
import torch


def transform_t(tran, scale=18, bias=180):
    """ From tran~[0,1] to tran_xyz of real unit.
        Simply intermediate layer. optimization objective is still tran.
    """
    tran_xyz = tran.clone()
    tran_xyz[:, 1] = tran_xyz[:, 1] - 1
    tran_xyz[:, 2] = tran_xyz[:, 2]*scale + bias

    return tran_xyz

def transform_p(pose):
    """ From 9d rot pose to 3d axis-angle pose
        Fast enough for right now.
    """
    batch_size = len(pose)
    
    pose = pose.to('cpu')
    pose = pose.detach().clone()
    pose = pose.reshape(batch_size, -1, 3, 3)
    new_pose = torch.zeros([batch_size, pose.shape[1]*3]).float()
    
    for i in range(batch_size):
        for j in range(pose.shape[1]):
            R = pose[i, j]
            aa, _ = cv2.Rodrigues(R.numpy())
            new_pose[i, 3*j:3*(j+1)] = torch.tensor(aa).squeeze()
        
    return new_pose