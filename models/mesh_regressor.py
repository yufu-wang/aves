import os
import torch
import torch.nn as nn
from utils.geometry import rot6d_to_rotmat

"""
A network trained with synthetic keypoints-pose to initialize the template model 
Parts of the code are taken from https://github.com/marcbadger/avian-mesh
"""
class MeshNet(nn.Module):
    def __init__(self):
        super(MeshNet, self).__init__()
        self.layer1 = self.make_layer(54, 512)
        self.layer2 = self.make_layer(512, 512)

        self.pose_layer = nn.Linear(512, 25*6)
        self.bone_layer = nn.Linear(512, 24)
        self.tran_layer = nn.Linear(512, 3)
        
    def make_layer(self, in_channel, out_channel):
        modules = [ nn.Linear(in_channel, out_channel),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU() ]

        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        pose = self.pose_layer(x)
        bone = self.bone_layer(x)
        tran = self.tran_layer(x)

        return pose, bone, tran

    def postprocess(self, p_est):
        """
        Convert 6d rotation to 9d rotation
        Input:
            p_est: pose_tran from forward()
            b_est: bone from forward()
        """
        pose_6d = p_est.contiguous()
        p_est_rot = rot6d_to_rotmat(pose_6d).view(-1, 25*9)

        return p_est_rot


def mesh_regressor(device='cpu'):
    this_dir = os.path.dirname(__file__)
    state_dict = torch.load(this_dir + '/reg.pth', map_location=device)['model_state_dict']

    model = MeshNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model

