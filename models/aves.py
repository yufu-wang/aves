import os
import json
import torch
from .LBS import LBS

class AVES():
    def __init__(self, device=torch.device('cpu'), high_res=False):

        self.device = device

        # read in bird_mesh from the same dir
        this_dir = os.path.dirname(__file__)
        if high_res is False:
            dfile = this_dir + '/aves.pt'
        else:
            dfile = this_dir + '/aves_high_res.pt'

        dd = torch.load(dfile)

        # kinematic tree, and map to keypoints from vertices
        self.dd = dd
        self.kintree_table = dd['kintree_table'].to(device)
        self.parents = self.kintree_table[0]
        self.weights = dd['weights'].to(device)
        self.vert2kpt = dd['vert2kpt'].to(device)

        # mean shape and default joints
        self.V = dd['V'].unsqueeze(0).to(device)
        self.J = dd['J'].unsqueeze(0).to(device)
        self.LBS = LBS(self.J, self.parents, self.weights)
        
        # pose and bone prior
        self.p_m = dd['pose_mean'].to(device)
        self.b_m = dd['bone_mean'].to(device)
        self.p_cov = dd['pose_cov'].to(device)
        self.b_cov = dd['bone_cov'].to(device)

        # standardized blend shape basis
        B = dd['Beta'].to(device)
        sigma = dd['Beta_sigma'].to(device)
        self.B = B * sigma[:,None,None]

        # PCA coefficient that is optimized to match the original template shape
        ### so in the __call__ funciton, if beta is set to self.beta_original,
        ### it will return the template shape from ECCV2020 (marcbadger/avian-mesh). 
        self.beta_original = dd['beta_original'].to(device)
        
        
    def __call__(self, global_pose, body_pose, bone_length, 
                scale=1, beta=None, pose2rot=True, d=None, part_scales=None):
        '''
        Input:
            global_pose [bn, 3] tensor for batched global_pose on root joint
            body_pose   [bn, 72] tensor for batched body pose
            bone_length [bn, 24] tensor for bone length; the bone variable 
                                 captures non-rigid joint articulation in this model

            beta [bn, 15] shape PCA coefficients
            If beta is None, it will return the mean shape
            If beta is self.beta_original, it will return the orignial tempalte shape

        '''
        batch_size = global_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale
        J = self.J.repeat([batch_size, 1, 1]) * scale

        # multi-bird shape space
        if beta is not None:
            V = V + torch.einsum('bk, kmn->bmn', beta, self.B)

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size,1]).to(self.device), bone_length], dim=1)
        pose = torch.cat([global_pose, body_pose], dim=1)

        # LBS          
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot) 

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = torch.einsum('bni,kn->bki', verts, self.vert2kpt)


        # Final output after articulation
        output = {'vertices': verts,
                  'keypoints': keypoints}


        return output
