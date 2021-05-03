import os
import json
import torch
from .LBS import LBS

class bird_model():
    def __init__(self, device=torch.device('cpu'), tailspread=None):

        self.device = device

        # read in bird_mesh from the same dir
        this_dir = os.path.dirname(__file__)
        with open(this_dir + '/bird_template.json', 'r') as infile:
            dd = json.load(infile)

        self.dd = dd
        self.kintree_table = torch.tensor(dd['kintree_table']).to(device)
        self.parents = self.kintree_table[0]
        self.weights = torch.tensor(dd['weights']).to(device)
        self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)

        self.J = torch.tensor(dd['J']).unsqueeze(0).to(device)
        self.V = torch.tensor(dd['V']).unsqueeze(0).to(device)
        self.LBS = LBS(self.J, self.parents, self.weights)
        
        prior = torch.load(this_dir + '/pose_bone_prior.pth')
        self.p_m = prior['p_m'].to(device)
        self.b_m = prior['b_m'].to(device)
        self.p_cov = prior['p_cov'].to(device)
        self.b_cov = prior['b_cov'].to(device)

        self.init_parts(dd, tailspread, device)

        
    def __call__(self, global_pose, body_pose, bone_length, 
                scale=1, d=None, part_scales=None, pose2rot=True):
        batch_size = global_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale

        # scale beak if applied
        if part_scales is not None:
            V = self.scale_parts(V, part_scales)

        # per-vertex deformation
        if d is not None:
            V, dxyz = self.deform(V, d)

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size,1]).to(self.device), bone_length], dim=1)
        pose = torch.cat([global_pose, body_pose], dim=1)

        # LBS          
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot) 

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = torch.einsum('bni,kn->bki', verts, self.vert2kpt)

        # Final output after articulation
        if d is not None:
            output = {'vertices': verts,
                      'dxyz': dxyz,
                      'keypoints': keypoints}
        else:
            output = {'vertices': verts,
                      'keypoints': keypoints}

        return output


    def init_parts(self, dd, tailspread, device):
        self.beak = torch.tensor(dd['beak']).to(device)
        self.lwing = torch.tensor(dd['lwing']).to(device)
        self.rwing = torch.tensor(dd['rwing']).to(device)
        self.tail  = torch.tensor(dd['tail']).to(device)
        self.tailidx = dd['tail_index']
        
        self.d2v = None

        if tailspread is not None:
            tailkey = torch.tensor(dd['tail_closed']).unsqueeze(0).to(device)
            self.V[:,self.tail] = tailspread * self.V[:,self.tail] \
                                + (1-tailspread) * tailkey[:,self.tail]


    def scale_parts(self, V, part_scales):
        beak_scale = part_scales[0]
        tail_scale = part_scales[1]
        V = self.scale_beak(V, beak_scale)
        V = self.scale_tail(V, tail_scale)
        return V


    def scale_tail(self, V, scale):
        '''
        Input:
            V [bn, V, 3] tensor for batched vertices
            scale [bn, 1] or [1] tensor for beak scale along y and z
                    requires grad during optimization
        '''
        V = V.clone()
        for i, idx in enumerate(self.tailidx):
            vertex = V[:, idx].clone()
            baseid = vertex[:,:,1].argmax(dim=1)[0]
            base = vertex[:,[baseid],:]

            vertex = vertex - base
            vertex[:,:,1] *= scale
            vertex = vertex + base

            V[:, idx] = vertex
        return V


    def scale_beak(self, V, scale):
        '''
        Input:
            V [bn, V, 3] tensor for batched vertices
            scale [bn, 1] or [1] tensor for beak scale along y
                    requires grad during optimization
        '''

        R, t = self.beak_coordinate(V)
        Rbw = R.transpose(1,2)
        tbw = torch.einsum('bij,bvj->bvi', Rbw, -t)
        
        x = V[:, self.beak].clone()
        new_V = V.clone()

        xb = torch.einsum('bij,bvj->bvi', Rbw, x) + tbw
        xb[:,:,1] *= scale
        xw = torch.einsum('bij,bvj->bvi', R, xb) + t
        new_V[:, self.beak] = xw

        return new_V


    def deform(self, V, d):
        '''
        Deform mesh V per vertex
        Input:
            V (bn, 3940, 3): mesh vertices
            d (1, 740, 3) or (bn, 740, 3): body deformation vector, requires grad
        Params:
            d2v[0] (1, 3940, 740): mapping from d to dv (x coordinate)
            d2v[1] (1, 3940, 740): mapping from d to dv (y,z coordinate)
            dxyz (1, 3940, 3) or (bn, 3940, 3): complete deformation vector
        '''
        if self.d2v is None:
            self.d2v = self.init_d2v()
            self.d2v = self.d2v.to(self.device)

        dx  = self.d2v[0] @ d[:,:,[0]]
        dyz = self.d2v[1] @ d[:,:,1:]
        dxyz = torch.cat([dx, dyz], dim=-1)

        new_V = V + dxyz

        # No deformation on feathers (which are thin plates insert on the main mesh)
        # We simpliy 'regress' them, by attaching them to points on the body
        new_V[:, self.lwing, :] += dxyz[:, [3879]]
        new_V[:, self.rwing, :] += dxyz[:, [3904]]
        new_V[:, self.tail,  :] += (dxyz[:, [3145]]+dxyz[:, [3146]])/2

        return new_V, dxyz


    def init_d2v(self):
        """
        Function to map deformation vector d to dv
        change this function if you want dv to include feather vertices
        (right now dv does not include feather)
        """
        
        feather = self.dd['feather']
        lr = self.dd['lr']
        mid = self.dd['mid']

        # remove feather idx from lr
        body_lr = []
        for (l, r) in lr:
            if l in feather or r in feather:
                continue
            else:
                body_lr.append([l, r])

        # d2vert
        body_lr = torch.tensor(body_lr)
        mid = torch.tensor(mid)
        num_v = len(self.dd['V'])
        num_d = len(body_lr) + len(mid)

        d2vert = torch.zeros([num_v, num_d]).float()
        for i in range(len(body_lr)):
            ml, mr = body_lr[i]
            d2vert[ml, i] = 1
            d2vert[mr, i] = 1
        for i in range(len(mid)):
            m = mid[i]
            d2vert[m, i+len(body_lr)] = 1
            
        # d2vert_inv
        d2vert_inv = torch.zeros([num_v, num_d]).float()
        for i in range(len(body_lr)):
            ml, mr = body_lr[i]
            d2vert_inv[ml, i] = 1
            d2vert_inv[mr, i] = -1
        for i in range(len(mid)):
            m = mid[i]
            d2vert_inv[m, i+len(body_lr)] = 1


        d2v = torch.stack([d2vert_inv, d2vert])
        d2v = d2v[:, None, :, :]

        return d2v


    def beak_coordinate(self, V):
        """
        Input:
            V [bn, v, 3] tensor for batched vertices
        Return:
            R [bn, 3, 3] beak coordinate orientation
            t [bn, 1, 3] beak coordinate origin 
        """
    
        y = V[:, [3850, 3853]].mean(axis=1) - V[:, 3928]
        x = V[:, 3831] - V[:, 3481]
        x = x/x.norm(dim=1, keepdim=True)
        y = y/y.norm(dim=1, keepdim=True)
        z = torch.cross(x,y)

        # new coordinate
        R = torch.stack([x,y,z], axis=2).clone().detach()
        tx = V[:, 3928, 0]
        tz = V[:, 3928, 2]
        ty = V[:, [3492, 3842], 1].mean(axis=1)
        t = torch.stack([tx,ty,tz], axis=1).unsqueeze(1).clone().detach()

        return R, t

