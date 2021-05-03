import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing
)

from .op_utils import transform_t
from .losses import body_fitting_loss, mask_loss
from .loss_arap import Arap_Loss


class Shape_Fitter():
    def __init__(self, model, kpts_weight=1, mask_weight=1, 
                 edge_w=1, lap_w=0.1, arap_w=0, ortho_w=0, sym_w=0,
                 step_size=1e-2,
                 num_iters=50,
                 num_refine=100,
                 renderer=None,
                 rigidity = None,
                 device=torch.device('cpu')):

        # Store options
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.num_refine = num_refine
        self.renderer = renderer

        # Load Bird Mesh Model
        self.bird = model
        self.faces = torch.tensor(self.bird.dd['F'])[None,:,:]
        self.kpts_weight = kpts_weight
        self.mask_weight = mask_weight
        self.edge_w = edge_w
        self.lap_w  = lap_w
        self.arap_w = arap_w
        self.ortho_w = ortho_w
        self.sym_w = sym_w
        self.rigidity = rigidity


    def render_silhouette(self, vertices):
        size = self.renderer.size
        faces = self.faces.clone()
        batch = vertices.shape[0]

        silhouette = torch.zeros([batch, size, size]).float().to(self.device)

        # right now we do it in loop to avoid GPU out of memory
        for i in range(batch):
            silhouette[i] = self.renderer(vertices[[i]], faces)[...,3]

        return silhouette


    def stage_1(self, init_pose, init_bone, init_t, parts, focal_length, camera_center, 
                keypoints, masks, lap_method='cot'):

        # Number of samples
        batch_size = init_pose.shape[0]

        # Get joint confidence
        keypoints_2d = keypoints[:, :, :2].to(self.device)
        keypoints_conf = keypoints[:, :, -1].to(self.device)

        # Body pose, global pose, translation, bone
        global_t = init_t.detach().clone().to(self.device)
        bone_length = init_bone.detach().clone().to(self.device)

        global_orient = init_pose.detach().clone()[:, :3].to(self.device)
        body_pose = init_pose.detach().clone()[:, 3:].to(self.device)

        # Fix articulation, bone, and translation 
        body_pose.requires_grad=False
        bone_length.requires_grad=False
        global_orient.requires_grad=False
        global_t.requires_grad = False

        # Initialize ARAP loss object
        bird_output = self.bird(global_pose=global_orient,
                                body_pose=body_pose,
                                bone_length=bone_length, part_scales=parts, pose2rot=True)
        global_txyz = transform_t(global_t)
        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        torch_mesh = Meshes(verts=model_mesh.to(self.device), 
                            faces=self.faces.repeat(batch_size,1,1).to(self.device))

        rigidity = self.rigidity.repeat(batch_size, 1).view(-1)
        arap = Arap_Loss(torch_mesh, device=self.device, vertex_w=rigidity)

        # record losses 
        losses = []

        # Step 1: Optimize shared per-vertex displacement dv
        d2v = self.bird.init_d2v()
        dim_dv = d2v.shape[-1]
        dv = torch.zeros([1, dim_dv, 3]).float().to(self.device)
        dv.requires_grad = True

        opt_params = [dv]
        optimizer = torch.optim.Adam(opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, part_scales=parts, d=dv, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
            model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
            dxyz = bird_output['dxyz']

            loss_kpt = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, 
                                prior_weight=1, use_wing=True, 
                                use_init=False) * self.kpts_weight

            loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)

            # mesh regularization 
            torch_dxyz = Meshes(verts=dxyz.to(self.device), 
                                faces=self.faces.to(self.device))
            torch_mesh = Meshes(verts=model_mesh.to(self.device), 
                                faces=self.faces.repeat(batch_size,1,1).to(self.device))

            loss_lap   = self.lap_w * mesh_laplacian_smoothing(torch_dxyz, method=lap_method)
            loss_edge  = self.edge_w * mesh_edge_loss(torch_dxyz)
            loss_arap  = self.arap_w * arap(torch_mesh)
            loss_sym   = self.sym_w * dxyz[:, self.bird.dd['mid'], 0].norm(dim=1, p=2).mean()
            
            
            # total loss
            loss_constraint = (loss_kpt + loss_mask )/batch_size
            loss_smooth = loss_lap + loss_edge + loss_arap + loss_sym

            loss =  loss_constraint + loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append([loss_kpt.item()/batch_size, loss_mask.item()/batch_size, 
                           loss_edge.item(), loss_lap.item(), loss_arap.item(), loss_sym.item(),
                           0])


        # Output
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().to('cpu')
        bone = bone_length.detach().to('cpu')
        global_t = global_t.detach().to('cpu')
        dv = dv.detach().to('cpu')

        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

        model_mesh = model_mesh.detach().to('cpu')
        model_keypoints = model_keypoints.detach().to('cpu')
        

        return pose, bone, global_t, dv, losses, model_mesh, model_keypoints


    def stage_2(self, init_pose, init_bone, init_t, parts, focal_length, camera_center, 
                keypoints, masks, dv, n_basis=3, lap_method='cot'):
        # Number of samples
        batch_size = init_pose.shape[0]

        # Get joint confidence
        keypoints_2d = keypoints[:, :, :2].to(self.device)
        keypoints_conf = keypoints[:, :, -1].to(self.device)

        # Body pose, global pose, translation, bone
        global_t = init_t.detach().clone().to(self.device)
        bone_length = init_bone.detach().clone().to(self.device)

        global_orient = init_pose.detach().clone()[:, :3].to(self.device)
        body_pose = init_pose.detach().clone()[:, 3:].to(self.device)
        dv = dv.detach().clone().to(self.device)

        # Fix articulation, bone, and translation 
        body_pose.requires_grad=False
        bone_length.requires_grad=False
        global_orient.requires_grad=False
        global_t.requires_grad = False

        # Initialize ARAP loss object
        bird_output = self.bird(global_pose=global_orient,
                                body_pose=body_pose,
                                bone_length=bone_length, part_scales=parts, pose2rot=True)
        global_txyz = transform_t(global_t)
        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        torch_mesh = Meshes(verts=model_mesh.to(self.device), 
                            faces=self.faces.repeat(batch_size,1,1).to(self.device))

        rigidity = self.rigidity.repeat(batch_size, 1).view(-1)
        arap = Arap_Loss(torch_mesh, device=self.device, vertex_w=rigidity)

        # record losses 
        losses = []

        # Step 2: Optimize basis and coefficient for each input
        d2v = self.bird.init_d2v()
        dim_dv = d2v.shape[-1]
        
        const = torch.eye(n_basis)[None].repeat(3, 1, 1).float().to(self.device)
        alpha = torch.zeros([batch_size, n_basis]).float().to(self.device)
        dvi = torch.zeros([n_basis, dim_dv, 3]).float().to(self.device)
        for i in range(n_basis):
            dvi[i, i] = 1.

        alpha.requires_grad = True
        dvi.requires_grad = True
        dv.requires_grad = False

        opt_params = [dvi, alpha]
        optimizer = torch.optim.Adam(opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_refine):
            dva = dv + torch.einsum('bn,nvj->bvj', alpha, dvi)

            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, part_scales=parts, d=dva, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
            model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
            dxyz = bird_output['dxyz']

            loss_kpt = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, 
                                prior_weight=1, use_wing=True, 
                                use_init=False) * self.kpts_weight

            loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)

            # mesh regularization 
            torch_dxyz = Meshes(verts=dxyz.to(self.device), 
                                faces=self.faces.repeat(batch_size,1,1).to(self.device))
            torch_mesh = Meshes(verts=model_mesh.to(self.device), 
                                faces=self.faces.repeat(batch_size,1,1).to(self.device))

            loss_lap   = self.lap_w * mesh_laplacian_smoothing(torch_dxyz, method=lap_method)
            loss_edge  = self.edge_w * mesh_edge_loss(torch_dxyz)
            loss_arap  = self.arap_w * arap(torch_mesh)
            loss_sym   = self.sym_w * dxyz[:, self.bird.dd['mid'], 0].norm(dim=1, p=2).mean()

            # orthogonomal loss
            p = dvi.permute(2,0,1) @ dvi.permute(2,1,0)
            loss_ortho = self.ortho_w * F.smooth_l1_loss(p, const)


            # total loss
            loss_constraint = (loss_kpt + loss_mask )/batch_size
            loss_smooth = loss_lap + loss_edge + loss_arap + loss_sym

            loss =  loss_constraint + loss_smooth + loss_ortho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append([loss_kpt.item()/batch_size, loss_mask.item()/batch_size, 
                           loss_edge.item(), loss_lap.item(), loss_arap.item(), loss_sym.item(),
                           loss_ortho.item()])



        # Output
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().to('cpu')
        bone = bone_length.detach().to('cpu')
        global_t = global_t.detach().to('cpu')
        alpha = alpha.detach().to('cpu')
        dvi = dvi.detach().to('cpu')
       
        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

        model_mesh = model_mesh.detach().to('cpu')
        model_keypoints = model_keypoints.detach().to('cpu')
        

        return pose, bone, global_t, alpha, dvi, losses, model_mesh, model_keypoints


