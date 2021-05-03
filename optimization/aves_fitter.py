import torch

from .op_utils import transform_t, transform_p
from .losses import camera_fitting_loss, body_fitting_loss, mask_loss, prior_loss


class AVES_Fitter():
    def __init__(self, model, prior_weight=1, mask_weight=1,
                 beta_weight = 1,
                 step_size=1e-2,
                 global_iters=180,
                 pose_iters=300,
                 mask_iters=100,
                 renderer=None,
                 device=torch.device('cpu')):

        # Store options
        self.device = device
        self.step_size = step_size
        self.renderer = renderer

        self.global_iters = global_iters
        self.pose_iters = pose_iters
        self.mask_iters = mask_iters


        # Load Bird Mesh Model
        self.bird = model
        self.faces = self.bird.dd['F']
        self.prior_weight = prior_weight
        self.mask_weight = mask_weight
        self.beta_weight = beta_weight

        self.p_m = self.bird.p_m
        self.b_m = self.bird.b_m
        self.p_cov_in = self.bird.p_cov.inverse()
        self.b_cov_in = self.bird.b_cov.inverse()


    def render_silhouette(self, vertices):
        size = self.renderer.size
        faces = self.faces.clone().repeat(1,1,1)
        batch = vertices.shape[0]

        silhouette = torch.zeros([batch, size, size]).float().to(self.device)

        # right now we do it in loop to avoid GPU out of memory
        for i in range(batch):
            silhouette[i] = self.renderer(vertices[[i]], faces)[...,3]

        return silhouette

    def __call__(self, init_pose, init_bone, init_t, focal_length, camera_center, 
                keypoints, masks=None, favor_mask=True):


        # Number of views
        batch_size = init_pose.shape[0]

        # Get joint confidence
        keypoints_2d = keypoints[:, :, :2].clone().to(self.device)
        keypoints_conf = keypoints[:, :, -1].clone().to(self.device)

        # Body pose, global pose, translation, bone
        global_t = init_t.detach().clone().to(self.device)
        bone_length = init_bone.detach().clone().to(self.device)

        init_pose = transform_p(init_pose)
        global_orient = init_pose.detach().clone()[:, :3].to(self.device)
        body_pose = init_pose.detach().clone()[:, 3:].to(self.device)

        # Step 1: Optimize global parameters from initialization from regressor
        body_pose.requires_grad=False
        bone_length.requires_grad=False
        global_orient.requires_grad=True
        global_t.requires_grad = True

        body_opt_params = [global_orient, global_t]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.global_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

            loss = camera_fitting_loss(model_keypoints, None, None, focal_length, camera_center, 
                                       keypoints_2d, keypoints_conf)


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Step 2: Optimize all parameters
        beta = torch.zeros([batch_size, self.bird.B.shape[0]]).float().to(self.device)
        beta.requires_grad = True

        body_pose.requires_grad=True
        bone_length.requires_grad=True
        global_orient.requires_grad=True
        global_t.requires_grad = True

        body_opt_params = [beta, body_pose, bone_length, global_orient, global_t]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.pose_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, beta=beta, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

            loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, 
                                prior_weight=self.prior_weight, use_wing=True, use_init=False)

            loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in, self.prior_weight)
            loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in, self.prior_weight)
            loss_beta = self.beta_weight * beta.norm(dim=1).sum()

            loss = loss + loss_p + loss_b + loss_beta


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Step 3: Optimize all parameters with silhouette
        init_pose = body_pose.detach().clone()
        init_bone = bone_length.detach().clone()
        if favor_mask is True:  
            # Disable ambiguous keypoints (e.g. back, breast...) in favor of mask
            # This produces more realistic shapes, higher IOU, but lower PCK scores
            keypoints_conf[:, [2,3,4,5,6,7]] = 0
        
        for i in range(self.mask_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, beta=beta, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
            model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)

            loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, init_pose, init_bone, 
                                prior_weight=self.prior_weight, use_wing=True, use_init=True)

            loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)

            loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in, self.prior_weight)
            loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in, self.prior_weight)
            loss_beta = self.beta_weight * beta.norm(dim=1).sum()

            loss = loss + loss_mask + loss_p + loss_b + loss_beta


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Output
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().to('cpu')
        bone = bone_length.detach().to('cpu')
        global_t = global_t.detach().to('cpu')
        beta = beta.detach().to('cpu')
        
        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

        model_mesh = model_mesh.detach().to('cpu')
        model_keypoints = model_keypoints.detach().to('cpu')


        return pose, bone, global_t, beta, model_mesh, model_keypoints


