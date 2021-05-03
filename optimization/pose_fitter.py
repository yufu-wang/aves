import torch

from .op_utils import transform_t, transform_p
from .losses import camera_fitting_loss, body_fitting_loss, mask_loss, prior_loss


class Pose_Fitter():
    def __init__(self, model, prior_weight=1, mask_weight=1,
                 step_size=1e-2,
                 global_iters=180,
                 num_iters=50,
                 part_iters=0,
                 use_mask=False,
                 renderer=None,
                 device=torch.device('cpu')):

        # Store options
        self.device = device
        self.step_size = step_size
        self.global_iters = global_iters
        self.num_iters = num_iters
        self.part_iters = part_iters
        self.use_mask = use_mask
        if use_mask:
            self.renderer = renderer

        # Load Bird Mesh Model
        self.bird = model
        self.faces = torch.tensor(self.bird.dd['F'])
        self.prior_weight = prior_weight
        self.mask_weight = mask_weight

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
                keypoints, masks=None, part_scales=None):


        # Number of views
        batch_size = init_pose.shape[0]

        # Get joint confidence
        keypoints_2d = keypoints[:, :, :2].to(self.device)
        keypoints_conf = keypoints[:, :, -1].to(self.device)

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
                                    bone_length=bone_length, part_scales=part_scales, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

            loss = camera_fitting_loss(model_keypoints, None, None, focal_length, camera_center, 
                                       keypoints_2d, keypoints_conf)


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Step 2: Optimize all parameters
        body_pose.requires_grad=True
        bone_length.requires_grad=True
        global_orient.requires_grad=True
        global_t.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, part_scales=part_scales, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)

            loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, 
                                prior_weight=self.prior_weight, use_wing=True, use_init=False)

            loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in, self.prior_weight)
            loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in, self.prior_weight)

            loss = loss + loss_p + loss_b


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Step 3: Optimize all parameters with silhouette
        if self.use_mask:
            init_pose = body_pose.detach().clone()
            init_bone = bone_length.detach().clone()

            for i in range(25):
                bird_output = self.bird(global_pose=global_orient,
                                        body_pose=body_pose,
                                        bone_length=bone_length, part_scales=part_scales, pose2rot=True)

                global_txyz = transform_t(global_t)
                model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
                model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)

                loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                    body_pose, bone_length, init_pose, init_bone, 
                                    prior_weight=self.prior_weight, use_wing=True, use_init=True)

                loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in, self.prior_weight)
                loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in, self.prior_weight)

                loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)

                loss = loss + loss_p + loss_b + loss_mask


                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()


        # Step 4: Add in part scaling, and optimize all parameters with it
        if part_scales is None:
          part_scales = torch.tensor([1.0, 1.0]).float()
        part_scales.requires_grad = True
        part_record = []
        
        body_opt_params = [body_pose, bone_length, global_orient, global_t, part_scales]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        
        init_pose = body_pose.detach().clone()
        init_bone = bone_length.detach().clone()

        for i in range(self.part_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, 
                                    part_scales=part_scales, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
            model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)

            loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, init_pose, init_bone, 
                                prior_weight=self.prior_weight, use_wing=True, use_init=True)

            loss_p = prior_loss(body_pose, self.p_m, self.p_cov_in, self.prior_weight)
            loss_b = prior_loss(bone_length, self.b_m, self.b_cov_in, self.prior_weight)

            if self.use_mask:
                loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)
                loss = loss + loss_p + loss_b + loss_mask
            else:
                loss = loss + loss_p + loss_b


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

            part_record.append(part_scales.clone().detach().to('cpu').numpy())

        #### fix pose and camera, give each instance
        scales = part_scales.detach().clone().to(self.device)                      # category level
        part_scales = scales[:,None,None].repeat([1,batch_size,1]).to(self.device) # instance level

        part_scales.requires_grad=True
        body_pose.requires_grad=False
        bone_length.requires_grad=False
        global_orient.requires_grad=False
        global_t.requires_grad = False

        body_opt_params = [part_scales]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(100):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length, 
                                    part_scales=part_scales, pose2rot=True)

            global_txyz = transform_t(global_t)
            model_keypoints = bird_output['keypoints'] + global_txyz.unsqueeze(1)
            model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)

            loss = body_fitting_loss(model_keypoints, focal_length, camera_center, keypoints_2d, keypoints_conf, 
                                body_pose, bone_length, prior_weight=self.prior_weight, use_wing=True, use_init=False)

            if self.use_mask:
                loss_mask = mask_loss(self.render_silhouette(model_mesh), masks, self.mask_weight)
                loss = loss + loss_mask

            loss_reg = (part_scales - scales[:,None,None])**2
            loss += loss_reg.sum()

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        #### guard against divergence
        scales = scales.detach().cpu()
        part_scales = part_scales.detach().cpu()
        part_scales[1,part_scales[1]<0.2] = scales[1]
        part_scales[1,part_scales[1]>5.0] = scales[1]

        # Output
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().to('cpu')
        bone = bone_length.detach().to('cpu')
        global_t = global_t.detach().to('cpu')
        part_scales = part_scales.detach().to('cpu')

        model_mesh = bird_output['vertices'] + global_txyz.unsqueeze(1)
        model_mesh.to('cpu')
        
        return pose, bone, global_t, part_scales, part_record, model_mesh


