"this formulation considers contact forces from the wall to the cuboid"
from isaacgym import gymapi
from MFR_benchmark.utils import get_assets_dir

import numpy as np
import pickle as pkl

import torch
import time
import copy
import yaml
import pathlib
from functools import partial

import time
import pytorch_volumetric as pv
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from torch.func import vmap, jacrev, hessian, jacfwd
# import pytorch3d.transforms as tf

import matplotlib.pyplot as plt
from utils.allegro_utils import *
from allegro_valve_roll import AllegroValveTurning, AllegroContactProblem, PositionControlConstrainedSVGDMPC
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



CCAI_PATH = pathlib.Path(__file__).resolve().parents[1]

obj_dof = 6
# instantiate environment
img_save_dir = pathlib.Path(f'{CCAI_PATH}/data/experiments/videos')

class AllegroCuboidAlignment(AllegroValveTurning):
    def get_constraint_dim(self, T):
        self.friction_polytope_k = 4
        wrench_dim = 0
        if self.obj_translational_dim > 0:
            wrench_dim += 3
        if self.obj_rotational_dim > 0:
            wrench_dim += 3
        if self.optimize_force:
            self.dg_per_t = self.num_fingers * (1 + 2 + 4) + wrench_dim + 1
        else:
            self.dg_per_t = self.num_fingers * (1 + 2) + wrench_dim + 1
        self.dg_constant = 0
        self.dg = self.dg_per_t * T + self.dg_constant  # terminal contact points, terminal sdf=0, and dynamics
        self.dz = (self.friction_polytope_k) * (self.num_fingers + 1) # one friction constraints per finger
        self.dz += self.num_fingers # min force constraint
        if self.collision_checking:
            self.dz += 1
        self.dh = self.dz * T  # inequality
    def __init__(self,
                 start,
                 goal,
                 T,
                 chain,
                 object_location,
                 object_type,
                 world_trans,
                 cuboid_asset_pos,
                 wall_asset_pos,
                 wall_dims,
                 fingers=['index', 'middle', 'ring', 'thumb'],
                 friction_coefficient=0.95,
                 optimize_force=False,
                 obj_dof_code=[1, 1, 1, 1, 1, 1],
                 obj_gravity=False,
                 arm_stiffness=None,
                 finger_stiffness=None,
                 collision_checking=True,
                 device='cuda:0', **kwargs):
        self.fingers = fingers
        self.num_fingers = len(fingers)
        self.optimize_force = optimize_force
        self.cuboid_asset_pos = cuboid_asset_pos
        self.cuboid_trans = tf.Transform3d(pos=torch.tensor(self.cuboid_asset_pos, device=device).float(),
                                          rot=torch.tensor(
                                        [1, 0, 0, 0],
                                        device=device).float(), device=device)
        self.wall_asset_pos = wall_asset_pos
        self.wall_dims = wall_dims.astype('float32')
        self.arm_dof = 0
        robot_dof = self.arm_dof + 4 * self.num_fingers
        du = robot_dof + 3 * self.num_fingers + 3 
        self.obj_mass = 0.01
        self.collision_checking = collision_checking

        super(AllegroCuboidAlignment, self).__init__(start=start, goal=goal, T=T, chain=chain, object_location=object_location,
                                                 object_type=object_type, world_trans=world_trans, object_asset_pos=cuboid_asset_pos,
                                                 fingers=fingers, friction_coefficient=friction_coefficient, obj_dof_code=obj_dof_code, 
                                                 obj_joint_dim=0, optimize_force=optimize_force, du=du, obj_gravity=obj_gravity, 
                                                 arm_stiffness=arm_stiffness, finger_stiffness=finger_stiffness, 
                                                 collision_checking=self.collision_checking, device=device)
        self.env_force = True
        self.friction_coefficient = friction_coefficient
        self.force_equlibrium_constr = vmap(self._force_equlibrium_constr_w_force)
        self.grad_force_equlibrium_constr = vmap(jacrev(self._force_equlibrium_constr_w_force, argnums=(0, 1, 2, 3, 4, 5, 6)))

        self.wall_friction_constr = vmap(self._wall_friction_constr, randomness='same')
        self.grad_wall_friction_constr = vmap(jacrev(self._wall_friction_constr, argnums=(0, 1)))


        # append the additional env force 
        max_f = torch.ones(3) * 10
        min_f = torch.ones(3) * -10
        self.x_max = torch.cat((self.x_max, max_f))
        self.x_min = torch.cat((self.x_min, min_f))
        self.min_force_dict = {'index': 0.1, 'middle': 0.1, 'ring': 0.1, 'thumb': 0.1}

    
    def _init_contact_scenes(self, asset_object, collision_checking):
        # robot and cuboid
        cuboid_chain = pk.build_chain_from_urdf(open(asset_object).read())
        cuboid_chain = cuboid_chain.to(device=self.device)
        cuboid_sdf = pv.RobotSDF(cuboid_chain, path_prefix=None, use_collision_geometry=True) # since we are using primitive shapes for the object, there's no need to define path for stl
        robot_sdf = pv.RobotSDF(self.chain, path_prefix=get_assets_dir() + '/xela_models', use_collision_geometry=True)

        robot2cuboid = self.world_trans.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.cuboid_asset_pos[0], self.cuboid_asset_pos[1], self.cuboid_asset_pos[2]))

        # contact checking
        collision_check_links = [self.collision_checking_ee_names[finger] for finger in self.fingers]
        self.robot_cuboid_scenes = pv.RobotScene(robot_sdf, cuboid_sdf, robot2cuboid,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            partial_patch=False,
                                            )
        viz_cuboid_sdf = pv.RobotSDF(cuboid_chain, path_prefix=None, use_collision_geometry=False) # since we are using primitive shapes for the object, there's no need to define path for stl
        viz_robot_sdf = pv.RobotSDF(self.chain, path_prefix=get_assets_dir() + '/xela_models', use_collision_geometry=False)

        self.viz_contact_scenes = pv.RobotScene(viz_robot_sdf, viz_cuboid_sdf, robot2cuboid,
                                            collision_check_links=collision_check_links,
                                            softmin_temp=1.0e3,
                                            points_per_link=1000,
                                            partial_patch=False,
                                            )
        # cuboid and wall
        wall_sdf = pv.BoxSDF([self.wall_dims[0], self.wall_dims[1], self.wall_dims[2]], device=self.device)
        world2cuboid = tf.Transform3d(pos=torch.tensor(self.cuboid_asset_pos, device=self.device).float(),
                                          rot=torch.tensor(
                                              [1, 0, 0, 0],
                                              device=self.device).float(), device=self.device)
        cuboid2wall = world2cuboid.inverse().compose(
            pk.Transform3d(device=self.device).translate(self.wall_asset_pos[0], self.wall_asset_pos[1], self.wall_asset_pos[2]))
        self.cuboid_wall_scenes = pv.RobotScene(cuboid_sdf, wall_sdf, cuboid2wall,
                                            collision_check_links=['cuboid'],
                                            softmin_temp=1.0e3,
                                            points_per_link=2000,
                                            partial_patch=False,
                                            )
        
        # robot and wall
        if self.collision_checking:
            robot2wall = self.world_trans.inverse().compose(
                pk.Transform3d(device=self.device).translate(self.wall_asset_pos[0], self.wall_asset_pos[1], self.wall_asset_pos[2]))
            self.robot_wall_scenes = pv.RobotScene(robot_sdf, wall_sdf, robot2wall,
                                                collision_check_links=['allegro_hand_oya_finger_link_15'],
                                                softmin_temp=1.0e3,
                                                points_per_link=2000,
                                                partial_patch=False,
                                                )

    def _preprocess(self, xu):
        N = xu.shape[0]
        xu = xu.reshape(N, self.T, -1)
        x = xu[:, :, :self.dx]
        # expand to include start
        x_expanded = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)

        q = x_expanded[:, :, :4 * self.num_fingers]
        if self.fixed_obj:
            theta = self.start_obj_pose.unsqueeze(0).repeat((N, self.T + 1, 1))
        else:
            theta = x_expanded[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof]
        # theta = x_expanded[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof]
        self._preprocess_fingers(q, theta)

    def _preprocess_fingers(self, q, theta):
        N, _, _ = q.shape

        # reshape to batch across time
        q_b = q.reshape(-1, 4 * self.num_fingers)
        theta_b = theta.reshape(-1, self.obj_dof)
        if self.obj_joint_dim > 0:
            theta_obj_joint = torch.zeros((theta_b.shape[0], self.obj_joint_dim),
                                          device=theta_b.device)  # add an additional dimension for the cap of the screw driver
            # the cap does not matter for the task, but needs to be included in the state for the model
            theta_b = torch.cat((theta_b, theta_obj_joint), dim=1)
        full_q = partial_to_full_state(q_b, fingers=self.fingers)
        ret_scene = self.robot_cuboid_scenes.scene_collision_check(full_q, theta_b,
                                                              compute_gradient=True,
                                                              compute_hessian=False)
        self.data['cuboid'] = {}
        for i, finger in enumerate(self.fingers):
            self.data[finger] = {}
            self.data[finger]['sdf'] = ret_scene['sdf'][:, i].reshape(N, self.T + 1)
            # reshape and throw away data for unused fingers
            grad_g_q = ret_scene.get('grad_sdf', None)
            self.data[finger]['grad_sdf'] = grad_g_q[:, i].reshape(N, self.T + 1, 16)[:, :, self.all_joint_index]

            # contact jacobian
            contact_jacobian = ret_scene.get('contact_jacobian', None)
            self.data[finger]['contact_jacobian'] = contact_jacobian[:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]

            # contact hessian
            contact_hessian = ret_scene.get('contact_hessian', None)
            contact_hessian = contact_hessian[:, i].reshape(N, self.T + 1, 3, 16, 16)[:, :, :, self.all_joint_index]
            contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            # contact_hessian = contact_hessian[:, :, :, :, self.all_joint_index]  # shape (N, T+1, 3, 8, 8)

            # gradient of contact point
            d_contact_loc_dq = ret_scene.get('closest_pt_q_grad', None)
            d_contact_loc_dq = d_contact_loc_dq[:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]  # [:, :, :, self.all_joint_index]
            self.data[finger]['closest_pt_q_grad'] = d_contact_loc_dq
            self.data[finger]['contact_hessian'] = contact_hessian
            self.data[finger]['closest_pt_world'] = ret_scene['closest_pt_world'][:, i]
            self.data[finger]['contact_normal'] = ret_scene['contact_normal'][:, i]

            # gradient of contact normal
            self.data[finger]['dnormal_dq'] = ret_scene['dnormal_dq'][:, i].reshape(N, self.T + 1, 3, 16)[:, :, :, self.all_joint_index]  # [:, :, :,
            # self.all_joint_index]

            self.data[finger]['dnormal_denv_q'] = ret_scene['dnormal_denv_q'][:, i, :, :self.obj_dof]
            self.data[finger]['grad_env_sdf'] = ret_scene['grad_env_sdf'][:, i, :self.obj_dof]
            dJ_dq = contact_hessian
            self.data[finger]['dJ_dq'] = dJ_dq  # Jacobian of the contact point

        ret_scene_cuboid_wall = self.cuboid_wall_scenes.scene_collision_check(theta_b, None,
                                                                compute_gradient=True,
                                                                compute_hessian=False)
        self.data['cuboid']['sdf_cuboid_wall'] = ret_scene_cuboid_wall['sdf'].reshape(N, self.T + 1)
        self.data['cuboid']['grad_sdf_cuboid_wall'] = ret_scene_cuboid_wall['grad_sdf'].reshape(N, self.T + 1, 6)
        d_contact_loc_dq = ret_scene_cuboid_wall.get('closest_pt_q_grad', None)
        d_contact_loc_dq = d_contact_loc_dq.reshape(N, self.T + 1, 3, 6)
        self.data['cuboid']['closest_pt_q_grad'] = d_contact_loc_dq
        # self.data['cuboid']['contact_hessian'] = contact_hessian
        self.data['cuboid']['closest_pt_world'] = ret_scene_cuboid_wall['closest_pt_world']
        self.data['cuboid']['contact_normal'] = ret_scene_cuboid_wall['contact_normal']

        # gradient of contact normal
        self.data['cuboid']['dnormal_dq'] = ret_scene_cuboid_wall['dnormal_dq'].reshape(N, self.T + 1, 3, 6)

        if self.collision_checking:
            ret_scene_robot_wall = self.robot_wall_scenes.scene_collision_check(full_q, None,
                                                                    compute_gradient=True,
                                                                    compute_hessian=False)
            self.data['allegro_hand_oya_finger_link_15'] = {}
            self.data['allegro_hand_oya_finger_link_15']['sdf'] = ret_scene_robot_wall['sdf'].reshape(N, self.T + 1)
            grad_g_q = ret_scene_robot_wall.get('grad_sdf', None)
            self.data['allegro_hand_oya_finger_link_15']['grad_sdf'] = grad_g_q.reshape(N, self.T + 1, 16)[:, :, self.all_joint_index]

    def get_initial_xu(self, N):
        # TODO: fix the initialization, for 6D movement, the angle is not supposed to be the linear interpolation of the euler angle. 
        """
        use delta joint movement to get the initial trajectory
        the action (force at the finger tip) is not used. it is randomly intiailized
        the actual dynamics model is not used
        """

        # u = 0.025 * torch.randn(N, self.T, self.du, device=self.device)
        u = 0.025 * torch.randn(N, self.T, 4 * self.num_fingers, device=self.device)
        force = 0.015 * torch.randn(N, self.T, 3 * (self.num_fingers + 1), device=self.device)
        force[:, :, -3:] = force[:, :, -3:] * 0.1
        force = force * 10 # increase the force to make it transferrable on the hardware
        u = torch.cat((u, force), dim=-1)

        x = [self.start.reshape(1, self.dx).repeat(N, 1)]
        for t in range(self.T):
            next_q = x[-1][:, :4 * self.num_fingers] + u[:, t, :4 * self.num_fingers]
            x.append(next_q)

        x = torch.stack(x[1:], dim=1)

        # if valve angle in state
        current_obj_position = self.start[4 * self.num_fingers: 4 * self.num_fingers + self.obj_translational_dim]
        current_obj_orientation = self.start[4 * self.num_fingers + self.obj_translational_dim:4 * self.num_fingers + self.obj_dof]
        current_obj_R = R.from_euler('XYZ', current_obj_orientation.cpu().numpy())
        goal_obj_R = R.from_euler('XYZ', self.goal[self.obj_translational_dim:self.obj_translational_dim + self.obj_rotational_dim].cpu().numpy())
        key_times = [0, self.T]
        times = np.linspace(0, self.T, self.T + 1)
        slerp = Slerp(key_times, R.concatenate([current_obj_R, goal_obj_R]))
        interp_rots = slerp(times)
        interp_rots = interp_rots.as_euler('XYZ')[1:]
        # current_obj_orientation = tf.euler_angles_to_matrix(current_obj_orientation, convention='XYZ')
        # current_obj_orientation = tf.matrix_to_rotation_6d(current_obj_orientation)

        theta_position = np.linspace(current_obj_position.cpu().numpy(), self.goal[:self.obj_translational_dim].cpu().numpy(), self.T + 1)[1:]
        theta = np.concatenate((theta_position, interp_rots), axis=-1)
        theta = torch.tensor(theta, device=self.device, dtype=torch.float32)
        theta = theta.unsqueeze(0).repeat((N,1,1))

        # DEBUG ONLY, use initial state as the initialization
        # theta = self.start[-self.obj_dof:].unsqueeze(0).repeat((N, self.T, 1))
        theta = torch.ones((N, self.T, self.obj_dof)).to(self.device) * self.start[-self.obj_dof:]
        x = torch.cat((x, theta), dim=-1)

        xu = torch.cat((x, u), dim=2)
        return xu

    def _cuboid_wall_contact_constraint(self, theta, compute_grads=True, compute_hess=False):
        N, T, _ = theta.shape
        # Retrieve pre-processed data
        g = self.data['cuboid'].get('sdf_cuboid_wall').reshape(N, T + 1, 1)  # - 0.0025
        grad_g_theta = self.data['cuboid'].get('grad_sdf_cuboid_wall', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)

        if compute_grads:
            T_range = torch.arange(T, device=theta.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=theta.device)
            grad_g[:, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = grad_g_theta.reshape(N, T + 1, self.obj_dof)[:, 1:]
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None

    def _thumb_repulsive(self, xu, link_name, compute_grads=True, compute_hess=False):
        """
        the thumb should not touch the wall
        """
        # print(xu[0, :2, 4 * self.num_fingers])
        N, T, _ = xu.shape
        # Retrieve pre-processed data
        ret_scene = self.data[link_name]
        g = -ret_scene.get('sdf').reshape(N, T + 1, 1)  + 0.01
        grad_g_q = -ret_scene.get('grad_sdf', None)

        # Ignore first value, as it is the start state
        g = g[:, 1:].reshape(N, -1)
        if compute_grads:
            T_range = torch.arange(T, device=xu.device)
            # compute gradient of sdf
            grad_g = torch.zeros(N, T, T, self.dx + self.du, device=xu.device)
            grad_g[:, T_range, T_range, :self.robot_dof] = grad_g_q[:, 1:]
            # is valve in state
            grad_g = grad_g.reshape(N, -1, T, self.dx + self.du)
            grad_g = grad_g.reshape(N, -1, T * (self.dx + self.du))
        else:
            return g, None, None

        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * (self.dx + self.du), T * (self.dx + self.du), device=self.device)
            return g, grad_g, hess

        return g, grad_g, None
    
    def _cost(self, xu, start, goal):
        # TODO: consider using quaternion difference for the orientation.
        state = xu[:, :self.dx]  # state dim = 9
        state = torch.cat((start.reshape(1, self.dx), state), dim=0)  # combine the first time step into it
        
        action = xu[:, self.dx:self.dx + 4 * self.num_fingers]  # action dim = 8
        next_q = state[:-1, :-self.obj_dof] + action
        smoothness_cost = 1 * torch.sum((state[1:] - state[:-1]) ** 2)

        action_cost = 0
        goal_cost = 0
        if self.obj_translational_dim:
            obj_position = state[:, -self.obj_dof:-self.obj_dof+self.obj_translational_dim]
            # terminal cost
            # goal_cost = goal_cost + torch.sum((100 * (obj_position[-1, 1:] - goal[1:self.obj_translational_dim]) ** 2)) # give flxibility in x direction
            # # running cost
            # goal_cost = goal_cost + torch.sum((1 * (obj_position[:, 1:] - goal[1:self.obj_translational_dim]) ** 2))
            smoothness_cost = smoothness_cost + 10000 * torch.sum((obj_position[1:] - obj_position[:-1]) ** 2)
        if self.obj_rotational_dim:
            obj_orientation = state[:, -self.obj_dof+self.obj_translational_dim:]
            obj_orientation = tf.euler_angles_to_matrix(obj_orientation, convention='XYZ')
            obj_orientation = tf.matrix_to_rotation_6d(obj_orientation)
            goal_orientation = tf.euler_angles_to_matrix(goal[-self.obj_rotational_dim:], convention='XYZ')
            goal_orientation = tf.matrix_to_rotation_6d(goal_orientation)
            # terminal cost
            # goal_cost = goal_cost + torch.sum((250 * (obj_orientation[-1] - goal_orientation) ** 2))
            goal_cost = goal_cost + torch.sum((1000 * (obj_orientation[-1] - goal_orientation) ** 2))
            # running cost 
            goal_cost = goal_cost + torch.sum((5 * (obj_orientation - goal_orientation) ** 2))
            smoothness_cost = smoothness_cost + 100 * torch.sum((obj_orientation[1:] - obj_orientation[:-1]) ** 2)
        # goal_cost = torch.sum((1000 * (state[-1, -self.obj_dof:] - goal) ** 2)).reshape(-1)
        # goal_cost += torch.sum((10 * (state[:, -self.obj_dof:] - goal.unsqueeze(0)) ** 2))
        return smoothness_cost + action_cost + goal_cost 

    def _con_eq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        g_contact, grad_g_contact, hess_g_contact = self._contact_constraints(xu=xu.reshape(N, T, self.dx + self.du),
                                                                              compute_grads=compute_grads,
                                                                              compute_hess=compute_hess)
        g_cuboid_contact, grad_g_cuboid_contact, hess_g_cuboid_contact \
            = self._cuboid_wall_contact_constraint(xu[:, :, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof],
                                                                                                  compute_grads=compute_grads,
                                                                                                  compute_hess=compute_hess)
        if self.optimize_force:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints_w_force(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)
        else:
            g_equil, grad_g_equil, hess_g_equil = self._force_equlibrium_constraints(
                xu=xu.reshape(N, T, self.dx + self.du),
                compute_grads=compute_grads,
                compute_hess=compute_hess)

        g_valve, grad_g_valve, hess_g_valve = self._kinematics_constraints(
            xu=xu.reshape(N, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess,
            projection=True)
        

        if verbose:
            print(f"max contact constraint: {torch.max(torch.abs(g_contact))}")
            print(f"max cuboid wall contact constraint: {torch.max(torch.abs(g_cuboid_contact))}")
            print(f"max valve kinematics constraint: {torch.max(torch.abs(g_valve))}")
            print(f"max force equilibrium constraint: {torch.max(torch.abs(g_equil))}")
            result_dict = {}
            result_dict['contact'] = torch.max(torch.abs(g_contact)).item()
            result_dict['cuboid_wall_contact'] = torch.max(torch.abs(g_cuboid_contact)).item()
            result_dict['kinematics'] = torch.max(torch.abs(g_valve)).item()
            result_dict['force'] = torch.max(torch.abs(g_equil)).item()
            result_dict['contact_mean'] = torch.mean(torch.abs(g_contact)).item()
            result_dict['cuboid_wall_contact_mean'] = torch.mean(torch.abs(g_cuboid_contact)).item()
            result_dict['kinematics_mean'] = torch.mean(torch.abs(g_valve)).item()
            result_dict['force_mean'] = torch.mean(torch.abs(g_equil)).item()

            return result_dict
        g_contact = torch.cat((
                                g_contact, 
                                g_cuboid_contact,
                                g_equil,
                                g_valve,
                               ), dim=1)

        if grad_g_contact is not None:
            grad_g_contact = torch.cat((
                                        grad_g_contact, 
                                        grad_g_cuboid_contact,
                                        grad_g_equil,
                                        grad_g_valve,
                                        ), dim=1)
        if hess_g_contact is not None:
            hess_g_contact = torch.cat((
                                        hess_g_contact, 
                                        hess_g_cuboid_contact,
                                        hess_g_equil,
                                        hess_g_valve,
                                        ), dim=1)

        return g_contact, grad_g_contact, hess_g_contact  

    def _con_ineq(self, xu, compute_grads=True, compute_hess=False, verbose=False):
        N = xu.shape[0]
        T = xu.shape[1]
        h, grad_h, hess_h = self._friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        h_wall, grad_h_wall, hess_h_wall = self._wall_friction_constraint(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        
        h_force, grad_h_force, hess_h_force = self._min_force_constraints(
            xu=xu.reshape(-1, T, self.dx + self.du),
            compute_grads=compute_grads,
            compute_hess=compute_hess)
        if self.collision_checking:
            h_rep, grad_h_rep, hess_h_rep = self._thumb_repulsive(
                xu.reshape(-1, T, self.dx + self.du),
                link_name='allegro_hand_oya_finger_link_15',
                compute_grads=compute_grads,
                compute_hess=compute_hess,
            )
        
        if verbose:
            print(f"max friction constraint: {torch.max(h)}")
            print(f"max wall friction constraint: {torch.max(h_wall)}")
            print(f"max force constraint: {torch.max(h_force)}")
            result_dict = {}
            result_dict['friction'] = torch.max(h).item()
            result_dict['friction_mean'] = torch.mean(h).item()
            result_dict['wall_friction'] = torch.max(h_wall).item()
            result_dict['wall_friction_mean'] = torch.mean(h_wall).item()
            result_dict['min_force'] = torch.max(h_force).item() 
            result_dict['min_force_mean'] = torch.mean(h_force).item()
            if self.collision_checking:
                print(f"max thumb repulsive constraint: {torch.max(h_rep)}")
                result_dict['thumb_repulsive'] = torch.max(h_rep).item()
                result_dict['thumb_repulsive_mean'] = torch.mean(h_rep).item()
            return result_dict

        h = torch.cat((h,
                       h_wall,
                       h_force,
                       ), dim=1)
        if self.collision_checking:
            h = torch.cat((h, h_rep), dim=1)
        if compute_grads:
            grad_h = grad_h.reshape(N, -1, self.T * (self.dx + self.du))
            grad_h = torch.cat((grad_h, 
                                grad_h_wall,
                                grad_h_force
                                ), dim=1)
            if self.collision_checking:
                grad_h = torch.cat((grad_h, grad_h_rep), dim=1)
        else:
            return h, None, None
        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], self.T * (self.dx + self.du), self.T * (self.dx + self.du),
                                device=self.device)
            return h, grad_h, hess_h
        return h, grad_h, None    
    
    def _force_equlibrium_constr_w_force(self, q, u, next_q, force_list, contact_jac_list, contact_point_list, next_env_q):
        # NOTE: the constriant is defined in the robot frame
        # the contact jac an contact points are all in the robot frame
        # this will be vmapped, so takes in a 3 vector and a [num_finger x 3 x 8] jacobian and a dq vector
        obj_robot_frame = self.world_trans.inverse().transform_points(self.object_location.reshape(1, 3))
        delta_q = q + u - next_q
        torque_list = []
        reactional_torque_list = []
        for i, finger_name in enumerate(self.fingers):
            # TODO: Assume that all the fingers are having an equlibrium, maybe we should change so that index finger is not considered
            contact_jacobian = contact_jac_list[i]
            force_robot_frame = self.world_trans.inverse().transform_normals(force_list[i].unsqueeze(0)).squeeze(0)
            reactional_torque_list.append(contact_jacobian.T @ -force_robot_frame)
            # pseudo inverse form
            contact_point_r_valve = contact_point_list[i] - obj_robot_frame[0]
            torque = torch.linalg.cross(contact_point_r_valve, force_robot_frame)
            torque_list.append(torque)
            # It does not matter for comuputing the force equilibrium constraint
        env_force = force_list[-1]
        cuboid_wall_contact_point_cuboid_frame = contact_point_list[-1]
        cuboid_wall_contact_point_robot_frame = self.world_trans.inverse().transform_points(self.cuboid_trans.transform_points(contact_point_list[-1].reshape(1,3)))
        cuboid_wall_r = cuboid_wall_contact_point_robot_frame[0] - obj_robot_frame[0]
        env_force_robot_frame = force_robot_frame = self.world_trans.inverse().transform_normals(env_force.unsqueeze(0)).squeeze(0)
        cuboid_wall_torque = torch.linalg.cross(cuboid_wall_r, env_force_robot_frame)
        torque_list.append(cuboid_wall_torque)
        # contact_point_r_valve = contact_point_list[-1] - obj_robot_frame[0]
        if self.obj_gravity:
            if self.obj_translational_dim > 0:
                g = self.obj_mass * torch.tensor([0, 0, -9.8], device=self.device, dtype=torch.float32)
                force_list = torch.cat((force_list, g.unsqueeze(0)))

        # force_world_frame = self.world_trans.transform_normals(force.unsqueeze(0)).squeeze(0)
        torque_list = torch.stack(torque_list, dim=0)
        sum_torque_list = torch.sum(torque_list, dim=0)
        sum_force_list = torch.sum(force_list, dim=0)

        g = []
        if self.obj_translational_dim > 0:
            g.append(sum_force_list)
        elif self.screwdriver_force_balance:
            # balance the force with the torque
            g.append(torch.sum(force_list, dim=0)[:2])
        if self.obj_rotational_dim > 0:
            g.append(sum_torque_list)
        g = torch.cat(g)
        reactional_torque_list = torch.stack(reactional_torque_list, dim=0)
        sum_reactional_torque = torch.sum(reactional_torque_list, dim=0)
        g_force_torque_balance = (sum_reactional_torque + 3.0 * delta_q)
        # print(g_force_torque_balance.max(), torque_list.max())
        g = torch.cat((g, g_force_torque_balance.reshape(-1)), dim=-1)
        # residual_list = torch.stack(residual_list, dim=0) * 100
        # g = torch.cat((torque_list, residual_list), dim=-1)
        return g

    def _force_equlibrium_constraints_w_force(self, xu, compute_grads=True, compute_hess=False):
        N, T, d = xu.shape
        x = xu[:, :, :self.dx]

        # we want to add the start state to x, this x is now T + 1
        x = torch.cat((self.start.reshape(1, 1, -1).repeat(N, 1, 1), x), dim=1)
        q = x[:, :-1, :self.num_fingers * 4]
        next_q = x[:, 1:, :self.num_fingers * 4]
        next_env_q = x[:, 1:, self.num_fingers * 4: self.num_fingers * 4 + self.obj_dof]
        u = xu[:, :, self.dx: self.dx + 4 * self.num_fingers]
        force = xu[:, :, self.dx + 4 * self.num_fingers:]
        force_list = force.reshape((force.shape[0], force.shape[1], self.num_fingers + 1, 3))
        # contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1].reshape(-1, 3, 4 * self.num_fingers)\
        #                      for finger_name in self.fingers]
        contact_jac_list = [self.data[finger_name]['contact_jacobian'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, 1:].reshape(-1, 3, 4 * self.num_fingers)\
                             for finger_name in self.fingers]
        contact_jac_list = torch.stack(contact_jac_list, dim=1).to(device=xu.device)
        contact_point_list = [self.data[finger_name]['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3) for finger_name in self.fingers]
        contact_point_list.append(self.data['cuboid']['closest_pt_world'].reshape(N, T + 1, 3)[:, :-1].reshape(-1, 3))
        contact_point_list = torch.stack(contact_point_list, dim=1).to(device=xu.device)

        g = self.force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                         u.reshape(-1, 4 * self.num_fingers), 
                                         next_q.reshape(-1, 4 * self.num_fingers), 
                                         force_list.reshape(-1, self.num_fingers + 1, 3),
                                         contact_jac_list,
                                         contact_point_list,
                                         next_env_q.reshape(-1, self.obj_dof)).reshape(N, T, -1)
        # print(g.abs().max().detach().cpu().item(), g.abs().mean().detach().cpu().item())
        if compute_grads:
            dg_dq, dg_du, dg_dnext_q, dg_dforce, dg_djac, dg_dcontact, dg_dnext_env_q = self.grad_force_equlibrium_constr(q.reshape(-1, 4 * self.num_fingers), 
                                                                                  u.reshape(-1, 4 * self.num_fingers), 
                                                                                  next_q.reshape(-1, 4 * self.num_fingers), 
                                                                                  force_list.reshape(-1, self.num_fingers + 1, 3),
                                                                                  contact_jac_list,
                                                                                  contact_point_list,
                                                                                  next_env_q.reshape(-1, self.obj_dof))
            dg_dforce = dg_dforce.reshape(dg_dforce.shape[0], dg_dforce.shape[1], (self.num_fingers + 1) * 3)
            
            T_range = torch.arange(T, device=x.device)
            T_plus = torch.arange(1, T, device=x.device)
            T_minus = torch.arange(T - 1, device=x.device)
            grad_g = torch.zeros(N, g.shape[2], T, T, self.dx + self.du, device=self.device)
            # dnormal_dq = torch.zeros(N, T, 3, 8, device=self.device)  # assume zero SDF hessian
            dg_dq = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            dg_dnext_q = dg_dnext_q.reshape(N, T, g.shape[2], 4 * self.num_fingers) 
            for i, finger_name in enumerate(self.fingers):
                # NOTE: assume fingers have joints independent of each other
                # djac_dq = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, :-1] # jacobian is the contact jacobian
                # dg_dq = dg_dq + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dq.reshape(N, T, -1, 4 * self.num_fingers)
                djac_dnext_q = self.data[finger_name]['dJ_dq'].reshape(N, T + 1, 3, 4 * self.num_fingers, 4 * self.num_fingers)[:, 1:]
                dg_dnext_q = dg_dnext_q + dg_djac[:, :, i].reshape(N, T, g.shape[2], -1) @ djac_dnext_q.reshape(N, T, -1, 4 * self.num_fingers)
                
                d_contact_loc_dq = self.data[finger_name]['closest_pt_q_grad'].reshape(N, T + 1, 3, 4 * self.num_fingers)[:, :-1]
                dg_dq = dg_dq + dg_dcontact[:, : ,i].reshape(N, T, g.shape[2], 3) @ d_contact_loc_dq 
            grad_g[:, :, T_plus, T_minus, :4 * self.num_fingers] = dg_dq.reshape(N, T, g.shape[2], 4 * self.num_fingers)[:, 1:].transpose(1, 2)  # first q is the start
            dg_du = torch.cat((dg_du, dg_dforce), dim=-1)  # check the dim
            grad_g[:, :, T_range, T_range, self.dx:] = dg_du.reshape(N, T, -1, self.du).transpose(1, 2)
            grad_g[:, :, T_range, T_range, :4 * self.num_fingers] = dg_dnext_q.reshape(N, T, -1, 4 * self.num_fingers).transpose(1, 2)
            if self.obj_gravity:
                grad_g[:, :, T_range, T_range, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dg_dnext_env_q.reshape(N, T, -1, self.obj_dof).transpose(1, 2)
            grad_g = grad_g.transpose(1, 2)
        else:
            return g.reshape(N, -1), None, None
        if compute_hess:
            hess = torch.zeros(N, g.shape[1], T * d, T * d, device=self.device)
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), hess
        else:
            return g.reshape(N, -1), grad_g.reshape(N, -1, T * (self.dx + self.du)), None

    def _wall_friction_constr(self, force, contact_normal):

        # compute the force in robot frame
        # transform contact normal to world frame
        contact_normal_world = self.cuboid_trans.transform_normals(contact_normal.unsqueeze(0)).squeeze(0)

        # transform force to contact frame
        R = self.get_rotation_from_normal(contact_normal_world.unsqueeze(0)).squeeze(0)
        # force_contact_frame = R.transpose(0, 1) @ force_world_frame.unsqueeze(-1)
        B = self.get_friction_polytope().detach()

        # here dq means the force in the world frame
        contact_v_contact_frame = R.transpose(0, 1) @ force
        return B @ contact_v_contact_frame

    def _wall_friction_constraint(self, xu, compute_grads=True, compute_hess=False):

        # assume access to class member variables which have already done some of the computation
        N, T, d = xu.shape
        u = xu[:, :, self.dx:]

        u = u[:, :, (4 + 3) * self.num_fingers: ].reshape(-1, 3)
        # retrieved cached values
        contact_normal = - self.data['cuboid']['contact_normal'].reshape(N, T + 1, 3)[:, :-1] # contact normal is pointing out 
        dnormal_dtheta = - self.data['cuboid']['dnormal_dq'].reshape(N, T + 1, 3, self.obj_dof)[:, :-1]

        # compute constraint value
        h = self.wall_friction_constr(u,
                                 contact_normal.reshape(-1, 3)).reshape(N, -1)

        # compute the gradient
        if compute_grads:
            dh_du, dh_dnormal = self.grad_wall_friction_constr(u,
                                                            contact_normal.reshape(-1, 3))

            dh = dh_dnormal.shape[1]
            dh_dtheta = dh_dnormal.reshape(N, T, dh, -1) @ dnormal_dtheta
            grad_h = torch.zeros(N, dh, T, T, d, device=self.device)
            T_range = torch.arange(T, device=self.device)
            T_range_minus = torch.arange(T - 1, device=self.device)
            T_range_plus = torch.arange(1, T, device=self.device)
            grad_h[:, :, T_range_plus, T_range_minus, 4 * self.num_fingers: 4 * self.num_fingers + self.obj_dof] = dh_dtheta[:, 1:].transpose(1, 2)
            grad_h[:, :, T_range, T_range, self.dx + 7 * self.num_fingers:] = dh_du.reshape(N, T, dh, 3).transpose(1, 2)
            grad_h = grad_h.transpose(1, 2).reshape(N, -1, T * d)
        else:
            return h, None, None

        if compute_hess:
            hess_h = torch.zeros(N, h.shape[1], T * d, T * d, device=self.device)
            return h, grad_h, hess_h

        return h, grad_h, None
