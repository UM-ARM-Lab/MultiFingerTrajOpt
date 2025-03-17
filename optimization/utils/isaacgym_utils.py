from MFR_benchmark.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv, AllegroCuboidTurningEnv, AllegroCuboidAlignmentEnv, AllegroReorientationEnv

def get_env(task, img_save_dir, config, num_envs=1):
    if task == 'screwdriver_turning':
        env = AllegroScrewdriverTurningEnv(num_envs=num_envs, 
                                           control_mode='joint_impedance',
                                            viewer=True,
                                            steps_per_action=60,
                                            friction_coefficient=1.0,
                                            device=config['sim_device'],
                                            video_save_path=img_save_dir,
                                            joint_stiffness=config['kp'],
                                            fingers=config['fingers'],
                                            gradual_control=config['gradual_control'],
                                            gravity=config['gravity'],
                                            )
    elif task == 'valve_turning':
        env = AllegroValveTurningEnv(num_envs=num_envs, 
                                    control_mode='joint_impedance',
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                valve_type=config['object_type'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                random_robot_pose=config['random_robot_pose'],
                                )
    elif task == 'peg_turning':
        env = AllegroCuboidTurningEnv(num_envs=num_envs,
                                control_mode='joint_impedance',
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=1.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                gradual_control=config['gradual_control'],
                            )
    elif task == 'peg_alignment':
        env = AllegroCuboidAlignmentEnv(num_envs=num_envs,
                                     control_mode='joint_impedance',
                                     viewer=True,
                                     steps_per_action=60,
                                     friction_coefficient=1.0,
                                     device=config['sim_device'],
                                     video_save_path=img_save_dir,
                                     joint_stiffness=config['kp'],
                                     fingers=config['fingers'],
                                     gravity=config['gravity'],
                                    gradual_control=config['gradual_control'],
                                     )
    elif task == 'reorientation':
        env = AllegroReorientationEnv(num_envs=num_envs,
                                control_mode='joint_impedance',
                                viewer=True,
                                steps_per_action=60,
                                friction_coefficient=2.0,
                                device=config['sim_device'],
                                video_save_path=img_save_dir,
                                joint_stiffness=config['kp'],
                                fingers=config['fingers'],
                                gravity=config['gravity'],
                                gradual_control=config['gradual_control'],
                                )
    return env