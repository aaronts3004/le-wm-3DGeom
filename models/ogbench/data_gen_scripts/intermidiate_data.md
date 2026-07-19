6
0 front_zoomed
1 front_pixels
2 left
3 right
4 side
5 top
Available cameras in the environment:  {'top', 'right', 'front_zoomed', 'front_pixels', 'left', 'side'}
Selected episodes: 1100
Total frames: 221100
Episodes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1100/1100 [1:21:45<00:00,  4.46s/it]
action | (221100, 5) | float32 | lzf                                                                                                                                                                                           
control | (221100, 7) | float64 | lzf
ep_idx | (221100,) | int32 | None
ep_len | (1100,) | int32 | None
ep_offset | (1100,) | int64 | None
id | (221100,) | int64 | lzf
observation | (221100, 28) | float64 | lzf
original_episode_ids | (1100,) | int32 | None
pixels | (221100, 224, 224, 3) | uint8 | lzf
prev_qpos | (221100, 21) | float64 | lzf
prev_qvel | (221100, 20) | float64 | lzf
privileged_block_0_pos | (221100, 3) | float64 | lzf
privileged_block_0_quat | (221100, 4) | float64 | lzf
privileged_block_0_yaw | (221100, 1) | float64 | lzf
privileged_target_block | (221100,) | int64 | lzf
privileged_target_block_pos | (221100, 3) | float64 | lzf
privileged_target_block_yaw | (221100, 1) | float64 | lzf
privileged_target_task | (221100,) | object | lzf
proprio_effector_pos | (221100, 3) | float64 | lzf
proprio_effector_yaw | (221100, 1) | float64 | lzf
proprio_gripper_contact | (221100, 1) | float64 | lzf
proprio_gripper_opening | (221100, 1) | float64 | lzf
proprio_gripper_vel | (221100, 1) | float64 | lzf
proprio_joint_pos | (221100, 6) | float64 | lzf
proprio_joint_vel | (221100, 6) | float64 | lzf
qpos | (221100, 21) | float64 | lzf
qvel | (221100, 20) | float64 | lzf
render_time | (221100,) | float64 | lzf
reward | (221100,) | float64 | lzf
step_idx | (221100,) | int64 | lzf
success | (221100,) | bool | lzf
target | (221100, 28) | float64 | lzf
terminated | (221100,) | bool | lzf
time | (221100, 1) | float64 | lzf
triple_cam_ex | (221100, 3, 4, 4) | float32 | lzf
triple_cam_in | (221100, 3, 3, 3) | float32 | lzf
triple_pixels | (221100, 3, 224, 224, 3) | uint8 | lzf
truncated | (221100,) | bool | lzf
Finished writing /home/student/data/ogbench/triple_views_with_cam_train_1000_val_100_episodes.h