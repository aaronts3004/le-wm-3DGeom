

import h5py
import numpy as np
from pathlib import Path
import os
os.environ["MUJOCO_GL"] = "egl"
import gymnasium
import ogbench
import mujoco
import open3d as o3d
from PIL import Image


'''
    GET qpos and qvel for episode 0 => then generate full data and save to global npz in frame-major format:: 
    images       : (201, 6, H, W, 3)
    depths       : (201, 6, H, W)
    intrinsics   : (201, 6, 3, 3)
    extrinsics   : (201, 6, 4, 4)
    qpos         : (201, nq)
    qvel         : (201, nv)

    points       : object array length 201
    colors       : object array length 201
'''

SOURCE_FILE = Path("~/data/ogbench/cube_single_expert.h5").expanduser()
print("Reading h5 from ", SOURCE_FILE)
print("Found: ", Path(SOURCE_FILE).exists())


# --------------------------------------------------
# Open files
# --------------------------------------------------

with h5py.File(SOURCE_FILE, "r") as f_src: 
    qpos_ds_all = f_src["qpos"]
    qvel_ds_all = f_src["qvel"] 


    print("qpos.shape=", qpos_ds_all.shape)

    episode_0_qpos = qpos_ds_all[:201]
    episode_0_qvel = qvel_ds_all[:201]

NUM_VIEWS = 6
CAMERA_NAMES = ["front_zoomed", "front_pixels", "left", "right", "side", "top"]

env = gymnasium.make(
    "visual-cube-single-v0",
    terminate_at_goal=False,
    mode="data_collection",
)
env.reset()

model = env.unwrapped.model
data = env.unwrapped.data

# test the object id in model
print("Bodies:")
for i in range(model.nbody):
    print(i, model.body(i).name)
# cube has the object id "object_0"

print("\nSites:")
for i in range(model.nsite):
    print(i, model.site(i).name)
# tcp site has the id "ur5e/robotiq/pinch"

print("\nCameras:") 
for i in range(model.ncam):
    print(i, model.cam(i).name)

# ============================================================
# Helper functions
# ============================================================
def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float,
    pose: str = "GLB",
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all = [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        
        ### SWITCH: 
        if pose == "GLB":
            Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  ### GLOBAL SPACE            !!! temporarily disable global shift
        elif pose == "CAM":
            Xw = Xc.T.astype(np.float32)                ### CAM SPACE
        else:
            raise ValueError(f"Unknown pose type: {pose}")

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)

        pts_all.append(Xw)
        col_all.append(cols)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0)

def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def get_camera_intrinsic(model, camera_name, width, height):
    """
    Return camera intrinsic matrix K.
    """

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    fovy = model.cam_fovy[cam_id]

    fy = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    fx = fy

    cx = width / 2.0
    cy = height / 2.0

    # K = np.array([
    #     [-fx, 0,  cx],
    #     [0,  fy, cy],
    #     [0,  0,  1 ],
    # ], dtype=np.float32)
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)

    return K

def get_camera_extrinsic(model, data, camera_name):
    """
    World -> Camera extrinsic matrix.
    """

    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    pos = data.cam_xpos[cam_id].copy()

    # camera rotation matrix (camera -> world)
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # world -> camera
    # R_w2c = R_c2w.T
    R_cv_mj = np.diag([1, -1, -1])                                  ### what is this

    R_w2c = R_cv_mj @ R_c2w.T
    t_w2c = -R_w2c @ pos

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_w2c
    T[:3, 3] = t_w2c

    return T

def get_body_position(model, data, body_name):
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    return data.xpos[body_id].copy()

def get_site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    return data.site_xpos[site_id].copy()



# ============================================================
# Example qpos / qvel
# ============================================================

all_images = []
all_depths = []
all_segments = []
all_intrinsics = []
all_extrinsics = []

all_qpos = []
all_qvel = []

all_points = []
all_colors = []

all_gripper_pos=[]
all_cube_pos=[]

save_dir_ply = "outputs/plys"
os.makedirs(save_dir_ply, exist_ok=True)
for frame_index in range(201):

    print(f"Processing frame {frame_index} / 201")
    qpos = episode_0_qpos[frame_index]
    qvel = episode_0_qvel[frame_index]

    qpos = np.array(qpos)
    qvel = np.array(qvel)

    env.unwrapped.set_state(qpos, qvel)
    mujoco.mj_forward(model, data)

    # ============================================================
    # get cube position and gripper position
    # ============================================================

    cube_pos = get_body_position(model, data, "object_0")
    # tcp_pos = get_site_position(model, data, "ur5e/robotiq/pinch")
    right_driver_pos = get_body_position(model, data, "ur5e/robotiq/right_driver")
    left_driver_pos = get_body_position(model, data, "ur5e/robotiq/left_driver")
    gripper_pos = (right_driver_pos + left_driver_pos) / 2.0
    # print("cube_pos:", cube_pos)
    # print("gripper_pos:", gripper_pos)
    # ============================================================
    # render rgb/depth/segmentation and get camera intrinsics / extrinsics
    # ============================================================
    rgb_views = []
    depth_views = []
    seg_views = []

    camera_intrinsics = []
    camera_extrinsics = []

    for camera_name in CAMERA_NAMES:

        # rgb
        rgb = env.unwrapped.render(
            camera=camera_name
        )

        rgb_views.append(rgb)

        H, W = rgb.shape[:2]

        # metric depth converted from z-buffer
        depth = env.unwrapped.render(
            camera=camera_name,
            depth=True,
        )

        depth_views.append(depth)

        # segmentation mask
        # img: (H, W, 2), uint32
        # 2-channel (object ID, object type) pair
        # for the object ID, -1 is background, 0 is table, and positive integers are different objects
        # for the object type, -1 is background, 5 is scene
        seg = env.unwrapped.render(
            camera=camera_name,
            segmentation=True,
        )

        seg_views.append(seg)

        K = get_camera_intrinsic(
            model,
            camera_name,
            W,
            H,
        )

        T = get_camera_extrinsic(
            model,
            data,
            camera_name,
        )

        camera_intrinsics.append(K)
        camera_extrinsics.append(T)

    rgb_views = np.stack(rgb_views)
    depth_views = np.stack(depth_views)
    seg_views = np.stack(seg_views)

    camera_intrinsics = np.stack(camera_intrinsics)
    camera_extrinsics = np.stack(camera_extrinsics)

    # print("rgb_views:", rgb_views.shape)
    # print("depth_views:", depth_views.shape)
    # print("seg_views:", seg_views.shape)
    # print("camera_intrinsics:", camera_intrinsics.shape)
    # print("camera_extrinsics:", camera_extrinsics.shape)

    points, colors = _depths_to_world_points_with_colors(
        depth_views, camera_intrinsics, camera_extrinsics, rgb_views, conf=None, conf_thr=0.0
    )

    # APPEND 6 CAMERA VIEWS + DATA FOR EACH VIEW FOR CURRENT FRAME TO GLOBAL LIST 
    all_images.append(rgb_views)
    all_depths.append(depth_views)
    all_intrinsics.append(camera_intrinsics)
    all_extrinsics.append(camera_extrinsics)
    all_segments.append(seg_views)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_gripper_pos.append(gripper_pos)
    all_cube_pos.append(cube_pos)
    all_points.append(points)
    all_colors.append(colors)

    # ============================================================
    # save the point cloud in PLY format for visualization
    # ============================================================

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(
    #     np.asarray(points)
    # )
    # pcd.colors = o3d.utility.Vector3dVector(
    #     np.asarray(colors.astype(np.float32) / 255.0)
    # )
    # o3d.io.write_point_cloud(
    #     f"{save_dir_ply}/mujoco_pcd_{frame_index}.ply",
    #     pcd
    # )

    # o3d.io.write_point_cloud(
    #     f"{save_dir_ply}/mujoco_pcd_{frame_index}.pcd",
    #     pcd
    # )

    # print(f"Also saved PLYs to {save_dir_ply}\n\n")

# ============================================================
# save the data in npz file for testing
# ============================================================

save_dict = {
        "camera_names": np.array(CAMERA_NAMES),
        "qpos": np.stack(all_qpos),
        "qvel": np.stack(all_qvel),
        "cube_pos": np.stack(all_cube_pos),
        "gripper_pos": np.stack(all_gripper_pos),
        "image": np.stack(all_images),
        "depth": np.stack(all_depths),
        "segmentation": np.stack(all_segments),
        "extrinsics": np.stack(all_extrinsics),
        "intrinsics": np.stack(all_intrinsics),
        "gt_points": np.stack(all_points),
        "gt_colors": np.stack(all_colors),
    }

print("camera_names:", save_dict["camera_names"].shape)
print("qpos:", save_dict["qpos"].shape)
print("qvel:", save_dict["qvel"].shape)
print("cube_pos:", save_dict["cube_pos"].shape)
print("gripper_pos:", save_dict["gripper_pos"].shape)
print("image:", save_dict["image"].shape)
print("depth:", save_dict["depth"].shape)
print("segmentation:", save_dict["segmentation"].shape)
print("extrinsics:", save_dict["extrinsics"].shape)
print("intrinsics:", save_dict["intrinsics"].shape)
print("gt_points:", save_dict["gt_points"].shape)
print("gt_colors:", save_dict["gt_colors"].shape)
np.savez("outputs/multiview_data.npz", **save_dict)

    # ============================================================
    # save the images for visualization
    # ============================================================
    # save_dir_rgb = "outputs/rgb_views"
    # os.makedirs(save_dir_rgb, exist_ok=True)

    # for i, img in enumerate(rgb_views):
    #     # img: (H, W, 3), uint8
    #     Image.fromarray(img).save(f"{save_dir_rgb}/view_{i}.png")

    # save_dir_depth = "outputs/depth_views"
    # os.makedirs(save_dir_depth, exist_ok=True)
    # for i, d in enumerate(depth_views):
    #     # print(np.isnan(d).any())
    #     # print(np.isinf(d).any())
    #     # d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    #     # d_min, d_max = d.min(), d.max()

    #     # print(f"Depth view {i}: min={d_min}, max={d_max}")
    #     # if d_max > d_min:
    #     #     d_norm = (d - d_min) / (d_max - d_min)
    #     # else:
    #     #     d_norm = np.zeros_like(d)

    #     # d_img = (d_norm * 255).astype(np.uint8)

    #     VIS_NEAR = 0.5
    #     VIS_FAR  = 1.5

    #     d_vis = np.clip(d, VIS_NEAR, VIS_FAR)
    #     d_vis = (d_vis - VIS_NEAR) / (VIS_FAR - VIS_NEAR)

    #     d_img = (255*d_vis).astype(np.uint8)

    #     Image.fromarray(d_img).save(f"{save_dir_depth}/depth_{i}.png")

    # save_dir_seg = "outputs/seg_views"
    # os.makedirs(save_dir_seg, exist_ok=True)

    # for i, seg in enumerate(seg_views):
    #     # img: (H, W, 2), uint32
    #     print("segmentation shape:", seg.shape)
    #     print("segmentation dtype:", seg.dtype)

    #     # object id in seg[:, :, 0]
    #     seg_id = seg[:, :, 0].copy()
    #     unique_ids = np.unique(seg_id)
    #     print(unique_ids)

    #     colors = {}

    #     for j in unique_ids:
    #         if j == 0:
    #             colors[j] = (255, 255, 255)  # id=0 white
    #         elif j == -1:
    #             colors[j] = (0, 0, 0)  # id=-1 black
    #         else:
    #             np.random.seed(int(j))
    #             colors[j] = tuple(np.random.randint(0, 255, 3))

    #     h, w = seg_id.shape
    #     rgb = np.zeros((h, w, 3), dtype=np.uint8)

    #     for k in unique_ids:
    #         rgb[seg_id == k] = colors[k]

    #     Image.fromarray(rgb).save(f"{save_dir_seg}/objid_view__{i}.png")

        # # object type in seg[:, :, 1]
        # seg_type = seg[:, :, 1].copy()
        # unique_types = np.unique(seg_type)
        # print(unique_types)

        # colors = {}

        # for j in unique_types:
        #     if j == 0:
        #         colors[j] = (255, 255, 255)  # type=0 white
        #     elif j == -1:
        #         colors[j] = (0, 0, 0)  # type=-1 black
        #     else:
        #         np.random.seed(int(j))
        #         colors[j] = tuple(np.random.randint(0, 255, 3))

        # h, w = seg_type.shape
        # rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # for k in unique_types:
        #     rgb[seg_type == k] = colors[k]
        # # Image.fromarray(rgb).save("seg.png")
        # Image.fromarray(rgb).save(f"{save_dir_seg}/objtype_view__{i}.png")




    #### ADDING THIS 


    # for view in range(num_views): 
    #     rgb = rgb_views[view]
    #     depth = depth_views[view]
    #     intrinsics = camera_intrinsics[view]
    #     extrinsics = camera_extrinsics[view]

    #     K_inv = np.linalg.inv(intrinsics)

        # colors = []
        # points = []
        # for i in range(H): 
        #     for j in range(W): 

        #         d = depth[i,j]
        #         if d >= 6.999:
        #             continue
        #         pixel = np.array([j, i, 1.0])
        #         ray = K_inv @ pixel
        #         point_cam = d * ray

        #         # fx = intrinsics[0][0]
        #         # fy = intrinsics[1][1]
        #         # cx = intrinsics[0][2]
        #         # cy = intrinsics[1][2]

        #         # d = depth[i][j]

        #         # x = (j - cx) * d / fx 
        #         # y = (i - cy) * d/ fy 
        #         # z = d 

        #         # point_cam = np.array([x,y,z])

        #         point_world = (
        #             extrinsics[:3, :3] @ point_cam
        #             + extrinsics[:3, 3]
        #         )

        #         points.append(point_world)
        #         colors.append(rgb[i, j] / 255.0)
        
        # all_points.extend(points)
        # all_colors.extend(colors)







    # with h5py.File(source_filename, 'r') as f_src:
    #     with h5py.File(target_filename, 'w') as f_tgt:
            
    #         # copy the datasets that we don't want to modify
    #         for key in f_src.keys():
    #             if key != 'pixels':  # to be replaced
    #                 f_src.copy(key, f_tgt)
            
    #         # define the shape and dtype for the new datasets based on existing ones
    #         total_samples = f_src['qpos'].shape[0]  # number of samples, assuming all relevant keys have the same number of samples

    #         # pixels
    #         pixels_shape = (total_samples, 3, 224, 224, 3)  # shape for the new multiview pixels dataset
    #         pixels_dtype = f_src['pixels'].dtype  # dtype for the new datasets (e.g., uint8)

    #         # depth_gt
    #         # depth_gt_shape = 
    #         # depth_gt_dtype =/home/student/users/Public_workspace/le-wm-3DGeom

    #         # cam extrinsics
    #         # cam_ex_shape = 
    #         # cam_ex_dtype =

    #         # cam intrinsics
    #         # cam_in_shape =
    #         # cam_in_dtype =

    #         # create the new datasets in the target file with appropriate shapes and dtypes
    #         f_tgt.create_dataset('pixels', shape=pixels_shape, dtype=pixels_dtype, chunks=(1, 3, 224, 224, 3))
    #         f_tgt.create_dataset('depth_gt', shape=depth_gt_shape, dtype=depth_gt_dtype, chunks=True)
    #         f_tgt.create_dataset('cam_extrinsics', shape=cam_ex_shape, dtype=cam_ex_dtype, chunks=True)
    #         f_tgt.create_dataset('cam_intrinsics', shape=cam_in_shape, dtype=cam_in_dtype, chunks=True)
            
    #         # add new datasets for depth_gt, cam_extrinsics, cam_intrinsics as needed
    #         for i in range(total_samples):
    #             qpos = f_src['qpos'][i]
    #             qvel = f_src['qvel'][i]
                
    #             # generate multiview pixels with ogbench based on qpos and qvel
    #             # generation logic here, e.g., using ogbench to render images from multiple views based on qpos and qvel
    #             three_views = np.stack([img1, img2, img3], axis=0)
    #             f_tgt['pixels'][i] = three_views
                
    #             # extract or compute depth_gt, cam_extrinsics, cam_intrinsics based on qpos and qvel
    #             # depth_gt = ...
    #             # cam_extrinsics = ...
    #             # cam_intrinsics = ...
    #             f_tgt['depth_gt'][i] = depth_gt
    #             f_tgt['cam_extrinsics'][i] = cam_extrinsics
    #             f_tgt['cam_intrinsics'][i] = cam_intrinsics
                
    #             if (i + 1) % 1000 == 0 or (i + 1) == total_samples:
    #                 print(f"read and generated {i + 1}/{total_samples} data...")

    # print(f"finished writing to {target_filename}")

    # with h5py.File(source_filename, 'r') as f:

    #     print("keys:", list(f.keys()))
        
    #     qpos = f['qpos']
    #     qvel = f['qvel']
    #     pixels = f['pixels']
        
    #     # 3. 将数据集转换为 NumPy 数组（真正将数据读入内存）
    #     # data_array = qpos[:] 
        
    #     print("qpos_shape:", qpos.shape)
    #     print("qpos_type:", type(qpos))
    #     print("qpos_dtype:", qpos.dtype)
    #     print("qpos_sample:", qpos[100])  # 打印第一条数据样本
    #     print("qvel_shape:", qvel.shape)
    #     print("qvel_type:", type(qvel))
    #     print("qvel_dtype:", qvel.dtype)
    #     print("qvel_sample:", qvel[100])  # 打印第一条数据样本
    #     print("pixels_shape:", pixels.shape)
    #     print("pixels_type:", type(pixels))
    #     print("pixels_dtype:", pixels.dtype)
        
print("\n\nDONE\n\n")