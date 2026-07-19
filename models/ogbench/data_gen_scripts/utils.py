import numpy as np
import mujoco


def _depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray | None,
    conf_thr: float = 0.0,
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

    pts_all, col_all, atten_all = [], [], []

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

        # recover shape (H,W,3)
        H, W = depth[i].shape

        point_map_hw = np.zeros((H, W, 3), dtype=np.float32)
        point_map_hw[valid] = Xw

        color_map_hw = np.zeros((H, W, 3), dtype=np.uint8)
        color_map_hw[valid] = cols

        attention_mask = valid # (H,W)

        pts_all.append(point_map_hw)
        col_all.append(color_map_hw)
        atten_all.append(attention_mask)

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0), np.concatenate(atten_all, 0)



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
    Return camera intrinsic matrix K (3, 3).
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

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)

    return K



def get_camera_extrinsic(model, data, camera_name):
    """
    World -> Camera extrinsic matrix (4,4).
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
    R_cv_mj = np.diag([1, -1, -1])

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



def extract_data(env, camera_name):
    # rgb
    rgb = env.unwrapped.render(
        camera=camera_name
    )

    H, W = rgb.shape[:2]

    # camera intrinsics
    K = get_camera_intrinsic(
        env.unwrapped.model,
        camera_name,
        W,
        H,
    )

    # camera extrinsics
    T = get_camera_extrinsic(
        env.unwrapped.model,
        env.unwrapped.data,
        camera_name,
    )

    depth = env.unwrapped.render(
        camera=camera_name,
        depth=True,
    )
    return rgb, K, T, depth
