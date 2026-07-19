

import open3d as o3d 
import numpy as np
import copy 

MUJOCO_GT_PLY = "/home/student/users/Public_workspace/ply_views/mujoco_pcd.pcd"
DA3_PRED_PLY  = "experiments/quick_outputs/test_inf.pcd"

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

src = o3d.io.read_point_cloud(MUJOCO_GT_PLY)
tgt = o3d.io.read_point_cloud(DA3_PRED_PLY)

bbox = src.get_axis_aligned_bounding_box()
xmin, ymin, zmin = bbox.min_bound
xmax, ymax, zmax = bbox.max_bound

bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[xmin, ymin, zmin],
    max_bound=[xmax, ymax, zmax]
)

gt_crop = src.crop(bbox)
pred_crop = tgt.crop(bbox)

threshold = 0.05

c_gt = gt_crop.get_center()
c_pred = pred_crop.get_center()

trans_init = np.eye(4)
trans_init[:3, 3] = c_gt - c_pred


reg_p2p = o3d.pipelines.registration.registration_icp(
        pred_crop, gt_crop, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
# draw_registration_result(src, tgt, reg_p2p.transformation)

gt_extent = gt_crop.get_axis_aligned_bounding_box().get_extent()
pred_extent = pred_crop.get_axis_aligned_bounding_box().get_extent()

print(gt_extent)
print(pred_extent)