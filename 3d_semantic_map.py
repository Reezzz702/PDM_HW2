import numpy as np
import os
import open3d as o3d
import copy
from tqdm import tqdm
from argparse import ArgumentParser
import csv


SIZE = 512
fov = 90 / 180 * np.pi
f = SIZE / (2 * np.tan(fov / 2.0))
depth_scale = 1000
K = np.array([[f, 0, SIZE / 2], [0, f, SIZE / 2], [0, 0, 1]])
K_inverse = np.linalg.inv(K)


def transform_depth(image):
    img = np.asarray(image, dtype=np.float32)
    depth_img = img / 255 * 10
    depth_img = o3d.geometry.Image(depth_img)
    return depth_img


def depth_to_point_cloud(rgb, depth):
    colors = np.zeros((512*512, 3))
    points = np.zeros((512*512, 3))
    u = np.array([range(512)]*512).reshape(512,512) - 256
    v = np.array([[i]*512 for i in range(512)]).reshape(512,512) - 256
    z = np.asarray(depth)
    colors = (np.asarray(rgb)/255).reshape(512*512, 3)
    points[:, 0] = (u * z / f).reshape(512*512)
    points[:, 1] = (v * z / f).reshape(512*512)
    points[:, 2] = z.reshape(512*512)
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.select_by_index(np.where(points[:, 2] != 0)[0])
    pcd.transform(np.array(([[1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])))
    #o3d.visualization.draw_geometries([pcd])
    return pcd


def voxel_down_sample(pcd, voxel_size):
    # TODO
    # determine a label of each voxel (e.g., the label could be the majority class in a voxel).

    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(
    source, target, voxel_size, global_transformation
):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        global_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def ICP(floor, data_size, voxel_size, seg, model):
    transformation_seq = []
    whole_scene_pcd = o3d.geometry.PointCloud()
    prev_pcd_down = None
    prev_pcd_fpfh = None
    for i in tqdm(range(data_size - 1)):
        if i == 0:
            target_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}_{i + 1}.png")
            target_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth_{i + 1}.png")
            )
            
            source_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}_{i + 1}.png")
            source_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth_{i + 2}.png")
            )

            target_pcd = depth_to_point_cloud(target_seg, target_depth)
            target_points = np.asarray(target_pcd.points)
            target_pcd = target_pcd.select_by_index(np.where(target_points[:, 1] < 0.5)[0])

            source_pcd = depth_to_point_cloud(source_seg, source_depth)
            source_points = np.asarray(source_pcd.points)
            source_pcd = source_pcd.select_by_index(np.where(source_points[:, 1] < 0.5)[0])

            target_pcd_down, target_pcd_fpfh = voxel_down_sample(
                target_pcd, voxel_size
            )
            source_pcd_down, source_pcd_fpfh = voxel_down_sample(
                source_pcd, voxel_size
            )

        else:
            source_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}_{i + 2}.png")
            source_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth_{i + 2}.png")
            )
            source_pcd = depth_to_point_cloud(source_seg, source_depth)
            source_points = np.asarray(source_pcd.points)
            source_pcd = source_pcd.select_by_index(np.where(source_points[:, 1] < 0.5)[0])

            source_pcd_down, source_pcd_fpfh = voxel_down_sample(
                source_pcd, voxel_size
            )

            target_pcd_down = copy.deepcopy(prev_pcd_down)
            target_pcd_fpfh = copy.deepcopy(prev_pcd_fpfh)

        prev_pcd_down = copy.deepcopy(source_pcd_down)
        prev_pcd_fpfh = copy.deepcopy(source_pcd_fpfh)
        # o3d.visualization.draw_geometries([source_pcd_down])

        # applt globla registration
        result_ransac = execute_global_registration(
            source_pcd_down,
            target_pcd_down,
            source_pcd_fpfh,
            target_pcd_fpfh,
            voxel_size,
        )

        # apply local registration(ICP)
        result_icp = refine_registration(
                source_pcd_down,
                target_pcd_down,
                source_pcd_fpfh,
                target_pcd_fpfh,
                voxel_size,
                result_ransac.transformation,
            )
        result_transformation = result_icp.transformation

        # T_t+1 = T_t @ currentT_t+1
        if i == 0:
            whole_scene_pcd += target_pcd
            cur_transformation = result_transformation
        else:
            cur_transformation = transformation_seq[-1] @ result_transformation

        # apply transform to project current pcd to original coordinate system
        source_pcd.transform(cur_transformation)
        transformation_seq.append(cur_transformation)
        # accumulate pcd
        whole_scene_pcd += source_pcd

    # record reconstructed poses
    reconstruct_pose = [[0, 0, 0]]
    reconstruct_link = []
    for i, t in enumerate(transformation_seq):
        translation = [t[0, -1], t[1, -1], t[2, -1]]
        reconstruct_pose.append(translation)
        reconstruct_link.append([i, i + 1])
    reconstruct_link.pop(-1)

    
    # create reconstructed lineset
    colors = [[1, 0, 0] for _ in range(len(reconstruct_link))]
    reconstruct_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(reconstruct_pose),
        lines=o3d.utility.Vector2iVector(reconstruct_link),
    )
    reconstruct_line_set.colors = o3d.utility.Vector3dVector(colors)
    # show result
    return whole_scene_pcd, reconstruct_line_set, reconstruct_pose


def main(args):
    # parameters
    voxel_size = args.voxel_size
    data_size = len(os.listdir(f"data/floor{args.floor}/depth"))

    GT_pose = []
    GT_link = []
    with open(f"data/floor{args.floor}/GT.csv", "r") as f:
        csvreader = csv.reader(f)
        first_row = next(csvreader)
        first_coor = None
        count = 1
        for i in range(3):
            first_row[i] = float(first_row[i])
            first_coor = np.asarray(first_row[:3], dtype=np.float32)
            coor = first_coor - first_coor
        GT_pose.append(coor)

        for row in csvreader:
            for i in range(3):
                row[i] = float(row[i])
                coor = np.asarray(row[:3], dtype=np.float32)
                coor -= first_coor
            GT_pose.append(coor)
            GT_link.append([count - 1, count])
            count += 1
        f.close()

    # create GT lineset
    colors = [[0, 0, 0] for _ in range(len(GT_link))]
    GT_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(GT_pose),
        lines=o3d.utility.Vector2iVector(GT_link),
    )
    GT_line_set.colors = o3d.utility.Vector3dVector(colors)
    
    pcd, recconstruction_line_set, reconstruct_pose = ICP(args.floor, data_size, voxel_size, args.seg, args.model)

    o3d.visualization.draw_geometries(
        [pcd, recconstruction_line_set, GT_line_set]
    )

     # calculate mean L2 norm
    GT = np.asarray(GT_pose)
    reconstruct = np.asarray(reconstruct_pose)
    print(np.mean(np.linalg.norm(GT - reconstruct, axis = 1)))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", dest="voxel_size", help="voxel size", type=float)
    parser.add_argument("-f", dest="floor", help="select floor: [0, 1]", type=int)
    parser.add_argument("-s", dest="seg", help="segment source(pred or seg)", type=str)
    parser.add_argument("-m", dest="model", help="select model(apartment_0 or others)", type=str)
    main(parser.parse_args())