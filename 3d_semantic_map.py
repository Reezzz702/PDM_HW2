import numpy as np
import os
import open3d as o3d
import copy
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors


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


def custom_voxel_down_sample(pcd, voxel_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    voxel_index = points // voxel_size # calculate voxel index

    voxel_index = np.unique(voxel_index, axis=0) # remove duplicate index
    nbrs = NearestNeighbors(n_neighbors=(len(points) // (len(voxel_index))), algorithm='ball_tree').fit(points) # build nearest neighbor tree

    voxel_colors = []
    voxel_points = []

    # for every voxel index:
    #   1. project back to original coordinate space
    #   2. find k-nearest point in original point cloud
    #   3. record the frequency of each color and find the majority color
    #   4. color of this voxel -> the majority color
    for index in voxel_index:
        point = index * voxel_size
        voxel_points.append(point)

        distance, indices = nbrs.kneighbors([point])
        color_count = {}
        max_count = 0
        max_color = None
        for i in indices[0]:
            color_count[colors[i].tobytes()] = color_count.get(colors[i].tobytes(), 0) + 1
            if color_count[colors[i].tobytes()] > max_count:
                max_count = color_count[colors[i].tobytes()]
                max_color = colors[i]
        voxel_colors.append(max_color)
    
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(voxel_points)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_colors)

    return pcd_down


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
    for i in tqdm(range(1, data_size)):
        if i == 1:
            target_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}{i}.png")
            target_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth{i}.png")
            )
            
            source_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}{i + 1}.png")
            source_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth{i + 1}.png")
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
            source_seg = o3d.io.read_image(f"data/floor{floor}/{model}/{seg}/{seg}{i + 1}.png")

            source_depth = transform_depth(
                o3d.io.read_image(f"data/floor{floor}/depth/depth{i + 1}.png")
            )
            source_pcd = depth_to_point_cloud(source_seg, source_depth)
            source_points = np.asarray(source_pcd.points)
            source_pcd = source_pcd.select_by_index(np.where(source_points[:, 1] < 0.5)[0])

            source_pcd_down, source_pcd_fpfh = voxel_down_sample(source_pcd, voxel_size)

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
                voxel_size,
                result_ransac.transformation,
            )
        result_transformation = result_icp.transformation

        # T_t+1 = T_t @ currentT_t+1
        if i == 1:
            whole_scene_pcd += target_pcd
            cur_transformation = result_transformation
        else:
            cur_transformation = transformation_seq[-1] @ result_transformation

        # apply transform to project current pcd to original coordinate system
        source_pcd_down.transform(cur_transformation)
        transformation_seq.append(cur_transformation)
        # accumulate pcd
        whole_scene_pcd += source_pcd_down

    return whole_scene_pcd


def main(args):
    # parameters
    voxel_size = args.voxel_size
    data_size = len(os.listdir(f"data/floor{args.floor}/depth"))
    
    pcd = ICP(args.floor, data_size, voxel_size, args.seg, args.model)
    pcd = custom_voxel_down_sample(pcd, 0.1)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", dest="voxel_size", help="voxel size", type=float)
    parser.add_argument("-f", dest="floor", help="select floor: [1, 2]", type=int)
    parser.add_argument("-s", dest="seg", help="segment source(pred or ground)", type=str)
    parser.add_argument("-m", dest="model", help="select model(apartment_0 or others)", type=str)
    main(parser.parse_args())