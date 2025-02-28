"""
Script for computing geometric error metrics between a GT and reconstructed visual
mesh.

The meshes are canonicalized and aligned with ICP before computing the metrics. 
"""

import argparse

import numpy as np
import open3d as o3d

from scipy.spatial import KDTree

from scalable_real2sim.output.canonicalize import canonicalize_mesh


def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute the chamfer distance between two point clouds.
    """
    # Build kd-trees for fast nearest neighbor queries.
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    # Find nearest neighbors in both directions.
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)

    # Compute mean and max chamfer distances.
    distance = (np.mean(dist1) + np.mean(dist2)) / 2.0

    return distance


def sample_points_with_curvature(
    mesh: o3d.geometry.TriangleMesh, num_points: int
) -> np.ndarray:
    """
    Sample points from the mesh surface, weighted by curvature.
    """
    # Compute vertex normals and curvature.
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    curvatures = np.linalg.norm(np.asarray(mesh.vertex_normals), axis=1)

    # Normalize curvatures to use as probabilities.
    probabilities = curvatures / np.sum(curvatures)

    # Weighted random sampling of vertices.
    vertices = np.asarray(mesh.vertices)
    sampled_indices = np.random.choice(len(vertices), size=num_points, p=probabilities)
    sampled_points = vertices[sampled_indices]

    return sampled_points


def sample_points_from_mesh(
    mesh: o3d.geometry.TriangleMesh, num_points: int, use_curvature: bool = False
) -> np.ndarray:
    """
    Sample points from a mesh using either uniform or curvature-based sampling.
    """
    if use_curvature:
        points = sample_points_with_curvature(mesh, num_points)
    else:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
    return points


def prepare_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load and canonicalize a mesh.
    """
    # Load and get main cluster.
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh = canonicalize_mesh(mesh)

    return mesh


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH features for a point cloud.
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def refine_alignment_with_icp(
    mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh
) -> o3d.geometry.TriangleMesh:
    """
    Refine the alignment of two meshes using feature matching and ICP.
    """
    # Sample points and create point clouds.
    source_pcd = mesh1.sample_points_uniformly(number_of_points=5000)
    target_pcd = mesh2.sample_points_uniformly(number_of_points=5000)

    # Voxel downsampling.
    voxel_size = 0.005
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals.
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )

    # Initial alignment using center of mass.
    source_center = source_down.get_center()
    target_center = target_down.get_center()
    initial_translation = target_center - source_center

    init_transform = np.eye(4)
    init_transform[:3, 3] = initial_translation

    # Compute FPFH features.
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    # Global registration using RANSAC.
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=voxel_size * 5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                voxel_size * 5
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500),
    )

    # Use RANSAC result if successful, otherwise use initial translation.
    initial_alignment = (
        result_ransac.transformation if result_ransac.fitness > 0 else init_transform
    )

    # Refine with ICP.
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance=voxel_size * 5,
        init=initial_alignment,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=500,
            relative_fitness=1e-7,
            relative_rmse=1e-7,
        ),
    )

    # Transform the source mesh.
    mesh1.transform(reg_p2p.transformation)
    return mesh1


def compute_f_score(
    points1: np.ndarray, points2: np.ndarray, threshold: float = 0.01
) -> tuple[float, float, float]:
    """
    Compute F-score between two point clouds.
    points1 is treated as prediction, points2 as ground truth.
    """
    tree2 = KDTree(points2)
    tree1 = KDTree(points1)

    # For precision: How many points in points1 (prediction) are close to points2 (GT)?
    dist_p = tree2.query(points1)[0]
    precision = np.mean(dist_p < threshold)

    # For recall: How many points in points2 (GT) are close to points1 (prediction)?
    dist_r = tree1.query(points2)[0]
    recall = np.mean(dist_r < threshold)

    # F-score is the harmonic mean.
    f_score = 2 * precision * recall / (precision + recall + 1e-8)

    return f_score, precision, recall


def compute_normal_consistency(
    mesh1: o3d.geometry.TriangleMesh,
    mesh2: o3d.geometry.TriangleMesh,
    num_points: int = 10000,
) -> float:
    """
    Compute normal consistency between two meshes.
    Samples points and their normals, then compares normal directions at closest points.
    """
    # Compute vertex normals for both meshes.
    mesh1.compute_vertex_normals()
    mesh2.compute_vertex_normals()

    # Sample points and get their normals directly from the mesh.
    pcd1 = mesh1.sample_points_poisson_disk(num_points)
    pcd2 = mesh2.sample_points_poisson_disk(num_points)

    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    normals1 = np.asarray(pcd1.normals)
    normals2 = np.asarray(pcd2.normals)

    # Find closest point correspondences.
    tree = KDTree(points2)
    _, indices = tree.query(points1)

    # Compute absolute dot product between corresponding normals.
    normal_consistency = np.mean(np.abs(np.sum(normals1 * normals2[indices], axis=1)))

    return normal_consistency


def compute_iou(
    mesh1: o3d.geometry.TriangleMesh,
    mesh2: o3d.geometry.TriangleMesh,
    resolution: int = 100,
) -> float:
    """
    Compute IoU (Intersection over Union) using voxel occupancy.
    """
    # Compute voxel size based on combined bounds of both meshes
    combined_bounds = np.vstack(
        [
            mesh1.get_min_bound(),
            mesh1.get_max_bound(),
            mesh2.get_min_bound(),
            mesh2.get_max_bound(),
        ]
    )
    combined_size = np.linalg.norm(
        combined_bounds.max(axis=0) - combined_bounds.min(axis=0)
    )
    voxel_size = combined_size / resolution

    # Create voxel grids for both meshes using the same voxel size
    voxel1 = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh1, voxel_size=voxel_size
    )
    voxel2 = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh2, voxel_size=voxel_size
    )

    # Get voxel indices.
    voxels1 = set(
        (v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxel1.get_voxels()
    )
    voxels2 = set(
        (v.grid_index[0], v.grid_index[1], v.grid_index[2]) for v in voxel2.get_voxels()
    )

    # Compute intersection and union.
    intersection = len(voxels1.intersection(voxels2))
    union = len(voxels1.union(voxels2))

    iou = intersection / (union + 1e-8)
    return iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_mesh", type=str, help="Path to the ground truth mesh")
    parser.add_argument("pred_mesh", type=str, help="Path to the predicted mesh")
    parser.add_argument(
        "--num_points",
        type=int,
        default=1000000,
        help="Number of points to sample from each mesh",
    )
    parser.add_argument(
        "--use_curvature",
        action="store_true",
        help="Use curvature-based sampling for the meshes",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Visualize the aligned meshes"
    )
    parser.add_argument(
        "--f_score_threshold",
        type=float,
        default=0.001,
        help="Distance threshold for F-score computation",
    )
    parser.add_argument(
        "--iou_resolution",
        type=int,
        default=25,
        help="Resolution for IoU computation",
    )
    args = parser.parse_args()
    num_points = args.num_points
    use_curvature = args.use_curvature
    f_score_threshold = args.f_score_threshold
    iou_resolution = args.iou_resolution

    # Prepare both meshes.
    gt_mesh = prepare_mesh(args.gt_mesh)
    raw_pred_mesh = prepare_mesh(args.pred_mesh)

    # Refine alignment using ICP.
    pred_mesh = refine_alignment_with_icp(raw_pred_mesh, gt_mesh)

    # Sample points after ICP refinement.
    gt_points = sample_points_from_mesh(gt_mesh, num_points, use_curvature)
    pred_points = sample_points_from_mesh(pred_mesh, num_points, use_curvature)

    # Compute all metrics.
    chamfer_distance = compute_chamfer_distance(pred_points, gt_points)
    f_score, precision, recall = compute_f_score(
        pred_points, gt_points, f_score_threshold
    )
    normal_consistency = compute_normal_consistency(pred_mesh, gt_mesh)
    iou = compute_iou(pred_mesh, gt_mesh, resolution=iou_resolution)

    # Print results.
    print("\nMesh Comparison Metrics:")
    print(f"Chamfer Distance (mm): {chamfer_distance * 1000:.6f}")
    print(f"F-score: {f_score:.6f}")
    print(f"  Precision: {precision:.6f} (% of predicted points matched)")
    print(f"  Recall: {recall:.6f} (% of ground truth points matched)")
    print(f"Normal Consistency: {normal_consistency:.6f}")
    print(f"IoU: {iou:.6f}")

    if args.vis:
        # Create point clouds for visualization.
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pred_points)
        pcd2.points = o3d.utility.Vector3dVector(gt_points)

        pcd1.paint_uniform_color([1, 0, 0])  # Red
        pcd2.paint_uniform_color([0, 0, 1])  # Blue
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd1, pcd2, frame])


if __name__ == "__main__":
    main()
