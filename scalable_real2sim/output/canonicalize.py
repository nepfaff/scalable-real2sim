from typing import Tuple

import numpy as np
import open3d as o3d
import trimesh

from scipy.spatial.transform import Rotation


def get_mesh_rotation_pca(vertices: np.ndarray) -> np.ndarray:
    """
    Returns a rotation matrix for rotating the vertices' principle component onto the
    z-axis and the minor component onto the x-axis.
    """
    cov = np.cov(vertices.T)
    eigval, eigvec = np.linalg.eig(cov)

    order = eigval.argsort()
    principal_component = eigvec[:, order[-1]]
    minor_component = eigvec[:, order[0]]

    # Rotate mesh to align the principal component with the z-axis and the minor
    # component with the x-axis.
    z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    rot_components_to_axes, _ = Rotation.align_vectors(
        np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
    )
    return rot_components_to_axes.as_matrix()


def get_mesh_rotation_obb(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Returns a rotation matrix for rotating the mesh OBB's longest side onto the z-axis
    and the shortest side onto the
    x-axis.
    """
    obb = (
        mesh.get_oriented_bounding_box()
    )  # Computes the OBB based on PCA of the convex hull
    box_points = np.asarray(obb.get_box_points())
    # The order of the points stays fixed (obtained these vectors from visual analysis)
    largest_vec = box_points[1] - box_points[0]
    smallest_candidate1 = box_points[2] - box_points[0]
    smallest_candidate2 = box_points[3] - box_points[0]
    smallest_vec = (
        smallest_candidate1
        if np.linalg.norm(smallest_candidate1) < np.linalg.norm(smallest_candidate2)
        else smallest_candidate2
    )

    # Rotate mesh to align the OBB's largest side with the z-axis and the smallest side
    # with the x-axis.
    z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    rot_components_to_axes, _ = Rotation.align_vectors(
        np.array([z_axis, x_axis]), np.stack([largest_vec, smallest_vec])
    )
    return rot_components_to_axes.as_matrix()


def axis_align_mesh(
    mesh: o3d.geometry.TriangleMesh,
    viz: bool = False,
    use_obb: bool = False,
    mesh_already_at_origin: bool = False,
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Axis aligned the mesh based on OBB if `use_obb` is true and based on PCA otherwise.
    """
    vertices = np.asarray(mesh.vertices)

    if not mesh_already_at_origin:
        # Put at world origin.
        vertices_at_origin = vertices - np.mean(vertices, axis=0)
    else:
        vertices_at_origin = vertices

    rot = get_mesh_rotation_obb(mesh) if use_obb else get_mesh_rotation_pca(vertices)
    vertices_rotated = vertices_at_origin @ rot.T

    mesh.vertices = o3d.utility.Vector3dVector(vertices_rotated)

    if viz:
        o3d.visualization.draw_geometries(
            [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
        )

    return mesh, rot


def canonicalize_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Center at origin and axis align the mesh."""
    # Center at origin.
    vertices = np.asarray(mesh.vertices)
    mesh_translation = np.mean(vertices, axis=0)
    mesh.translate(-mesh_translation)

    # Axis align the mesh.
    mesh, _ = axis_align_mesh(
        mesh, viz=False, use_obb=True, mesh_already_at_origin=True
    )

    return mesh


def canonicalize_mesh_from_file(mesh_path: str, output_path: str) -> None:
    """Canonicalize a mesh from a file while preserving texture and save the result."""

    # Load mesh with trimesh (preserves texture/materials)
    mesh = trimesh.load(mesh_path, process=False)

    # Convert Trimesh to Open3D (only vertices and faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Apply canonicalization (assumes canonicalize_mesh returns an Open3D mesh)
    canonicalized_mesh = canonicalize_mesh(o3d_mesh)

    # Convert back to Trimesh (preserving original materials)
    canonicalized_trimesh = trimesh.Trimesh(
        vertices=np.asarray(canonicalized_mesh.vertices),
        faces=np.asarray(canonicalized_mesh.triangles),
        visual=mesh.visual,  # Keep original materials and textures
    )

    # Save with texture/materials
    canonicalized_trimesh.export(output_path)
