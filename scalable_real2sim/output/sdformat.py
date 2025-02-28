"""
Based on our script at
https://github.com/RussTedrake/manipulation/blob/master/manipulation/create_sdf_from_mesh.py
"""

import argparse
import logging
import os

from pathlib import Path
from typing import List

import coacd
import numpy as np
import trimesh

from lxml import etree as ET


def perform_convex_decomposition(
    mesh: trimesh.Trimesh,
    mesh_parts_dir_name: str,
    mesh_dir: Path,
    preview_with_trimesh: bool,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
) -> List[Path]:
    """Given a mesh, performs a convex decomposition of it with either VHACD or CoACD.
    The resulting convex parts are saved in a subfolder named `<mesh_filename>_parts`.

    Args:
        mesh (trimesh.Trimesh): The mesh to decompose.
        mesh_parts_dir_name (str): The name of the mesh parts directory.
        mesh_dir (Path): The path to the directory that the mesh is stored in. This is
        used for creating the mesh parts directory.
        preview_with_trimesh (bool): Whether to open (and block on) a window to preview
        the decomposition.
        use_coacd (bool): Whether to use CoACD instead of VHACD for decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.

    Returns:
        List[Path]: The paths of the convex pieces.
    """
    # Create a subdir for the convex parts
    out_dir = mesh_dir / mesh_parts_dir_name
    os.makedirs(out_dir, exist_ok=True)

    if preview_with_trimesh:
        logging.info(
            "Showing mesh before convex decomposition. Close the window to proceed."
        )
        mesh.show()

    logging.info(
        "Performing convex decomposition. This might take a couple of minutes for "
        + "complicated meshes and fine resolution settings."
    )
    try:
        # Create a copy of the mesh for decomposition.
        mesh_copy = trimesh.Trimesh(
            vertices=mesh.vertices.copy(), faces=mesh.faces.copy()
        )
        if use_coacd:
            coacd.set_log_level("error")
            coacd_mesh = coacd.Mesh(mesh_copy.vertices, mesh_copy.faces)
            coacd_result = coacd.run_coacd(coacd_mesh, **(coacd_kwargs or {}))
            # Convert CoACD result to trimesh objects.
            convex_pieces = []
            for vertices, faces in coacd_result:
                piece = trimesh.Trimesh(vertices, faces)
                convex_pieces.append(piece)
        else:
            vhacd_settings = vhacd_kwargs or {}
            convex_pieces = mesh_copy.convex_decomposition(**vhacd_settings)
            if not isinstance(convex_pieces, list):
                convex_pieces = [convex_pieces]
    except Exception as e:
        logging.error(f"Problem performing decomposition: {e}")
        exit(1)

    if preview_with_trimesh:
        # Display the convex decomposition, giving each a random colors
        for part in convex_pieces:
            this_color = trimesh.visual.random_color()
            part.visual.face_colors[:] = this_color
        scene = trimesh.scene.scene.Scene()
        for part in convex_pieces:
            scene.add_geometry(part)

        logging.info(
            f"Showing the mesh convex decomposition into {len(convex_pieces)} parts. "
            + "Close the window to proceed."
        )
        scene.show()

    convex_piece_paths: List[Path] = []
    for i, part in enumerate(convex_pieces):
        piece_name = f"convex_piece_{i:03d}.obj"
        path = out_dir / piece_name
        part.export(path)
        convex_piece_paths.append(path)

    return convex_piece_paths


def create_sdf(
    model_name: str,
    mesh_parts_dir_name: str,
    output_path: Path,
    visual_mesh_path: Path,
    collision_mesh_path: Path,
    mass: float,
    center_of_mass: np.ndarray,
    moment_of_inertia: np.ndarray,
    use_hydroelastic: bool = False,
    is_compliant: bool = False,
    hydroelastic_modulus: float | None = None,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = None,
    mu_static: float | None = None,
    preview_with_trimesh: bool = False,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
) -> None:
    """Performs convex decomposition of the collision mesh and adds it to the SDFormat
    file with all other input properties.

    Args:
        model_name (str): The name of the model. The link will be named
        `<model_name>_body_link`.
        mesh_parts_dir_name (str): The name of the mesh parts directory.
        output_path (Path): The path to the output SDFormat file. Must end in `.sdf`.
        visual_mesh_path (Path): The path to the mesh that will be used as the visual
        geometry.
        collision_mesh_path (Path): The path to the mesh that will be used for convex
        decomposition into collision pieces. NOTE that this mesh is expected to
        align with the visual mesh.
        mass (float): The mass in kg of the mesh.
        center_of_mass (np.ndarray): The center of mass of the mesh, expressed in the
        mesh's local frame.
        moment_of_inertia (np.ndarray): The moment of inertia of the mesh expressed in
        the mesh's local frame and about the center of mass.
        use_hydroelastic (bool): Whether to use Hydroelastic contact by adding Drake
        specific tags to the SDFormat file.
        is_compliant (bool): Whether the SDFormat file will be used for compliant
        Hydroelastic simulations. The object will behave as rigid Hydroelastic if this
        is not specified.
        hydroelastic_modulus (float): The Hydroelastic Modulus. This is only used if
        `is_compliant` is True. The default value leads to low compliance. See
        https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for how
        to pick a value.
        hunt_crossley_dissipation (Union[float, None]): The optional Hydroelastic
        Hunt-Crossley dissipation (s/m). See
        https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for how
        to pick a value.
        mu_dynamic (Union[float, None]): The coefficient of dynamic friction.
        mu_static (Union[float, None]): The coefficient of static friction.
        preview_with_trimesh (bool): Whether to open (and block on) a window to preview
        the decomposition.
        use_coacd (bool): Whether to use CoACD instead of VHACD for convex decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.
    """
    # Handle string paths.
    visual_mesh_path = Path(visual_mesh_path)
    collision_mesh_path = Path(collision_mesh_path)
    output_path = Path(output_path)

    # Validate input.
    if not output_path.suffix == ".sdf":
        raise ValueError("Output path must end in `.sdf`.")
    if (use_coacd and vhacd_kwargs is not None) or (
        not use_coacd and coacd_kwargs is not None
    ):
        raise ValueError("Cannot use both CoACD and VHACD.")

    # Generate the SDFormat headers
    root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=model_name)
    link_item = ET.SubElement(model_item, "link", name=f"{model_name}_body_link")
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Add the physical properties
    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = str(mass)
    com_item = ET.SubElement(inertial_item, "pose")
    com_item.text = (
        f"{center_of_mass[0]:.5f} {center_of_mass[1]:.5f} {center_of_mass[2]:.5f} 0 0 0"
    )
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            item.text = f"{moment_of_inertia[i, j]:.5e}"

    # Add the original mesh as the visual mesh
    visual_mesh_path = visual_mesh_path.relative_to(output_path.parent)
    visual_item = ET.SubElement(link_item, "visual", name="visual")
    geometry_item = ET.SubElement(visual_item, "geometry")
    mesh_item = ET.SubElement(geometry_item, "mesh")
    uri_item = ET.SubElement(mesh_item, "uri")
    uri_item.text = visual_mesh_path.as_posix()

    # Compute the convex decomposition and use it as the collision geometry
    collision_mesh = trimesh.load(
        collision_mesh_path, skip_materials=True, force="mesh"
    )
    mesh_piece_paths = perform_convex_decomposition(
        mesh=collision_mesh,
        mesh_parts_dir_name=mesh_parts_dir_name,
        mesh_dir=output_path.parent,
        preview_with_trimesh=preview_with_trimesh,
        use_coacd=use_coacd,
        coacd_kwargs=coacd_kwargs,
        vhacd_kwargs=vhacd_kwargs,
    )
    for i, mesh_piece_path in enumerate(mesh_piece_paths):
        mesh_piece_path = mesh_piece_path.relative_to(output_path.parent)
        collision_item = ET.SubElement(
            link_item, "collision", name=f"collision_{i:03d}"
        )
        geometry_item = ET.SubElement(collision_item, "geometry")
        mesh_item = ET.SubElement(geometry_item, "mesh")
        uri_item = ET.SubElement(mesh_item, "uri")
        uri_item.text = mesh_piece_path.as_posix()
        ET.SubElement(mesh_item, "{drake.mit.edu}declare_convex")

        if use_hydroelastic:
            # Add proximity properties
            proximity_item = ET.SubElement(
                collision_item, "{drake.mit.edu}proximity_properties"
            )
            if is_compliant:
                ET.SubElement(proximity_item, "{drake.mit.edu}compliant_hydroelastic")
                hydroelastic_moulus_item = ET.SubElement(
                    proximity_item, "{drake.mit.edu}hydroelastic_modulus"
                )
                hydroelastic_moulus_item.text = f"{hydroelastic_modulus:.3e}"
            else:
                ET.SubElement(proximity_item, "{drake.mit.edu}rigid_hydroelastic")
            if hunt_crossley_dissipation is not None:
                hunt_crossley_dissipation_item = ET.SubElement(
                    proximity_item, "{drake.mit.edu}hunt_crossley_dissipation"
                )
                hunt_crossley_dissipation_item.text = f"{hunt_crossley_dissipation:.3f}"
            if mu_dynamic is not None:
                mu_dynamic_item = ET.SubElement(
                    proximity_item, "{drake.mit.edu}mu_dynamic"
                )
                mu_dynamic_item.text = f"{mu_dynamic:.3f}"
            if mu_static is not None:
                mu_static_item = ET.SubElement(
                    proximity_item, "{drake.mit.edu}mu_static"
                )
                mu_static_item.text = f"{mu_static:.3f}"

    logging.info(f"Writing SDF to {output_path}")
    ET.ElementTree(root_item).write(output_path, pretty_print=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Drake-compatible SDFormat file for a triangle mesh."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to mesh file.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        required=True,
        help="The mass in kg of the object that is represented by the mesh. This is "
        + "used for computing the moment of inertia.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to convert the specified mesh's coordinates to meters.",
    )
    parser.add_argument(
        "--compliant",
        action="store_true",
        help="Whether the SDFormat file will be used for compliant Hydroelastic "
        + "simulations. The object will behave as rigid Hydroelastic if this is not "
        + "specified.",
    )
    parser.add_argument(
        "--hydroelastic_modulus",
        type=float,
        default=1.0e8,
        help="The Hydroelastic Modulus. This is only used if --compliant is specified. "
        + "The default value leads to low compliance. See "
        + "https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for "
        + "how to pick a value.",
    )
    parser.add_argument(
        "--hunt_crossley_dissipation",
        type=float,
        default=None,
        help="The Hydroelastic Hunt-Crossley dissipation (s/m). See "
        + "https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for "
        + "how to pick a value.",
    )
    parser.add_argument(
        "--mu_dynamic",
        type=float,
        default=None,
        help="The coefficient of dynamic friction.",
    )
    parser.add_argument(
        "--mu_static",
        type=float,
        default=None,
        help="The coefficient of static friction.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Whether to preview the decomposition.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    # Create argument groups for VHACD and CoACD.
    vhacd_group = parser.add_argument_group("VHACD parameters")
    coacd_group = parser.add_argument_group("CoACD parameters")

    parser.add_argument(
        "--use_coacd",
        action="store_true",
        help="Use CoACD instead of VHACD for convex decomposition.",
    )

    # CoACD arguments.
    coacd_group.add_argument(
        "--threshold",
        type=float,
        help="CoACD threshold parameter for determining concavity.",
    )
    coacd_group.add_argument(
        "--preprocess_resolution",
        type=int,
        help="Resolution used in preprocessing step.",
    )
    coacd_group.add_argument(
        "--coacd_resolution",
        type=int,
        help="Main resolution parameter for decomposition.",
    )
    coacd_group.add_argument(
        "--mcts_nodes",
        type=int,
        help="Number of nodes for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--mcts_iterations",
        type=int,
        help="Number of iterations for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--mcts_max_depth",
        type=int,
        help="Maximum depth for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--preprocess_mode",
        type=str,
        default="auto",
        choices=["auto", "voxel", "sampling"],
        help="CoACD preprocess mode.",
    )
    coacd_group.add_argument(
        "--pca", action="store_true", help="Enable PCA pre-processing."
    )

    # VHACD arguments.
    vhacd_group.add_argument(
        "--vhacd_resolution",
        type=int,
        default=10000000,
        help="VHACD voxel resolution.",
    )
    vhacd_group.add_argument(
        "--maxConvexHulls",
        type=int,
        default=64,
        help="VHACD maximum number of convex hulls/ mesh pieces.",
    )
    vhacd_group.add_argument(
        "--minimumVolumePercentErrorAllowed",
        type=float,
        default=1.0,
        help="VHACD minimum allowed volume percentage error.",
    )
    vhacd_group.add_argument(
        "--maxRecursionDepth",
        type=int,
        default=10,
        help="VHACD maximum recursion depth.",
    )
    vhacd_group.add_argument(
        "--no_shrinkWrap",
        action="store_true",
        help="Whether or not to shrinkwrap the voxel positions to the source mesh on "
        + "output.",
    )
    vhacd_group.add_argument(
        "--fillMode",
        type=str,
        default="flood",
        choices=["flood", "raycast", "surface"],
        help="VHACD maximum recursion depth.",
    )
    vhacd_group.add_argument(
        "--maxNumVerticesPerCH",
        type=int,
        default=64,
        help="VHACD maximum number of triangles per convex hull.",
    )
    vhacd_group.add_argument(
        "--no_asyncACD",
        action="store_true",
        help="Whether or not to run VHACD asynchronously, taking advantage of "
        + "additional cores.",
    )
    vhacd_group.add_argument(
        "--minEdgeLength",
        type=int,
        default=2,
        help="VHACD minimum voxel patch edge length.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    # Separate VHACD and CoACD parameters.
    vhacd_params = (
        {
            "resolution": args.vhacd_resolution,
            "maxConvexHulls": args.maxConvexHulls,
            "minimumVolumePercentErrorAllowed": args.minimumVolumePercentErrorAllowed,
            "maxRecursionDepth": args.maxRecursionDepth,
            "shrinkWrap": not args.no_shrinkWrap,
            "fillMode": args.fillMode,
            "maxNumVerticesPerCH": args.maxNumVerticesPerCH,
            "asyncACD": not args.no_asyncACD,
            "minEdgeLength": args.minEdgeLength,
        }
        if not args.use_coacd
        else None
    )
    coacd_params = {}
    for param in [
        "threshold",
        "preprocess_resolution",
        "coacd_resolution",
        "mcts_nodes",
        "mcts_iterations",
        "mcts_max_depth",
        "preprocess_mode",
    ]:
        value = getattr(args, param)
        if value is not None:
            key = "resolution" if param == "coacd_resolution" else param
            coacd_params[key] = value

    # create_sdf_from_mesh(
    #     mesh_path=mesh_path,
    #     mass=mass,
    #     scale=args.scale,
    #     is_compliant=is_compliant,
    #     hydroelastic_modulus=hydroelastic_modulus,
    #     hunt_crossley_dissipation=hunt_crossley_dissipation,
    #     mu_dynamic=mu_dynamic,
    #     mu_static=mu_static,
    #     preview_with_trimesh=args.preview,
    #     use_coacd=args.use_coacd,
    #     coacd_kwargs=coacd_params,
    #     vhacd_kwargs=vhacd_params,
    # )
