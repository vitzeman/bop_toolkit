"""
Converts the meshes from quad to triangle and adjusts the meshes to be in the same scale and origin.
Also computes models_info.json file with the information of the meshes i.e. diameter, bounding box, etc.


Usage:
TODO
"""


import os
import json

import numpy as np
import open3d as o3d
import meshio as mio
import pymeshlab

import scipy.spatial as spatial


def mesh_quad2tri(path2mesh, path2save=None):
    """Converts a mesh from quad to triangle.

    Args:
        path2mesh (str): Path to the mesh.
        path2save (str, optional): Path to save. If None, it will overwrite the input mesh. Defaults to None.
    """    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path2mesh)
    ms.apply_filter("meshing_poly_to_tri")

    path2save = path2mesh if path2save is None else path2save
    ms.save_current_mesh(path2save)

def adjust_meshes(path2mesh, path2save=None, scale=1000):
    """Adjusts the meshes to be in the same scale and origin.

    Args:
        path2mesh (str): Path to the mesh.
        path2save (str, optional): Path to save. If None, it will overwrite the input mesh. Defaults to None.
    """    
    mesh = o3d.io.read_triangle_mesh(path2mesh)
    mesh.compute_vertex_normals()

    # scale the mesh
    mesh.scale(scale, center=(0, 0, 0))

    path2save = path2mesh if path2save is None else path2save
    o3d.io.write_triangle_mesh(path2save, mesh)

def get_diameter(mesh) -> float:
    """Computes the diameter of the mesh. Diameter is defined as the maximal distance between any pair of points in the

    Args:
        mesh_path (str): Path to the mesh file

    Returns:
        float: The diameter of the mesh in mesh units
    """
    # Compute the convex hull of the mesh to decrease the number og points
    convex_hull = mesh.compute_convex_hull()[0]
    points = np.asarray(convex_hull.vertices)

    # Compute the pairwise distances between the points
    distances = spatial.distance.pdist(points)
    diameter = np.max(distances)

    return diameter


def get_mesh_info(path2mesh):
    """Gets the information of the mesh.

    Args:
        path2mesh (str): Path to the mesh.

    Returns:
        dict: Dictionary with the information of the mesh.
    """    
    mesh = o3d.io.read_triangle_mesh(path2mesh)

    diameter = get_diameter(mesh)
    bbox3d = mesh.get_axis_aligned_bounding_box()
    min_x, min_y, min_z = bbox3d.get_min_bound()
    max_x, max_y, max_z = bbox3d.get_max_bound()
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    
    mesh_info = {
        "diameter": diameter,
        "min_x": min_x, 
        "min_y": min_y,
        "min_z": min_z,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "symmetries_discrete": [],
        "symmetries_continuous": [],        
    }

    return mesh_info

if __name__ == "__main__":
    PATH2DIRS = "/home/testbed/Projects/bop_toolkit/clearpose_downsample_100/model"
    models_info = {

    }
    scale = 1/1000
    for directory in sorted(os.listdir(PATH2DIRS)):
        # check if it is a directory
        if not os.path.isdir(os.path.join(PATH2DIRS, directory)):
            print(f"Skipping: {directory} - not a directory")
            continue
        file_name = f"{directory}.obj"
        path2mesh = os.path.join(PATH2DIRS, directory, file_name)
        print(f"Processing: {path2mesh}")
        # mesh_quad2tri(path2mesh) # Already done 
        adjust_meshes(path2mesh, scale=scale)
        mesh_info = get_mesh_info(path2mesh)
        models_info[directory] = mesh_info
        break
    
    # with open(os.path.join(PATH2DIRS, "models_info.json"), "w") as f:
        # json.dump(models_info, f, indent=2)
    
