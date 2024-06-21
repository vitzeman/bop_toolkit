"""
Showcase of the meshes and their symmetries both continuous and discrete in 3D

USAGE: 
TODO:
"""

import os
import json
import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

COLORS = [
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [1, 0, 1],  # magenta
    [0, 1, 1],  # cyan
]


def visualize_symmetries(
    mesh_path: str, symmetries_discrete: list = [], symmetries_continuous: list = [], cont_step = 45
) -> None:
    """Visualize the discrete symmetries of a mesh

    Args:
        mesh_path (str): Path to the mesh to visualize
        symmetries_discrete (list): List of 4x4 matrices representing the symmetries
            either as a list of 16 elements or as a 4x4 matrix
        symmetries_continuous (list): List of continuous symmetries represented 
            as a dictionary with keys "axis" and "offset"
        cont_step (int): Step for the continuous symmetries to be visualized
    """

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    # mesh.paint_uniform_color([1, 0, 0])

    symetric_meshes = []

    for sym in symmetries_discrete:
        T_mtx = np.array(sym).reshape(4, 4)
        symmetric_mesh = copy.deepcopy(mesh)
        random_color = np.random.rand(3).tolist()
        symmetric_mesh.paint_uniform_color(random_color)
        symmetric_mesh.transform(T_mtx)
        symetric_meshes.append(symmetric_mesh)

    for sym in symmetries_continuous:
        axis = np.array(sym["axis"])
        offset = np.array(sym["offset"])
        for angle in range(cont_step, 360, cont_step):
            T_mtx = np.eye(4)
            T_mtx[:3, :3] = R.from_rotvec(axis * np.deg2rad(angle)).as_matrix()
            T_mtx[:3, 3] = offset
            symmetric_mesh = copy.deepcopy(mesh)
            random_color = np.random.rand(3).tolist()
            symmetric_mesh.paint_uniform_color(random_color)
            symmetric_mesh.transform(T_mtx)
            symetric_meshes.append(symmetric_mesh)


    # visualize the object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # axis frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    vis.add_geometry(mesh)
    for symmetric_mesh in symetric_meshes:
        vis.add_geometry(symmetric_mesh)

    vis.run()


if __name__ == "__main__":
    # tless_model_path = "datasets/megapose-clear/tless_models_cad"
    # selected_obj = "27"
    # object_name = "obj_" + selected_obj.zfill(6) + ".ply"

    # path2obj = os.path.join(tless_model_path, object_name)
    # obj_info_path = os.path.join(tless_model_path, "models_info.json")
    # with open(obj_info_path, "r") as f:
    #     obj_info = json.load(f)

    # obj_info = obj_info[selected_obj]

    path2objs = "clearpose_downsample_100/model"
    path2models_info = os.path.join(path2objs, "models_info.json")

    with open(path2models_info, "r") as f:
        models_info = json.load(f)

    skip_to = 62
    True 
    for e, model in enumerate(models_info):
        if e < skip_to:
            continue
        print("Processing:",e,  model)
        path2obj = os.path.join(path2objs, model, f"{model}.obj")
        symms_disc = models_info[model].get("symmetries_discrete", [])
        symms_cont = models_info[model].get("symmetries_continuous", [])
        visualize_symmetries(path2obj, symms_disc, symms_cont)
        

    # directory_list = sorted(os.listdir(path2objs))
    
    # for directory in directory_list:
    #     symms_disc = []
    #     if not os.path.isdir(os.path.join(path2objs, directory)):
    #         print(f"Skipping: {directory} - not a directory")
    #         continue
    #     file_name = f"{directory}.obj"
    #     path2obj = os.path.join(path2objs, directory, file_name)
    #     print(f"Processing: {path2obj}")

    #     visualize_discrete_symmetries(path2obj, symms_disc)


    # id = 11
    # path2obj = os.path.join(path2objs, directory_list[id])
    # discrete_symmetries = []

    # print("Name of the object:", directory_list[id])

    # symetry_cont_z = {"axis": [0, 0, 1], "offset": [0, 0, 0]}
    # Tz_90 = np.eye(4)
    # Tz_90[:3, :3] = R.from_euler("z", 90, degrees=True).as_matrix()
    # discrete_symmetries = []
    # discrete_symmetries.append(Tz_90.flatten().tolist())
    # Tz_180 = np.eye(4)
    # Tz_180[:3, :3] = R.from_euler("z", 180, degrees=True).as_matrix()
    # discrete_symmetries.append(Tz_180.flatten().tolist())
    # Tz_270 = np.eye(4)
    # Tz_270[:3, :3] = R.from_euler("z", 270, degrees=True).as_matrix()
    # discrete_symmetries.append(Tz_270.flatten().tolist())

    # Ty_180 = np.eye(4)
    # Ty_180[:3, :3] = R.from_euler("y", 180, degrees=True).as_matrix()
    # discrete_symmetries.append(Ty_180.flatten().tolist())

    # visualize_discrete_symmetries(path2obj, discrete_symmetries)
    # print("Number of discrete symmetries:", len(discrete_symmetries))
    # print('    "symmetries_discrete":', discrete_symmetries)
    # print(obj_info)

    # # T_mtx = np.eye(4)
    # mesh = o3d.io.read_triangle_mesh(path2obj)
    # mesh.compute_vertex_normals()
    # mesh.compute_triangle_normals()
    # # pa
    # mesh.paint_uniform_color([1, 0, 0])

    # symetric_meshes = []

    # for sym in obj_info["symmetries_discrete"]:
    #     T_mtx = np.array(sym).reshape(4, 4)
    #     symmetric_mesh = copy.deepcopy(mesh)
    #     random_color = np.random.rand(3).tolist()
    #     symmetric_mesh.paint_uniform_color(random_color)
    #     symmetric_mesh.transform(T_mtx)
    #     symetric_meshes.append(symmetric_mesh)

    # # visualize the object
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # axis frame
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=100, origin=[0, 0, 0]
    # )
    # vis.add_geometry(coord_frame)
    # vis.add_geometry(mesh)
    # for symmetric_mesh in symetric_meshes:
    #     vis.add_geometry(symmetric_mesh)

    # vis.run()
