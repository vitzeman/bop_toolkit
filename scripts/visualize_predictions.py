"""
Scripts that isualizes the prediction of the BOP challange in 3D scene

Usage:
TODO
"""

import json 
import os 
import csv
import copy

import cv2 
import numpy as np
import open3d as o3d
import scipy.io
import keyboard as kb


def make_point_cloud(rgb_img:np.ndarray, depth_img:np.ndarray, cam_K:np.ndarray, depth_scale:int=1) -> o3d.geometry.PointCloud:
    """Converts the rgb and depth image to a point cloud

    Args:
        rgb_img (np.ndarray): RGB image [H, W, 3] np.uint8 BGR
        depth_img (np.ndarray): Depth image [H, W] 
        cam_K (np.ndarray): Camera intrinsics [3, 3]
        depth_scale (int, optional): Scale of the depth image. Defaults to 1.

    Returns:
        o3d.geometry.PointCloud: Pointcloud
    """    
    
    # convert images to open3d types
    rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_img_o3d = o3d.geometry.Image(depth_img * depth_scale)

    # convert image to point cloud
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        rgb_img.shape[0],
        rgb_img.shape[1],
        cam_K[0, 0],
        cam_K[1, 1],
        cam_K[0, 2],
        cam_K[1, 2],
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img_o3d, depth_img_o3d, depth_scale=1, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd

def visualize_geometries(pointcloud, gt_model, pred_model=[], gt_color=[0, 1, 0], pred_color=[1, 0, 1]):
    """ Run visualization of the 2 point clouds and the models in the scene

    Args:
        pointcloud (_type_): Pointcloud of the scene
        gt_model (_type_): Ground truth pose of the models
        pred_model (list, optional): Predictions of the models. Defaults to [].
        gt_color (list, optional): Color of the gt models. Defaults to [0, 1, 0].
        pred_color (list, optional): Color of the prediction models. Defaults to [1, 0, 1].
    """    
    T = np.diag([1, -1, -1, 1]) # DO NOT ASK ME WHY IT HAS TO BE HERE
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0])
    axis.transform(T)

    pointcloud.transform(T)

    list2show = [axis, pointcloud]
    for gt in gt_model:
        gt.transform(T)
        gt.paint_uniform_color(gt_color)
        list2show.append(gt)

    for pred in pred_model:
        pred.transform(T)
        pred.paint_uniform_color(pred_color)
        list2show.append(pred)

    # SOMEHOW CREATE THE RGB RENDERING OF THE SCENE

    # camera = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # # vis.add_geometry(axis)
    # vis.add_geometry(pointcloud)
    # for gt in gt_model:
    #     vis.add_geometry(gt)
    # for pred in pred_model:
    #     vis.add_geometry(pred)
    # img = vis.capture_screen_float_buffer(True)
    # print(np.array(img).shape)
    # cv2.imshow("frame", np.array(img)[:,:,::-1])
    # cv2.waitKey(0)



    o3d.visualization.draw_geometries(list2show)


def test_clearGrasp_csv():
    path2dataset = "/home/testbed/Projects/bop_toolkit/clearGrasp/test"
    path2csv = "/home/testbed/Projects/bop_toolkit/clearGrasp/results_csv/mesh_clearGrasp-test.csv"
    path2models = "/home/testbed/Projects/bop_toolkit/clearGrasp/models_sampled"

    csv_file = open(path2csv, "r")
    csv_reader = csv.reader(csv_file, delimiter=",")
    
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0])

    for e, row in enumerate(csv_reader):
        if e == 0:
            continue

        # Prediction
        scene_id, im_id, obj_id, score, R, t, time = row
        scene_id = int(scene_id)
        im_id = int(im_id)
        obj_id = int(obj_id)
        R_pred = np.array(R.split(" ")).astype(np.float32).reshape(3, 3)
        t_pred = np.array(t.split(" ")).astype(np.float32).reshape(3, 1)

        # DATASET_DATA
        rgb_path = os.path.join(path2dataset, str(scene_id).zfill(6), "rgb", str(im_id).zfill(6)+".png")
        depth_path = os.path.join(path2dataset, str(scene_id).zfill(6), "depth", str(im_id).zfill(6)+".png")
        scene_camera_path = os.path.join(path2dataset, str(scene_id).zfill(6), "scene_camera.json")
        scene_gt_path = os.path.join(path2dataset, str(scene_id).zfill(6), "scene_gt.json")
        model_path = os.path.join(path2models, f"obj_{obj_id:06d}.ply")

        # Load the scene data  + ground truth and process them 
        with open(scene_camera_path, "r") as f:
            scene_camera = json.load(f)
        cam_K = np.array(scene_camera[str(im_id)]["cam_K"]).reshape(3, 3)
        depth_scale = scene_camera[str(im_id)]["depth_scale"]        

        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)
        s_gt = scene_gt[str(im_id)]
        T_gts = [] 
        for obj in s_gt:
            if obj["obj_id"] == obj_id:
                R_gt = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                t_gt = np.array(obj["cam_t_m2c"]).reshape(3, 1)
                T_gt = np.eye(4)
                T_gt[:3, :3] = R_gt
                T_gt[:3, 3] = t_gt.squeeze() / 1000
                T_gts.append(T_gt)

        if len(T_gts) == 0:
            print(f"Object {obj_id} not found in scene {scene_id} at image {im_id}")
            continue

        T_pred = np.eye(4)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = t_pred.squeeze() / 1000


        # Load the image and make the point cloud
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, -1)
        depth = np.float32(depth * depth_scale / 1000)

        scene_pcd = make_point_cloud(rgb, depth, cam_K)
        # Load the model
        model_gt = o3d.io.read_point_cloud(model_path)
        model_gt.scale(1/1000, center=(0, 0, 0)) # Scale to meters
        model_pred = copy.deepcopy(model_gt)
        model_pred.transform(T_pred)
        model_pred.paint_uniform_color([1, 0, 1])
        model_gt.paint_uniform_color([0, 1, 0])



        # Transform the scene and the models so the camera init view is aproximate to camera
        T = np.diag([1, -1, -1, 1]) # DO NOT ASK ME WHY IT HAS TO BE HERE
        list2show = [axis.transform(T), scene_pcd.transform(T), model_pred.transform(T)]
        for T_gt in T_gts:
            model_gt_tmp = copy.deepcopy(model_gt)
            model_gt_tmp.transform(T_gt)
            model_gt_tmp.transform(T)
            list2show.append(model_gt_tmp)


        # Visualize the scene and the model
        o3d.visualization.draw_geometries(list2show)

def test_clearPose_orig():
    path2dataset = "/home/testbed/Projects/bop_toolkit/clearpose_downsample_100"
    path2models = os.path.join(path2dataset, "models")

    set_id = 1
    scene_id = 1
    im_id = 0
    path2scene_data = os.path.join(path2dataset, f"set{set_id}",f"scene{scene_id}")
    scene_metadata_path = os.path.join(path2scene_data, "metadata.mat")

    # Load the metadata
    print("Loading metadata")
    metadata = scipy.io.loadmat(scene_metadata_path)
    print("Metadata loaded")
    # print(metadata.keys())
    key = str(im_id).zfill(6)

    annot = metadata[key]
    print(annot)
    print(type(annot))
    print(len(annot))
    print(annot.shape)
    print(annot[0].shape)
    print(annot[0])
   

def show_CNCpicking():
    path2dataset = "/home/testbed/Projects/bop_toolkit/CNC-picking/real_d415"
    path2models = "/home/testbed/Projects/bop_toolkit/CNC-picking/models"

    for scene in sorted(os.listdir(path2dataset)):
        scene_gt_path = os.path.join(path2dataset, scene, "scene_gt.json")
        scene_gt_info_path = os.path.join(path2dataset, scene, "scene_gt_info.json")
        scene_camera_path = os.path.join(path2dataset, scene, "scene_camera.json")
        scene_depth_path = os.path.join(path2dataset, scene, "depth")
        scene_rgb_path = os.path.join(path2dataset, scene, "rgb")

        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        
        for img_id, annot_list in scene_gt.items():
            img_name = img_id.zfill(6)
            rgb = cv2.imread(os.path.join(scene_rgb_path, img_name+".png"))
            depth = cv2.imread(os.path.join(scene_depth_path, img_name+".png"), -1)
            depth = np.float32(depth / 1000)

            with open(scene_camera_path, "r") as f:
                scene_camera = json.load(f)
            cam_K = np.array(scene_camera[img_id]["cam_K"]).reshape(3, 3)
            # depth_scale = scene_camera[img_id]["depth_scale"]
            # depth_scale IS NOT PROVIDED IN THE SCENE_CAMERA.JSON 
            pcd = make_point_cloud(rgb, depth, cam_K)
            print(f"Processing {scene}/{img_id}")
            for annot in annot_list:
                obj_id = annot["obj_id"]
                Rmx = np.array(annot["cam_R_m2c"]).reshape(3, 3)
                tv = np.array(annot["cam_t_m2c"]).reshape(3, 1)
                model_path = os.path.join(path2models, f"obj_{obj_id:06d}.ply")
                model = o3d.io.read_triangle_mesh(model_path)
                model.scale(1/1000, center=(0, 0, 0))
                
                Tmx = np.eye(4)
                Tmx[:3, :3] = Rmx
                Tmx[:3, 3] = tv.squeeze() / 1000
                model.transform(Tmx)

                visualize_geometries(pcd, [model])
            break

def eval_CNCpicking():
    path2dataset = "/home/testbed/Projects/bop_toolkit/CNCpicking/real_d415"
    path2models = "/home/testbed/Projects/bop_toolkit/CNCpicking/models_eval"

    path2csv = "/home/testbed/Projects/bop_toolkit/CNCpicking/results/foundationPose_CNCpicking-test.csv"

    csv_file = open(path2csv, "r")
    csv_reader = csv.reader(csv_file, delimiter=",")

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0])

    for e, row in enumerate(csv_reader):
        if e == 0:
            continue

        scene_id, im_id, obj_id, score, R, t, time = row
        scene_id = int(scene_id)
        im_id = int(im_id)
        obj_id = int(obj_id)

        if obj_id in [5,6]:
            continue
        R_pred = np.array(R.split(" ")).astype(np.float32).reshape(3, 3)
        t_pred = np.array(t.split(" ")).astype(np.float32).reshape(3, 1)

        rgb_path = os.path.join(path2dataset, str(scene_id).zfill(6), "rgb", str(im_id).zfill(6)+".png")
        depth_path = os.path.join(path2dataset, str(scene_id).zfill(6), "depth", str(im_id).zfill(6)+".png")
        scene_camera_path = os.path.join(path2dataset, str(scene_id).zfill(6), "scene_camera.json")
        scene_gt_path = os.path.join(path2dataset, str(scene_id).zfill(6), "scene_gt.json")
        model_path = os.path.join(path2models, f"obj_{obj_id:06d}.ply")

        with open(scene_camera_path, "r") as f:
            scene_camera = json.load(f)
        cam_K = np.array(scene_camera[str(im_id)]["cam_K"]).reshape(3, 3)
        depth_scale = 1

        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        s_gt = scene_gt[str(im_id)]
        T_gts = []
        for obj in s_gt:
            if obj["obj_id"] == obj_id:
                R_gt = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                t_gt = np.array(obj["cam_t_m2c"]).reshape(3, 1)
                T_gt = np.eye(4)
                T_gt[:3, :3] = R_gt
                T_gt[:3, 3] = t_gt.squeeze() / 1000
                T_gts.append(T_gt)
        
        if len(T_gts) == 0:
            print(f"Object {obj_id} not found in scene {scene_id} at image {im_id}")
            continue

        T_pred = np.eye(4)
        T_pred[:3, :3] = R_pred
        T_pred[:3, 3] = t_pred.squeeze() / 1000

        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, -1)
        depth = np.float32(depth * depth_scale / 1000)

        scene_pcd = make_point_cloud(rgb, depth, cam_K)
        model_gt = o3d.io.read_point_cloud(model_path)
        model_gt.scale(1/1000, center=(0, 0, 0))
        model_pred = copy.deepcopy(model_gt)
        model_pred.transform(T_pred)
        model_pred.paint_uniform_color([1, 0, 1])
        model_gt.paint_uniform_color([0, 1, 0])

        T = np.diag([1, -1, -1, 1]) # DO NOT ASK ME WHY IT HAS TO BE HERE
        list2show = [axis.transform(T), scene_pcd.transform(T), model_pred.transform(T)]
        for T_gt in T_gts:
            model_gt_tmp = copy.deepcopy(model_gt)
            model_gt_tmp.transform(T_gt)
            model_gt_tmp.transform(T)
            list2show.append(model_gt_tmp)

        o3d.visualization.draw_geometries(list2show)








if __name__ == "__main__":
    
    # test_clearGrasp_csv()
    # import keyboard
    # show_CNCpicking()
    eval_CNCpicking()

    # print("Press any key to continue...")
    # key = keyboard.wait()
    # print(f"You pressed {key}") 



    
