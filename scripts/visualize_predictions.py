"""
Script for the visualization of the prediction of the model 
"""

import json 
import os 
import csv
import copy

import cv2 
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import scipy.io
import keyboard as kb

import pandas as pd

PRED_COLOR = [1,0.7,0] # Orange in RGB format 0-1
GT_COLOR = [0,1,0] # Green in RGB format 0-1


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
    assert rgb_img.shape[:2] == depth_img.shape[:2], f"RGB and depth image have different shapes {rgb_img.shape} != {depth_img.shape}"
    assert cam_K.shape == (3, 3), f"Camera intrinsics have wrong shape {cam_K.shape} != (3, 3)"
    
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

def visualize_geometries(pointcloud, gt_model, pred_model=[], gt_color=GT_COLOR, pred_color=PRED_COLOR):
    """ Run visualization of the 2 point clouds and the models in the scene

    Args:
        pointcloud (_type_): Pointcloud of the scene
        gt_model (_type_): Ground truth pose of the models
        pred_model (list, optional): Predictions of the models. Defaults to [].
        gt_color (list, optional): Color of the gt models. Defaults to [0, 1, 0].
        pred_color (list, optional): Color of the prediction models. Defaults to [1, 0, 1].
    """    
    T = np.diag([1, -1, -1, 1]) # DO NOT ASK ME WHY IT HAS TO BE HERE it is camera viewpoint basically 
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0, 0, 0])
    axis.transform(T)
    list2show = [axis]

    if pointcloud is not None:
        pointcloud.transform(T)  
        list2show.append(pointcloud)
    
    for gt in gt_model:
        gt.transform(T)
        gt.paint_uniform_color(gt_color)
        list2show.append(gt)

    for pred in pred_model:
        pred.transform(T)
        pred.paint_uniform_color(pred_color)
        list2show.append(pred)

    o3d.visualization.draw_geometries(list2show)


def visualize_prediction(path2dataset:str, path2models:str, csv_file_path:str=None):
    """Visualizes the dataset in interactive 3D viewer, if given the predictions
        are also visualized. Additionally the 2D image is shown with the input
        bounding boxes and the contours/overlay of the predictions.

    Args:
        path2dataset (str): Path to the dataset
        path2models (str): Path to the dataset models
        csv_file_path (str, optional): Path to the results of the predictions. Defaults to None.
    """    
    # TODO: add the argument parser
    assert os.path.isdir(path2dataset), f"Directory {path2dataset} does not exist"
    assert os.path.isdir(path2models), f"Directory {path2models} does not exist"

    df = None
    if csv_file_path is not None:
        assert os.path.isfile(csv_file_path), f"File {csv_file_path} does not exist"
        df = pd.read_csv(csv_file_path)

    PRED_COLOR = [1,0.7,0]

    cv2.namedWindow("2d Projections", cv2.WINDOW_NORMAL)
    black = np.zeros((720, 1080, 3), dtype=np.uint8)
    cv2.putText(black, "Place this window preferably on other screen", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(black, "Then press any key to continue", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("2d Projections", black)
    cv2.waitKey(0)

    for scene in sorted(os.listdir(path2dataset)):
        scene_id = int(scene)

        # Just skipping for debug
        if scene_id != 256:
            continue

        scene_gt_path = os.path.join(path2dataset, scene, "scene_gt.json")
        scene_gt_info_path = os.path.join(path2dataset, scene, "scene_gt_info.json")
        scene_camera_path = os.path.join(path2dataset, scene, "scene_camera.json")
        scene_depth_path = os.path.join(path2dataset, scene, "depth")
        scene_rgb_path = os.path.join(path2dataset, scene, "rgb")

        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)
 
        with open(scene_gt_info_path, "r") as f:
            scenes_gt_info = json.load(f)
        
        for img_id, annot_list in scene_gt.items():
            img_name = img_id.zfill(6)
            rgb = cv2.imread(os.path.join(scene_rgb_path, img_name+".png"))
            img2show = copy.deepcopy(rgb)
            cv2.putText(img2show, "RGB", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img2show, "RGB", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img2show, "Input bbox", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img2show, "Input bbox", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            scene_gt_info = scenes_gt_info[img_id]
            for annot in scene_gt_info:
                bbox_obj = annot["bbox_obj"]
                bbox_visib = annot["bbox_visib"]
                obj_x, obj_y, obj_w, obj_h = bbox_obj
                visib_x, visib_y, visib_w, visib_h = bbox_visib
                cv2.rectangle(img2show, (obj_x, obj_y), (obj_x+obj_w, obj_y+obj_h), (0, 255, 0), 2)
                cv2.rectangle(img2show, (visib_x, visib_y), (visib_x+visib_w, visib_y+visib_h), (0, 175, 0), 1)
            
            with open(scene_camera_path, "r") as f:
                scene_camera = json.load(f)
            cam_K = np.array(scene_camera[img_id]["cam_K"]).reshape(3, 3)

            # >>> Point cloud creation >>>            
            depth_path = os.path.join(scene_depth_path, img_name+".png")
            pcd = None
            if os.path.isfile(depth_path):
                depth = cv2.imread(os.path.join(scene_depth_path, img_name+".png"), -1)
                depth_scale = scene_camera[img_id].get("depth_scale", 1)
                depth = np.float32(depth *depth_scale / 1000)
             
                pcd = make_point_cloud(rgb, depth, cam_K, depth_scale=depth_scale)
            # <<< Point cloud creation <<<
        
            # >>> 2D image prejection initialization >>>
            img_w = rgb.shape[1]
            img_h = rgb.shape[0]
            renderer = rendering.OffscreenRenderer(img_w, img_h)
            pinhole = o3d.camera.PinholeCameraIntrinsic(img_w, img_h, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2])
            renderer.scene.set_background([0., 0., 0., 0.])
            renderer.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
            renderer.setup_camera(pinhole, np.eye(4))
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
            mtl.shader = "defaultUnlit"
            # <<< 2D image prejection initialization <<<

            print(f"Processing {scene}/{img_id}")
            
            # >>> Ground truth objects >>>
            gt_models = []
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

                gt_models.append(model)
            # <<< Ground truth objects <<<
        
            # >>> Predicted objects >>>
            pred_models = []
            if df is not None:
                predictions = df.loc[(df["scene_id"] == scene_id) & (df["im_id"] == int(img_id))]
                contours_img = copy.deepcopy(rgb)
                cv2.putText(contours_img, "Contours", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(contours_img, "Contours", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                for e, row in predictions.iterrows():
                    obj_id = row["obj_id"]
                    Rmx = np.array(row["R"].split(" ")).astype(np.float32).reshape(3, 3)
                    t = np.array(row["t"].split(" ")).astype(np.float32).reshape(3, 1)
                    model_path = os.path.join(path2models, f"obj_{obj_id:06d}.ply")
                    model = o3d.io.read_triangle_mesh(model_path)
                    model.scale(1/1000, center=(0, 0, 0))
                    Tmx = np.eye(4)
                    Tmx[:3, :3] = Rmx
                    Tmx[:3, 3] = t.squeeze() / 1000
                    model.transform(Tmx)
                    
                    # >>> Contours generation >>>
                    model.paint_uniform_color([1, 1, 1])
                    renderer.scene.add_geometry(str(0),model, mtl)
                    img_o3d = renderer.render_to_image()
                    img = np.array(img_o3d)[:,:,::-1]
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(img_gray, 50, 255, 0)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contours_img, contours, -1, tuple([x*255 for x in PRED_COLOR[::-1]]), 2)
                    renderer.scene.remove_geometry(str(0))
                    # <<< Contours generation <<<

                    model.paint_uniform_color(PRED_COLOR)
                    pred_models.append(model)

                # >>> 2D image projection >>>
                for e, model in enumerate(pred_models):
                    renderer.scene.add_geometry(str(e), model, mtl)

                img_o3d = renderer.render_to_image()

                img = np.array(img_o3d)[:,:,::-1]
                # print(img.shape, img.dtype, img.max(), img.min())
                overlay = cv2.addWeighted(rgb, 0.7, img, 0.3, 0)
                print(img2show.shape, overlay.shape, contours_img.shape)
                cv2.putText(overlay, "Overlay", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 6, cv2.LINE_AA)
                cv2.putText(overlay, "Overlay", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # <<< 2D image projection <<<

                img2show = np.vstack([img2show, overlay, contours_img])
            # <<< Predicted objects <<<

            # >>> Visualization >>>
            cv2.imshow("2d Projections", img2show)
            cv2.waitKey(1)
            visualize_geometries(pcd, gt_models, pred_models)
            # <<< Visualization <<<


            

if __name__ == "__main__":

    path2dataset = "/home/testbed/Projects/bop_toolkit/CNCpicking/real_d415"
    path2models = "/home/testbed/Projects/bop_toolkit/CNCpicking/models"
    csv_file_path = "/home/testbed/Projects/bop_toolkit/CNCpicking/results/foundationPose_CNCpicking-test.csv"    
    visualize_prediction(path2dataset=path2dataset, path2models=path2models, csv_file_path=csv_file_path)
   
