import os 
import sys
import json
# import argparse


import cv2
import numpy as np
import open3d as o3d



def make_point_cloud(rgb_img, depth_img, cam_K):
    # convert images to open3d types
    rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    depth_img_o3d = o3d.geometry.Image(depth_img)

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

def pick_points(pcd):
    print("Pick a point using [shift + left click]")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def get_transformation(scene_pcd, mesh_id, path2meshes="/home/testbed/Projects/bop_toolkit/clearGrasp/models_sampled"):
    print("Adding transformation to mesh", mesh_id)
    mesh_path = os.path.join(path2meshes, "obj_{:06d}.ply".format(mesh_id))
    #read point cloud
    pcd = o3d.io.read_point_cloud(mesh_path)
    pcd.points = o3d.utility.Vector3dVector(
            np.array(pcd.points) / 1000
        )  # convert mm to meter
    # colorize
    pcd.paint_uniform_color([0, 0, 1])

    num_points_check = False
    #pick points
    while not num_points_check:
        scene_points_ids = pick_points(scene_pcd)
        mesh_points_ids = pick_points(pcd)
        num_points_check = len(mesh_points_ids) == len(scene_points_ids) and len(mesh_points_ids) >= 3
        if not num_points_check:
            print("The number of points picked in the mesh and the scene should be the same and at least 3 - RETRY!")



    #get points
    corrseps =  np.zeros((len(mesh_points_ids), 2))
    corrseps[:, 0] = mesh_points_ids
    corrseps[:, 1] = scene_points_ids

    #estimate transformation
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    
    transf_init = p2p.compute_transformation(
        pcd, scene_pcd, o3d.utility.Vector2iVector(corrseps)
    )

    #apply transformation
    pcd.transform(transf_init)
    o3d.visualization.draw_geometries([pcd, scene_pcd])

    return transf_init


def main():
    # Cand select the scene and starting image: DO NOT OVERSHOOT 
    scene_id = 3
    img_id = 0

    # TODO: change this path to the path of the dataset
    path2dataset = "/home/testbed/Projects/bop_toolkit/clearGrasp"

    scene_path = os.path.join(path2dataset, "eval", str(scene_id).zfill(6))
    print(scene_path)
    scene_exist = os.path.exists(scene_path)
    should_exit = False
        
    while True:
        scene_path = os.path.join(path2dataset, "eval", str(scene_id).zfill(6))
        if not os.path.exists(scene_path):
            print(f"Scene {scene_id} does not exist")
            if scene_id < 1:
                scene_id = 1
                print("Scene id cannot be negative")
            else:
               scene_id -= 1
            continue

        scene_gt_path = os.path.join(scene_path, "scene_gt.json")
        if not os.path.exists(scene_gt_path):
            with open(scene_gt_path, "w") as f:
                json.dump({}, f, indent=4)
        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        img_path = os.path.join(path2dataset, "eval", str(scene_id).zfill(6), "rgb", str(img_id).zfill(6) + ".png")
        if not os.path.exists(img_path):
            print(f"Image {img_id} does not exist")
            if img_id < 0:
                img_id = 0
                print("Image id cannot be negative")
            else:
                img_id -= 1
            continue

        img = cv2.imread(img_path)

        # Load the camera parameters
        cam_param_path = os.path.join(path2dataset, "eval", str(scene_id).zfill(6), "scene_camera.json")

        with open(cam_param_path, "r") as f:
            cam_param = json.load(f)
            cam_K = cam_param[str(img_id)]["cam_K"]
            cam_K = np.array(cam_K).reshape((3, 3))
            depth_scale = cam_param[str(img_id)]["depth_scale"]
        
        # Load the depth image
        depth_path = os.path.join(path2dataset, "eval", str(scene_id).zfill(6), "depth", str(img_id).zfill(6) + ".png")
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_img = np.float32(depth_img * depth_scale / 1000)

        scene_pcd = make_point_cloud(img, depth_img, cam_K)

        object_list = scene_gt.get(str(img_id), [])

        #rgb image show
        while True:
            img2show = img.copy()
            num_imgs_annotated = len(object_list)
           
            cv2.putText(img2show, "Scene id: " + str(scene_id), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 0), thickness=2)
            cv2.putText(img2show, "Scene id: " + str(scene_id), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), thickness=1)
            cv2.putText(img2show, "Image id: " + str(img_id), (10, 40), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 0), thickness=2)
            cv2.putText(img2show, "Image id: " + str(img_id), (10, 40), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), thickness=1)
            cv2.putText(img2show, "Anotated: " + str(num_imgs_annotated), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=2)
            cv2.putText(img2show, "Anotated: " + str(num_imgs_annotated), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), thickness=1)


            cv2.imshow("rgb", img2show)
            key = cv2.waitKey(1)
            # if key != -1:
            #     print(key)

            if key == 27: #esc
                should_exit = True
            
                break

            # nums 1-9
            elif key >= 49 and key <= 57:
                mesh_id = key - 48
                transformation = get_transformation(scene_pcd, mesh_id)
                R = transformation[:3, :3].tolist()
                t = (transformation[:3, 3]*1000).tolist()

                t_dict = {
                    "cam_R_m2c": R,
                    "cam_t_m2c": t,
                    "obj_id": mesh_id,
                }
                object_list.append(t_dict)
                scene_gt[str(img_id)] = object_list
                with open(scene_gt_path, "w") as f:
                    json.dump(scene_gt, f, indent=4)
                    

            elif key in [ord("m"), 83]:
                img_id += 1
                break
            
            elif key in  [ord("n"), 81]:
                img_id -= 1
                if img_id < 0:
                    img_id = 0
                    print("Image id cannot be negative")
                break

            elif key in [ ord("l"), 82]:
                scene_id += 1
                img_id = 0
                break
            elif key in [ord("k"), 84]:
                scene_id -= 1
                img_id = 0
                if scene_id < 0:
                    scene_id = 0
                    print("Scene id cannot be negative")
                break

            elif key == ord("g"):
                print("Models keys:")
                print("\t 1: - rectangular cup")
                print("\t 2: - leaf")
                print("\t 3: - vase") 
                print("\t 4: - glass bottle")
                print("\t 5: - heart")
                print("\t 6: - rectangular bottle")
                print("\t 7: - star")
                print("\t 8: - glass")
                print("\t 9: - tree")

            elif key == ord("h"):
                print("Press ESC to exit")
                print("Press 1-9 to annotate the object")
                print("Press m or right arrow to go to the next image")
                print("Press n or left arrow to go to the previous image")
                print("Press l or up arrow to go to the next scene")
                print("Press k or down arrow to go to the previous scene")
                print("Press h to show this help")

            elif key != -1:
                print("Key", key, "not recognized")
            


        if should_exit:
            break
    

if __name__ == "__main__":
    main()