"""
Script to visualize the bounding boxes of the object in scene

Usage:
TODO
"""

import os
import json

import cv2

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]

def load_image(path2image):
    for ext in IMG_EXTENSIONS:
        if os.path.exists(path2image + ext):
            path2image += ext
            break
    
    if not os.path.exists(path2image):
        raise FileNotFoundError(f"Image not found: {path2image}")

    return cv2.imread(path2image)

    

def visualize_dataset_scene(path2scene):
    print(f"Processing: {path2scene}")
    scene_gt_info = os.path.join(path2scene, "scene_gt_info.json")
    with open(scene_gt_info, "r") as f:
        scene_gt_info = json.load(f)

    for img_id, annotation_list in scene_gt_info.items():
        img_name = img_id.zfill(6)
        img = load_image(os.path.join(path2scene, "rgb", f"{img_name}"))

        for annotation in annotation_list:
            bbox_obj = annotation["bbox_obj"]
            bbox_vis = annotation["bbox_visib"]
            cv2.rectangle(img, (bbox_obj[0], bbox_obj[1]), (bbox_obj[0]+bbox_obj[2], bbox_obj[1]+bbox_obj[3]), (0, 255, 0), 2)
            cv2.rectangle(img, (bbox_vis[0], bbox_vis[1]), (bbox_vis[0]+bbox_vis[2], bbox_vis[1]+bbox_vis[3]), (0, 255, 255), 1)

            print(annotation)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        

def visualize_dataset(path2dataset):
    for directpry in sorted(os.listdir(path2dataset)):
        path2scene = os.path.join(path2dataset, directpry)
        visualize_dataset_scene(path2scene)
        break

if __name__ == "__main__":
    path2dataset = "/home/testbed/Projects/bop_toolkit/CNC-picking/real_d415"
    visualize_dataset(path2dataset)