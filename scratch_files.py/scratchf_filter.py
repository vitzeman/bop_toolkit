"""Filters out the dataset to not include the objects 5 and 6."""

import os
import json



if __name__ == "__main__":
    path2dataset = "/home/testbed/Projects/bop_toolkit/CNC-picking/real_d415"
    empty_scenes = []

    for directpry in sorted(os.listdir(path2dataset)):
        path2scene = os.path.join(path2dataset, directpry)
        scene_gt_path = os.path.join(path2scene, "scene_gt.json")
        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        added = False

        scene_gt_new = {}
        for img_id, annotation_list in scene_gt.items():
            new_annotation_list = []
            for annotation in annotation_list:
                if annotation["obj_id"] in [5, 6]:
                    continue
                added = True
                new_annotation_list.append(annotation)
            

            scene_gt_new[img_id] = new_annotation_list

        if not added:
            empty_scenes.append(img_id)
            print(f"Empty scene: {directpry}")



        all_path = scene_gt_path.replace(".json", "_all.json")
        with open(all_path, "w") as f:
            json.dump(scene_gt, f, indent=2)

        with open(scene_gt_path, "w") as f:
            json.dump(scene_gt_new, f, indent=2)

    with open("empty_scenes.json", "w") as f:
        json.dump(empty_scenes, f, indent=2)
        print(f"Empty scenes: {len(empty_scenes)}\n{empty_scenes}")
    print("Done")