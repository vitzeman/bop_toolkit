"""
Script that generates the targets for the BOP evaluation scripts from the BOP datasaet

Usage:
TODO

TODO: Add argument parsing for running the script from the terminal
"""

import os
import json 


if __name__ == "__main__":
    folder = "/home/testbed/Projects/bop_toolkit/CNCpicking/real_d415"
    print(f"Creating targets from {folder}")
    targets_json = []
    for scene in sorted(os.listdir(folder)):
        # if int(scene) not in [2]: # skip all scenes except 2
        #     continue

        scene_gt_path = os.path.join(folder, scene, "scene_gt.json")
        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)

        for img_id, img_infos in scene_gt.items():
            objects_list = {}
            for annot in img_infos:
                obj_id = annot["obj_id"]
                # if obj_id in [1, 2, 3, 4]:
                #     continue


                if obj_id in objects_list.keys(): # 
                    objects_list[obj_id]["inst_count"] += 1
                else:
                    obj_id_dict = {
                        "im_id": int(img_id),
                        "inst_count": 1,
                        "obj_id": int(obj_id),
                        "scene_id": int(scene),
                    }
                    objects_list[obj_id] = obj_id_dict

            for obj in objects_list.values():
                targets_json.append(obj)


    targets_path = os.path.join("/home/testbed/Projects/bop_toolkit/CNCpicking", "test_targets56_bop19.json")
    with open(targets_path, "w") as f:
        json.dump(targets_json, f, indent=2)
