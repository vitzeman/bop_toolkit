# Author: Vit Zeman
# CTU, CIIRC, Testbed for Industry 4.0

"""Visualization of the prediction results from the csv file

Shows the ground truth and the predicted poses of the objects in the scene
"""

# Native imports
import os
import json
import warnings
import argparse
import logging
import sys
import glob
from dataclasses import dataclass
from typing import List, Tuple, Union
from pathlib import Path

# Third party imports
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
import pandas as pd


class Dataset:
    """ Handles the dataset paths and the csv file and provides the access to the data
    """    
    def __init__(self, scenes_path: Union[str, Path], models_path: Union[str, Path], csv_path: Union[str, Path]) -> None:
        assert os.path.isdir(scenes_path), f"Path {scenes_path} is not a valid directory"
        assert os.path.isdir(models_path), f"Path {models_path} is not a valid directory"
        assert os.path.isfile(csv_path), f"Path {csv_path} is not a valid file"

        self.scenes_path = scenes_path
        self.models_path = models_path
        self.csv_path = csv_path

class Settings:
    UNLIT = "defaultUnlit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [0.9, 0.9, 0.9, 1.0]
        self.scene_material.shader = Settings.UNLIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        self.annotation_obj_material.shader = Settings.UNLIT

class AppWindow:
    def __init__(self, scene:Dataset, resolution: Tuple[int, int] = (1920,1080)) -> None:
        self.scene = scene

        # >>> Path preparation >>>
        self.scenes_path = Path(self.scene.scenes_path)
        self.scenes_names = sorted(os.listdir(self.scene.scenes_path))
        self.max_scene_num = len(self.scenes_paths)
        self.scene_idx = 0

        if len(self.scenes_names) == 0:
            raise ValueError(f"No scenes found in {self.scenes_path}")
        # <<< Path preparation <<<

        self.settings = Settings()

        self.main_window = gui.Application.instance.create_window(
            "Prediction Visualizer", *resolution
        )
        mw = self.main_window

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(mw.renderer)

        em = mw.theme.font_size

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # >>> View control >>>
        view_ctrls = gui.CollapsableVert("View control", 0, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._highlight_obj = gui.Checkbox("Highligh annotation objects")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        view_ctrls.add_child(self._highlight_obj)

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # <<< View control <<<

        # This I do not understand
        mw.set_on_layout(self._on_layout)
        mw.add_child(self._scene)
        mw.add_child(self._settings_panel)

        # >>> Scene control >>>
        self._scene_control = gui.CollapsableVert(
            "Scene Control", 0.33 * em, gui.Margins(em, 0, 0, 0)
        )
        self._scene_control.set_is_open(True)

        self._images_buttons_label = gui.Label("Images:")
        self._samples_buttons_label = gui.Label("Scene: ")

        self._pre_image_button = gui.Button("Previous")
        self._pre_image_button.horizontal_padding_em = 0.8
        self._pre_image_button.vertical_padding_em  = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)

        self._next_image_button = gui.Button("Next")
        self._next_image_button.horizontal_padding_em = 0.8
        self._next_image_button.vertical_padding_em = 0
        self._next_image_button.set_on_clicked(self._on_next_image)

        self._pre_sample_button = gui.Button("Previous")
        self._pre_sample_button.horizontal_padding_em = 0.8
        self._pre_sample_button.vertical_padding_em = 0
        self._pre_sample_button.set_on_clicked(self._on_previous_scene)

        self._next_sample_button = gui.Button("Next")
        self._next_sample_button.horizontal_padding_em = 0.8
        self._next_sample_button.vertical_padding_em = 0
        self._next_sample_button.set_on_clicked(self._on_next_scene)

        h = gui.Horiz(0.4 * em)  # row 1
        h.add_stretch()
        h.add_child(self._images_buttons_label)
        h.add_child(self._pre_image_button)
        h.add_child(self._next_image_button)
        h.add_stretch()
        self._scene_control.add_child(h)
        h = gui.Horiz(0.4 * em)  # row 2
        h.add_stretch()
        h.add_child(self._samples_buttons_label)
        h.add_child(self._pre_sample_button)
        h.add_child(self._next_sample_button)
        h.add_stretch()
        self._scene_control.add_child(h)

        self._view_numbers = gui.Horiz(0.4 * em)
        self._image_number = gui.Label("Image: " + f"{0:06}")
        self._scene_number = gui.Label("Scene: " + f"{0:06}")
        self._view_numbers.add_child(self._image_number)
        self._view_numbers.add_child(self._scene_number)
        self._scene_control.add_child(self._view_numbers)

        self._settings_panel.add_child(self._scene_control)
        # <<< Scene control <<<

        # >>> Menu >>>
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        mw.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        mw.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # <<< Menu <<<

        self._on_point_size(1)  # set default size to 1

        self._apply_settings()

        self._annotation_scene = None

        # set callbacks for key control
        self._scene.set_on_key(self._transform)

        self._left_shift_modifier = False





    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red,
            self.settings.bg_color.green,
            self.settings.bg_color.blue,
            self.settings.bg_color.alpha,
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.modify_geometry_material(
                "annotation_scene", self.settings.scene_material
            )
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._point_size.double_value = self.settings.scene_material.point_size


    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Prediction visualization")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Prediction visualization"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    # THESE NEED TO BE REWRITEN BASED on the indexes not just 1,2,3 etc
    def _on_next_scene(self):
        if self._check_changes():
            return

        if self._annotation_scene.scene_num + 1 > len(
            next(os.walk(self.scenes.scenes_path))[1]
        ):  # 1 for how many folder (dataset scenes) inside the path
            self._on_error("There is no next scene.")
            return
        self.scene_load(
            self.scenes.scenes_path, self._annotation_scene.scene_num + 1, 0
        )  # open next scene on the first image

    def _on_previous_scene(self):
        if self._check_changes():
            return

        if self._annotation_scene.scene_num - 1 < 1:
            self._on_error("There is no scene number before scene 1.")
            return
        self.scene_load(
            self.scenes.scenes_path, self._annotation_scene.scene_num - 1, 0
        )  # open next scene on the first image

    def _on_next_image(self):
        if self._check_changes():
            return

        num = len(
            next(
                os.walk(
                    os.path.join(
                        self.scenes.scenes_path,
                        f"{self._annotation_scene.scene_num:06}",
                        "depth",
                    )
                )
            )[2]
        )
        if self._annotation_scene.image_num + 1 >= len(
            next(
                os.walk(
                    os.path.join(
                        self.scenes.scenes_path,
                        f"{self._annotation_scene.scene_num:06}",
                        "depth",
                    )
                )
            )[2]
        ):  # 2 for files which here are the how many depth images
            self._on_error("There is no next image.")
            return
        self.scene_load(
            self.scenes.scenes_path,
            self._annotation_scene.scene_num,
            self._annotation_scene.image_num + 1,
        )

    def _on_previous_image(self):
        if self._check_changes():
            return

        if self._annotation_scene.image_num - 1 < 0:
            self._on_error("There is no image number before image 0.")
            return
        self.scene_load(
            self.scenes.scenes_path,
            self._annotation_scene.scene_num,
            self._annotation_scene.image_num - 1,
        )


    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        elif not light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(
                mesh.obj_name, self.settings.annotation_obj_material
            )

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

        
        pass

    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
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
    
    def scene_load(self, scene_idx:int, image_idx:int) -> None:
        self._annotation_changed = False

        self._scene.scene.clear_geometries()


        # >>> Limiting the scene index >>>
        if scene_idx < 0:
            scene_idx = 0
        elif scene_idx >= self.max_scene_num:
            scene_idx = self.max_scene_num - 1

        # <<< Limiting the scene index <<<
        scene_path = self.scenes_path / self.scenes_names[scene_idx]

        camera_params_path = scene_path / "scene_camera.json"
        with open(camera_params_path) as f:
            camera_params = json.load(f)
            cam_K = np.array(camera_params[str(image_idx)]["cam_K"]).reshape(3,3)
            depth_scale = camera_params[str(image_idx)]["depth_scale"]

        rgbs_path = scene_path / "rgb"
        rgbs_names = sorted(os.listdir(rgbs_path))

        if len(rgbs_names) == 0:
            raise ValueError(f"No images found in {rgbs_path}")
        
        if image_idx < 0:
            image_idx = 0
        elif image_idx >= len(rgbs_names):
            image_idx = len(rgbs_names) - 1

        rgb_path = rgbs_path / rgbs_names[image_idx]
        rgb = cv2.imread(str(rgb_path))
        depth_path = scene_path / "depth" / rgbs_names[image_idx]
        depth_exists = depth_path.exists()
        if depth_exists:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) * depth_scale / 1000
        else:
            depth = None
            pcd = None
        if depth is not None:
            pcd = self._make_point_cloud(rgb, depth, cam_K)

            self._scene.scene.add_geometry(
                "annotation_scene",
                pcd,
                self.settings.scene_material,
                add_downsampled_copy_for_fast_rendering=True,
            )

            bounds = pcd.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())
            center = np.array([0, 0, 0])
            eye = center + np.array([0, 0, -0.5])
            up = np.array([0, -1, 0])
            self._scene.look_at(center, eye, up)

            self._annotation_scene = AnnotationScene(geometry, scene_num, image_num)





        

        

def main():
    # TODO: add parsing of the arguments
    scenes = Dataset(
        scenes_path="/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/tests",
        models_path="/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/models",
        csv_path="/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/results/foundationposepartial_clearpose_downsample_100_bop-test.csv"
    )

    gui.Application.instance.initialize()
    app = AppWindow(scenes) # Initializes the window

    app.scene_load(scenes.scenes_path, 0)
    app.update_obj_list()

    gui.Application.instance.run()

if __name__ == "__main__":
    main()