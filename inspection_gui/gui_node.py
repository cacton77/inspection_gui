#!/usr/bin/env python3
import rclpy

import os
import yaml
import platform
import cv2  # OpenCV library
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
from multiprocessing import Process, Queue
from inspection_gui.ros_thread import RosThread
from inspection_gui.reconstruct import ReconstructThread
from inspection_gui.partitioner import Partitioner, NPCD

plt.style.use('dark_background')

isMacOS = (platform.system() == "Darwin")


class MyGui():
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_SAVE_AS = 3
    MENU_IMPORT_MODEL = 4
    MENU_IMPORT_PCD = 5
    MENU_QUIT = 6
    MENU_NEW = 7
    MENU_UNDO = 8
    MENU_REDO = 9
    MENU_PREFERENCES = 10
    MENU_SHOW_AXES = 11
    MENU_SHOW_GRID = 12
    MENU_SHOW_MODEL = 13
    MENU_SHOW_POINT_CLOUDS = 14
    MENU_SHOW_REGIONS = 15
    MENU_SHOW_VIEWPOINT = 16
    MENU_SHOW_SETTINGS = 17
    MENU_SHOW_ERRORS = 18
    MENU_ABOUT = 21
    GEOM_NAME = "Geometry"
    SCENE_TAB = 0
    MONITOR_TAB = 1

    def __init__(self, update_delay=-1):

        self.config_file = os.path.expanduser(
            '~/Inspection/Parts/config/default.yaml')
        self.config_dict = yaml.load(
            open(self.config_file), Loader=yaml.FullLoader)

        self.inspection_root_path = self.config_dict['inspection_root']

        self.update_delay = update_delay
        self.is_done = False
        self.lock = threading.Lock()

        self.app = gui.Application.instance

        # icons_font = gui.FontDescription(
        #     '/tmp/MaterialIcons-Regular.ttf', point_size=12)
        # icons_font.add_typeface_for_code_points(
        #     '/tmp/MaterialIcons-Regular.ttf', [0xE037, 0xE034])
        # icons_font_id = gui.Application.instance.add_font(icons_font)

        self.window = self.app.create_window(
            "Inspection GUI", width=1920, height=1080, x=0, y=30)

        em = self.window.theme.font_size
        r = self.window.content_rect
        self.menu_height = 2.5 * em
        self.header_height = 3 * em
        self.footer_height = 10 * em

        w = self.window
        self.window.set_on_close(self.on_main_window_closing)
        if self.update_delay < 0:
            self.window.set_on_tick_event(self.on_main_window_tick_event)

        ###############################

        self.depth_trunc = 1.0

        # Common settings for all panels
        self.panel_color = gui.Color(44/255, 54/255, 57/255, 0.8)
        self.background_color = [22/255, 29/255, 36/255, 1.0]

        ###############################

        self.plot_cmap = 'cubehelix'
        self.webcam_fig = plt.figure()

        # Threads
        self.ros_thread = RosThread(stream_id=0)  # 0 id for main camera
        self.reconstruct_thread = ReconstructThread(rate=20)
        self.ros_thread.start()  # processing frames in input stream
        self.reconstruct_thread.start()  # processing frames in input stream

        # 3D SCENE ################################################################
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.scene_widget.scene.set_background(self.background_color)
        self.scene_widget.enable_scene_caching(False)

        # MAIN TABS ################################################################

        self.main_tabs = gui.TabControl()
        self.main_tabs.background_color = self.panel_color

        # 3D SCENE TAB ####################

        # Add XY Axes

        self.xy_axes = o3d.geometry.LineSet()
        self.xy_axes.points = o3d.utility.Vector3dVector(
            np.array([[-1000, 0, 0], [1000, 0, 0], [0, -1000, 0], [0, 1000, 0]]))
        self.xy_axes.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1], [2, 3]]))
        self.xy_axes.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0]]))

        # Part Model
        self.part_model_name = "Part Model"
        self.part_model = None
        self.part_model_units = self.config_dict['part']['model_units']

        self.part_model_material = o3d.visualization.rendering.MaterialRecord()
        self.part_model_material.shader = "defaultLit"
        self.part_model_material.base_color = [
            0.8, 0.8, 0.8, 1.0]  # RGBA, Red color

        part_model_file = self.config_dict['part']['model']

        # Part Point Cloud
        self.part_point_cloud_name = "Part Point Cloud"
        self.part_point_cloud = None
        self.part_point_cloud_units = self.config_dict['part']['point_cloud_units']

        self.part_point_cloud_material = o3d.visualization.rendering.MaterialRecord()
        self.part_point_cloud_material.shader = 'defaultLit'
        self.part_point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.part_point_cloud_material.point_size = 8.0

        part_pcd_file = self.config_dict['part']['point_cloud']

        # Viewpoints etc.
        self.viewpoint_material = o3d.visualization.rendering.MaterialRecord()
        self.viewpoint_material.shader = 'defaultUnlit'
        self.viewpoint_material.base_color = [1.0, 1.0, 1.0, 1.0]

        self.line_material = o3d.visualization.rendering.MaterialRecord()
        self.line_material.shader = 'unlitLine'
        self.line_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.line_material.line_width = 2.0

        # Live Point Cloud
        self.live_point_cloud_name = "Point Cloud"

        self.live_point_cloud_material = o3d.visualization.rendering.MaterialRecord()
        self.live_point_cloud_material.shader = 'defaultUnlit'
        self.live_point_cloud_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.live_point_cloud_material.point_size = 5.0

        # Add geometry
        if part_model_file is not None:
            self._import_model(part_model_file)
        if part_pcd_file is not None:
            self._import_point_cloud(part_pcd_file)

        # Scene ribbon

        def _on_part_model_file_edit(path):
            self._import_model(path)

        def _on_part_pcd_file_edit(path):
            self._import_point_cloud(path)

        def _on_part_model_units_select(value, i):
            self.part_model_units = value

        def _on_part_pcd_units_select(value, i):
            self.part_point_cloud_units = value

        # Tab buttons
        self.scene_ribbon = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.scene_ribbon.add_child(gui.Label("Part: "))
        load_part_button = gui.Button("Load YAML")
        load_part_button.set_on_clicked(self._on_load_part_config)
        self.part_model_file_edit = gui.TextEdit()
        self.part_model_file_edit.placeholder_text = "/path/to/model.stl"
        self.part_model_file_edit.text_value = part_model_file
        self.part_model_file_edit.set_on_value_changed(
            _on_part_model_file_edit)
        self.part_model_units_select = gui.Combobox()
        self.part_model_units_select.add_item("mm")
        self.part_model_units_select.add_item("cm")
        self.part_model_units_select.add_item("m")
        self.part_model_units_select.add_item("in")
        self.part_model_units_select.set_on_selection_changed(
            _on_part_model_units_select)
        self.part_pcd_file_edit = gui.TextEdit()
        self.part_pcd_file_edit.placeholder_text = "/path/to/pcd.ply"
        self.part_pcd_file_edit.text_value = part_pcd_file
        self.part_pcd_file_edit.set_on_value_changed(_on_part_pcd_file_edit)
        self.part_pcd_units_select = gui.Combobox()
        self.part_pcd_units_select.add_item("mm")
        self.part_pcd_units_select.add_item("cm")
        self.part_pcd_units_select.add_item("m")
        self.part_pcd_units_select.add_item("in")
        self.part_pcd_units_select.set_on_selection_changed(
            _on_part_pcd_units_select)

        grid = gui.VGrid(3, 0.25 * em)
        grid.add_child(gui.Label("Model: "))
        grid.add_child(self.part_model_file_edit)
        grid.add_child(self.part_model_units_select)
        grid.add_child(gui.Label("Point Cloud: "))
        grid.add_child(self.part_pcd_file_edit)
        grid.add_child(self.part_pcd_units_select)

        self.scene_ribbon.add_child(load_part_button)
        self.scene_ribbon.add_fixed(0.5 * em)
        self.scene_ribbon.add_child(grid)
        self.scene_ribbon.add_fixed(5 * em)

        # Scene ribbon defect panels

        def _on_defect_select(value, i):
            defect_camera = self.defects[i]['camera']
            self.fov_height_mm = defect_camera['fov']['height_mm']
            self.fov_width_mm = defect_camera['fov']['width_mm']
            self.roi_height = defect_camera['roi']['height_px']
            self.roi_width = defect_camera['roi']['width_px']
            self.focal_distance_mm = defect_camera['focal_distance_mm']

            self.fov_height_mm_edit.double_value = self.fov_height_mm
            self.fov_width_mm_edit.double_value = self.fov_width_mm
            self.roi_height_edit.int_value = self.roi_height
            self.roi_width_edit.int_value = self.roi_width
            self.focal_distance_edit.double_value = self.focal_distance_mm

        self.defects = self.config_dict['defects']

        self.scene_ribbon.add_child(gui.Label("Defect Selection:"))
        self.defect_selection = gui.Combobox()
        for i in range(len(self.defects)):
            self.defect_selection.add_item(self.defects[i]['name'])
        self.defect_selection.set_on_selection_changed(_on_defect_select)
        self.scene_ribbon.add_child(self.defect_selection)
        width = self.window.content_rect.width
        self.scene_ribbon.add_fixed(width)

        # Scanner buttons
        # Attempt to add Material Icons to button
        # 0xE037, 0xE034
        play_button = gui.Button("Play")
        # play_button.add_child(play_label)

        stop_button = gui.Button('Stop')
        part_frame_button = gui.Button('v')
        part_frame_button.toggleable = True

        # STEREO CAMERA PANEL #########################################################

        self.stereo_camera_panel = gui.CollapsableVert("Stereo Camera", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.stereo_camera_panel.background_color = self.panel_color
        self.stereo_camera_panel.set_is_open(False)

        grid = gui.VGrid(1, 0.25 * em)

        # Add image widget to webcam panel
        rgb_grid = gui.VGrid(1, 0.25 * em)
        depth_grid = gui.VGrid(1, 0.25 * em)
        illuminance_grid = gui.VGrid(1, 0.25 * em)

        self.illuminance_image = gui.ImageWidget()

        # RGB TAB #################################

        self.rgb_image = gui.ImageWidget()
        self.rgb_image.set_on_mouse(self._monitor_mouse_event)
        rgb_grid.add_child(self.rgb_image)

        # DEPTH TAB ###############################

        self.depth_image = gui.ImageWidget()

        def on_depth_trunc_changed(value):
            self.depth_trunc = value
            self.reconstruct_thread.depth_trunc = self.depth_trunc
            self.ros_thread.depth_trunc = self.depth_trunc

        depth_trunc_edit = gui.Slider(gui.Slider.DOUBLE)
        depth_trunc_edit.set_limits(0.01, 1)
        depth_trunc_edit.double_value = self.depth_trunc
        depth_trunc_edit.background_color = gui.Color(0, 0, 0, 0.8)
        depth_trunc_edit.set_on_value_changed(on_depth_trunc_changed)

        depth_grid.add_child(self.depth_image)
        depth_grid.add_child(depth_trunc_edit)

        # ILLUMINANCE TAB #########################

        self.illuminance_image = gui.ImageWidget()
        illuminance_grid.add_child(self.illuminance_image)

        tabs = gui.TabControl()
        # tabs.add_tab("RGB", rgb_grid)
        tabs.add_tab("RGB", self.rgb_image)
        tabs.add_tab("Depth", depth_grid)
        tabs.add_tab("Illuminance", illuminance_grid)
        tabs.add_child(gui.TabControl())
        grid.add_child(tabs)

        self.stereo_camera_panel.add_child(grid)

        # PART FRAME PANEL #############################################################

        self.part_frame_panel = gui.CollapsableVert("Part Frame", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.part_frame_panel.background_color = self.panel_color
        self.part_frame_panel.set_is_open(False)

        # origin_frame_dropdown = gui.Combobox()

        self.part_frame_parent = self.config_dict['part']['frame']['parent']
        self.part_frame = self.config_dict['part']['frame']['child']
        x = self.config_dict['part']['frame']['x']
        y = self.config_dict['part']['frame']['y']
        z = self.config_dict['part']['frame']['z']
        roll = self.config_dict['part']['frame']['roll']
        pitch = self.config_dict['part']['frame']['pitch']
        yaw = self.config_dict['part']['frame']['yaw']

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.part_frame_parent, self.part_frame)

        def _on_part_x_edit(x):
            self.part_x_edit.double_value = x
            self._get_and_send_part_frame_tf()

        def _on_part_y_edit(y):
            self.part_y_edit.double_value = y
            self._get_and_send_part_frame_tf()

        def _on_part_z_edit(z):
            self.part_z_edit.double_value = z
            self._get_and_send_part_frame_tf()

        def _on_part_roll_edit(roll):
            self.part_roll_edit.double_value = roll
            self._get_and_send_part_frame_tf()

        def _on_part_pitch_edit(pitch):
            self.part_pitch_edit.double_value = pitch
            self._get_and_send_part_frame_tf()

        def _on_part_yaw_edit(yaw):
            self.part_yaw_edit.double_value = yaw
            self._get_and_send_part_frame_tf()

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("X: "))
        self.part_x_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_x_edit.double_value = x
        self.part_x_edit.set_on_value_changed(_on_part_x_edit)
        grid.add_child(self.part_x_edit)
        grid.add_child(gui.Label("Y: "))
        self.part_y_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_y_edit.double_value = y
        self.part_y_edit.set_on_value_changed(_on_part_y_edit)
        grid.add_child(self.part_y_edit)
        grid.add_child(gui.Label("Z: "))
        self.part_z_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_z_edit.double_value = z
        self.part_z_edit.set_on_value_changed(_on_part_z_edit)
        grid.add_child(self.part_z_edit)
        grid.add_child(gui.Label("Roll: "))
        self.part_roll_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_roll_edit.double_value = roll
        self.part_roll_edit.set_on_value_changed(_on_part_roll_edit)
        grid.add_child(self.part_roll_edit)
        grid.add_child(gui.Label("Pitch: "))
        self.part_pitch_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_pitch_edit.double_value = pitch
        self.part_pitch_edit.set_on_value_changed(_on_part_pitch_edit)
        grid.add_child(self.part_pitch_edit)
        grid.add_child(gui.Label("Yaw: "))
        self.part_yaw_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.part_yaw_edit.double_value = yaw
        self.part_yaw_edit.set_on_value_changed(_on_part_yaw_edit)
        grid.add_child(self.part_yaw_edit)

        self.part_frame_panel.add_child(grid)

        # CAMERA FRAME PANEL ############################################################

        self.camera_frame_panel = gui.CollapsableVert("Camera Frame", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.camera_frame_panel.background_color = self.panel_color
        self.camera_frame_panel.set_is_open(False)

        # origin_frame_dropdown = gui.Combobox()

        self.camera_frame_parent = self.config_dict['camera']['frame']['parent']
        self.camera_frame = self.config_dict['camera']['frame']['child']
        x = self.config_dict['camera']['frame']['x']
        y = self.config_dict['camera']['frame']['y']
        z = self.config_dict['camera']['frame']['z']
        roll = self.config_dict['camera']['frame']['roll']
        pitch = self.config_dict['camera']['frame']['pitch']
        yaw = self.config_dict['camera']['frame']['yaw']

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.camera_frame_parent, self.camera_frame)

        def _on_camera_x_edit(x):
            self.camera_x_edit.double_value = x
            self._get_and_send_camera_frame_tf()

        def _on_camera_y_edit(y):
            self.camera_y_edit.double_value = y
            self._get_and_send_camera_frame_tf()

        def _on_camera_z_edit(z):
            self.camera_z_edit.double_value = z
            self._get_and_send_camera_frame_tf()

        def _on_camera_roll_edit(roll):
            self.camera_roll_edit.double_value = roll
            self._get_and_send_camera_frame_tf()

        def _on_camera_pitch_edit(pitch):
            self.camera_pitch_edit.double_value = pitch
            self._get_and_send_camera_frame_tf()

        def _on_camera_yaw_edit(yaw):
            self.camera_yaw_edit.double_value = yaw
            self._get_and_send_camera_frame_tf()

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("X: "))
        self.camera_x_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_x_edit.double_value = x
        self.camera_x_edit.set_on_value_changed(_on_camera_x_edit)
        grid.add_child(self.camera_x_edit)
        grid.add_child(gui.Label("Y: "))
        self.camera_y_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_y_edit.double_value = y
        self.camera_y_edit.set_on_value_changed(_on_camera_y_edit)
        grid.add_child(self.camera_y_edit)
        grid.add_child(gui.Label("Z: "))
        self.camera_z_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_z_edit.double_value = z
        self.camera_z_edit.set_on_value_changed(_on_camera_z_edit)
        grid.add_child(self.camera_z_edit)
        grid.add_child(gui.Label("Roll: "))
        self.camera_roll_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_roll_edit.double_value = roll
        self.camera_roll_edit.set_on_value_changed(_on_camera_roll_edit)
        grid.add_child(self.camera_roll_edit)
        grid.add_child(gui.Label("Pitch: "))
        self.camera_pitch_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_pitch_edit.double_value = pitch
        self.camera_pitch_edit.set_on_value_changed(_on_camera_pitch_edit)
        grid.add_child(self.camera_pitch_edit)
        grid.add_child(gui.Label("Yaw: "))
        self.camera_yaw_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.camera_yaw_edit.double_value = yaw
        self.camera_yaw_edit.set_on_value_changed(_on_camera_yaw_edit)
        grid.add_child(self.camera_yaw_edit)

        self.camera_frame_panel.add_child(grid)

        # VIEWPOINT GENERATION PANEL ###############################################

        # INSPECTION ACTION PANEL ##################################################

        self.action_panel = gui.Vert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.action_panel.background_color = self.panel_color

        # VIEWPOINT GENERATION ######################

        self.fov_width_px = self.config_dict['camera']['fov']['width_px']
        self.fov_height_px = self.config_dict['camera']['fov']['height_px']
        self.fov_width_mm = self.config_dict['camera']['fov']['width_mm']
        self.fov_height_mm = self.config_dict['camera']['fov']['height_mm']
        self.roi_width = self.config_dict['camera']['roi']['width_px']
        self.roi_height = self.config_dict['camera']['roi']['height_px']
        self.dof = self.config_dict['camera']['dof_mm']
        self.focal_distance_mm = self.config_dict['camera']['focal_distance_mm']

        # PARTITIONER SETTINGS
        self.viewpoint_dict = None
        self.partitioner = Partitioner()

        self.partitioner.fov_height = self.fov_height_mm * \
            (self.roi_height/self.fov_height_px) / 10

        self.partitioner.fov_width = self.fov_width_mm * \
            (self.roi_width/self.fov_width_px) / 10

        self.partitioner.focal_distance = self.focal_distance_mm / 10

        # self.partitioning_progress_queue = Queue()
        # self.partitioning_results_queue = Queue()

        # npcd = NPCD.from_o3d_point_cloud(o3d.geometry.PointCloud())

        # self.partitioning_process = Process(
        #     target=self.partitioner.rg_not_smart_partition_worker, args=(npcd,
        #                                                                  self.partitioning_progress_queue,
        #                                                                  self.partitioning_results_queue))

        def _on_fov_width_mm_edit(value):
            self.fov_width_mm = value
            self.partitioner.fov_width = self.fov_width_mm * \
                (self.roi_width/self.fov_width_px)

        def _on_fov_height_mm_edit(value):
            self.fov_height_mm = value
            self.partitioner.fov_height = self.fov_height_mm * \
                (self.roi_height/self.fov_height_px)

        def _on_fov_width_px_edit(value):
            self.fov_width_px = value
            self.partitioner.fov_width = self.fov_width_mm * \
                (self.roi_width/self.fov_width_px)

        def _on_fov_height_px_edit(value):
            self.fov_height_px = value
            self.partitioner.fov_height = self.fov_height_mm * \
                (self.roi_height/self.fov_height_px)

        def _on_roi_width_px_edit(value):
            self.roi_width = value
            self.partitioner.fov_width = self.fov_width_mm * \
                (self.roi_width/self.fov_width_px)

        def _on_roi_height_edit(value):
            self.roi_height = value
            self.partitioner.fov_height = self.fov_height_mm * \
                (self.roi_height/self.fov_height_px)

        def _on_dof_edit(value):
            self.config_dict['camera']['dof'] = value
            self.dof = value
            self.partitioner.dof = value

        def _on_focal_distance_edit(value):
            self.config_dict['camera']['focal_distance_mm'] = value
            self.focal_distance_mm = value
            self.partitioner.focal_distance = value / 10

        def _on_viewpoint_generation_button_clicked():
            # Disable UI buttons
            self.defect_selection.enabled = False
            self.roi_width_edit.enabled = False
            self.roi_height_edit.enabled = False
            self.fov_height_mm_edit.enabled = False
            self.fov_width_mm_edit.enabled = False
            self.part_model_file_edit.enabled = False
            self.part_pcd_file_edit.enabled = False

            # Clear existing viewpoints
            self.viewpoint_dict = None
            self.viewpoint_stack.selected_index = 0
            self.part_point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            self.partitioner.run(self.part_point_cloud)
            self.generate_viewpoints_button.enabled = False

        fov_px_grid = gui.VGrid(2, 0.25 * em)
        fov_px_grid.add_child(gui.Label("width (px): "))
        self.fov_width_px_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        self.fov_width_px_edit.set_on_value_changed(_on_fov_width_px_edit)
        self.fov_width_px_edit.int_value = self.fov_width_px
        self.fov_width_px_edit.enabled = False
        fov_px_grid.add_child(self.fov_width_px_edit)
        fov_px_grid.add_child(gui.Label("height (px): "))
        self.fov_height_px_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        self.fov_height_px_edit.set_on_value_changed(_on_fov_height_px_edit)
        self.fov_height_px_edit.int_value = self.fov_height_px
        self.fov_height_px_edit.enabled = False
        fov_px_grid.add_child(self.fov_height_px_edit)

        fov_mm_grid = gui.VGrid(2, 0.25 * em)
        fov_mm_grid.add_child(gui.Label("width (mm): "))
        self.fov_width_mm_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.fov_width_mm_edit.set_on_value_changed(_on_fov_width_mm_edit)
        self.fov_width_mm_edit.double_value = self.fov_width_mm
        fov_mm_grid.add_child(self.fov_width_mm_edit)
        fov_mm_grid.add_child(gui.Label("height (mm): "))
        self.fov_height_mm_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.fov_height_mm_edit.set_on_value_changed(_on_fov_height_mm_edit)
        self.fov_height_mm_edit.double_value = self.fov_height_mm
        fov_mm_grid.add_child(self.fov_height_mm_edit)

        roi_grid = gui.VGrid(2, 0.25 * em)
        roi_grid.add_child(gui.Label("width (px): "))
        self.roi_width_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        self.roi_width_edit.set_on_value_changed(_on_roi_width_px_edit)
        self.roi_width_edit.int_value = self.roi_width
        roi_grid.add_child(self.roi_width_edit)
        roi_grid.add_child(gui.Label("height (px): "))
        self.roi_height_edit = gui.NumberEdit(gui.NumberEdit.Type.INT)
        self.roi_height_edit.set_on_value_changed(_on_roi_height_edit)
        self.roi_height_edit.int_value = self.roi_height
        roi_grid.add_child(self.roi_height_edit)

        self.dof_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.dof_edit.double_value = self.dof
        self.dof_edit.set_on_value_changed(_on_dof_edit)
        self.focal_distance_edit = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
        self.focal_distance_edit.double_value = self.focal_distance_mm
        self.focal_distance_edit.set_on_value_changed(_on_focal_distance_edit)

        dof_focal_distance_grid = gui.VGrid(2, 0.25 * em)
        dof_focal_distance_grid.add_child(gui.Label("Depth of focus (mm): "))
        dof_focal_distance_grid.add_child(self.dof_edit)
        dof_focal_distance_grid.add_child(gui.Label("Focal distance (mm): "))
        dof_focal_distance_grid.add_child(self.focal_distance_edit)

        self.generate_viewpoints_button = gui.Button("Generate Viewpoints")
        self.generate_viewpoints_button.set_on_clicked(
            _on_viewpoint_generation_button_clicked)

        self.viewpoint_stack = gui.StackedWidget()
        self.viewpoint_progress_bar = gui.ProgressBar()
        self.viewpoint_progress_bar.background_color = gui.Color(0, 1, 0, 0.8)

        def _on_viewpoint_slider_changed(value):

            old_region_name = f"region_{self.selected_viewpoint}"
            self.scene_widget.scene.remove_geometry(old_region_name)
            self.scene_widget.scene.remove_geometry(f"{old_region_name}_line")
            self.scene_widget.scene.remove_geometry(
                f"{old_region_name}_viewpoint")

            old_region = self.viewpoint_dict[old_region_name]

            viewpoint_tf = old_region['viewpoint']
            point_cloud = old_region['point_cloud']
            origin = old_region['origin']
            point = old_region['point']
            color = old_region['color']

            point_cloud.paint_uniform_color(color)
            viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
                radius=1)
            viewpoint_geom.paint_uniform_color(color)
            viewpoint_geom.transform(viewpoint_tf)

            viewpoint_line = o3d.geometry.LineSet()
            viewpoint_line.points = o3d.utility.Vector3dVector(
                np.array([origin, point]))
            viewpoint_line.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1]]))
            viewpoint_line.paint_uniform_color(color)

            self.scene_widget.scene.add_geometry(
                f"{old_region_name}_viewpoint", viewpoint_geom, self.viewpoint_material)
            self.scene_widget.scene.add_geometry(
                old_region_name, point_cloud, self.part_point_cloud_material)
            self.scene_widget.scene.add_geometry(
                f"{old_region_name}_line", viewpoint_line, self.line_material)

            # get new selected_viewpoint
            self.selected_viewpoint = self.partitioner.best_path[int(value-1)]

            new_region_name = f"region_{self.selected_viewpoint}"
            self.scene_widget.scene.remove_geometry(new_region_name)
            self.scene_widget.scene.remove_geometry(f"{new_region_name}_line")
            self.scene_widget.scene.remove_geometry(
                f"{new_region_name}_viewpoint")

            new_region = self.viewpoint_dict[new_region_name]

            viewpoint_tf = new_region['viewpoint']
            point_cloud = new_region['point_cloud']
            origin = new_region['origin']
            point = new_region['point']
            color = [0.0, 1.0, 0.0]

            point_cloud.paint_uniform_color(color)
            viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
                radius=1)
            viewpoint_geom.paint_uniform_color(color)
            viewpoint_geom.transform(viewpoint_tf)

            viewpoint_line = o3d.geometry.LineSet()
            viewpoint_line.points = o3d.utility.Vector3dVector(
                np.array([origin, point]))
            viewpoint_line.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1]]))
            viewpoint_line.paint_uniform_color(color)

            self.scene_widget.scene.add_geometry(
                f"{new_region_name}_viewpoint", viewpoint_geom, self.viewpoint_material)
            self.scene_widget.scene.add_geometry(
                new_region_name, point_cloud, self.part_point_cloud_material)
            self.scene_widget.scene.add_geometry(
                f"{new_region_name}_line", viewpoint_line, self.line_material)

        self.selected_viewpoint = 0
        self.viewpoint_slider = gui.Slider(gui.Slider.INT)
        self.viewpoint_slider.set_limits(0, 100)
        self.viewpoint_slider.enabled = False
        self.viewpoint_slider.set_on_value_changed(
            _on_viewpoint_slider_changed)
        self.viewpoint_stack.add_child(self.viewpoint_progress_bar)
        self.viewpoint_stack.add_child(self.viewpoint_slider)
        self.viewpoint_stack.selected_index = 0
        action_grid = gui.Vert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz.add_child(gui.Label("Viewpoint: "))
        horiz.add_child(self.viewpoint_stack)
        action_grid.add_child(horiz)
        horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz.add_child(gui.Button("Focus"))
        horiz.add_child(gui.Button("Move"))

        # Image capture
        self.image_count = 0

        def _on_capture_image():
            file_path = self.inspection_root_path + \
                '/Images/image_' + str(self.image_count) + '.jpg'
            self.ros_thread.capture_image(file_path)

        self.capture_image_button = gui.Button("Capture")
        self.capture_image_button.set_on_clicked(_on_capture_image)
        horiz.add_child(self.capture_image_button)
        # action_grid.add_child(horiz)

        go_button = gui.Button(" Go ")
        go_button.background_color = gui.Color(0.0, 0.8, 0.0, 1.0)
        # viewpoint_select = gui.NumberEdit(gui.NumberEdit.Type.INT)
        # viewpoint_select.int_value = 0
        # viewpoint_select.set_limits(0, 100)

        horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz.add_child(gui.Label("FOV: "))
        horiz.add_fixed(0.5 * em)
        horiz.add_child(fov_mm_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(fov_px_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(gui.Label("ROI: "))
        horiz.add_fixed(0.5 * em)
        horiz.add_child(roi_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(dof_focal_distance_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(self.generate_viewpoints_button)
        # horiz.add_fixed(self.window.content_rect.width/2)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(action_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(gui.Button("Move"))
        horiz.add_fixed(0.5 * em)
        horiz.add_child(gui.Button("Focus"))
        horiz.add_fixed(0.5 * em)
        horiz.add_child(self.capture_image_button)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(go_button)

        self.action_panel.add_child(horiz)
        # LOG PANEL ################################################################

        self.log_panel = gui.CollapsableVert("Log", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.log_panel.background_color = gui.Color(
            36/255, 36/255, 36/255, 1.0)
        ros_log_vert = gui.ScrollableVert(
            0.25 * em)
        self.ros_log_text = gui.ListView()
        self.ros_log_text.background_color = gui.Color(0, 0, 0, 0.8)
        self.ros_log_text.enabled = False
        self.log_list = ["Log 1"]
        self.ros_log_text.set_items(self.log_list)
        self.ros_log_text.selected_index = 0
        self.log_panel.add_child(self.ros_log_text)
        self.log_panel.set_is_open(False)

        # MONITOR TAB ################################################################

        # Pan and Orbit Settings
        self.pan_pos = None
        self.pan_goal = None
        self.orbit_pos = None
        self.orbit_goal = None

        self.monitor_ribbon = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        for i in range(10):
            button = gui.Button(f"Button {i}")
            self.monitor_ribbon.add_child(button)

        self.monitor_image_panel = gui.Vert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        class MonitorImageWidget(gui.ImageWidget):
            def __init__(self):
                super().__init__()

            def on_mouse_event(self, event):
                return o3d.visualization.gui.Widget.EventCallbackResult.CONSUMED

        self.monitor_image_widget = MonitorImageWidget()
        self.monitor_image_widget.update_image(o3d.geometry.Image(
            np.zeros((480, 640, 3), dtype=np.uint8)))
        self.monitor_image_widget.set_on_mouse(self._monitor_mouse_event)

        self.monitor_image_panel.add_child(self.monitor_image_widget)
        self.monitor_image_panel.background_color = gui.Color(0, 0, 0, 1)
        self.monitor_image_panel.enabled = False
        self.monitor_image_panel.visible = False

        self.camera_config_panel = gui.CollapsableVert("Camera Configuration", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.camera_config_panel.background_color = self.panel_color
        self.camera_config_panel.enabled = False
        self.camera_config_panel.visible = False

        svert = gui.ScrollableVert(0.25 * em)
        svert.background_color = self.panel_color

        camera_params = self.ros_thread.read_camera_params()

        def on_camera_param_changed(value):
            self.ros_thread.set_camera_params()

        # Order the camera parameters

        param_names_ordered = camera_params.keys()

        grid = gui.VGrid(2, 0.25 * em)

        # Loop through camera_params and add widgets to grid
        for name in param_names_ordered:
            # Truncate name to 20 characters
            if len(name) > 20:
                display_name = name[:20] + '...'
            else:
                display_name = name
            description = camera_params[name]
            # grid = gui.Horiz(0, gui.Margins(
            # 0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
            grid.add_child(gui.Label(f'{display_name}: '))

            if description['type'] == 2:
                edit = gui.NumberEdit(gui.NumberEdit.INT)
                edit.int_value = int(description['value'])

            elif description['type'] == 3:
                edit = gui.TextEdit()
                # edit.double_value = float(description['value'])

            elif description['type'] == 1:
                edit = gui.Checkbox(name)
                # edit.checked = description['value']

            elif description['type'] == 4:
                if len(description['choices']) > 1:
                    edit = gui.Combobox()
                    for choice in description['choices']:
                        edit.add_item(choice)
                else:
                    edit = gui.TextEdit()
                    edit.text_value = description['value']
            else:
                grid.add_child(gui.Label("Unknown type"))
                grid.add_child(gui.Label("Unknown type"))

            if description['read_only']:
                edit.enabled = False

            grid.add_child(edit)
            # self.camera_config_panel.add_child(grid)

        svert.add_child(grid)

        self.camera_config_panel.add_child(svert)
        # self.monitor_widget.add_child(self.camera_config_panel)
        # self.monitor_widget.add_child(self.monitor_image_widget)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item(
                    "About", MyGui.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item(
                    "Quit", MyGui.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item(
                "New", MyGui.MENU_NEW)
            file_menu.add_item(
                "Open...", MyGui.MENU_OPEN)
            file_menu.add_separator()
            file_menu.add_item(
                "Save", MyGui.MENU_SAVE)
            file_menu.add_item(
                "Save As...", MyGui.MENU_SAVE_AS)
            file_menu.add_separator()
            file_menu.add_item("Import Model", MyGui.MENU_IMPORT_MODEL)
            file_menu.add_item("Import Point Cloud", MyGui.MENU_IMPORT_PCD)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item(
                    "Quit", MyGui.MENU_QUIT)
            edit_menu = gui.Menu()
            edit_menu.add_item("Undo", MyGui.MENU_UNDO)
            edit_menu.add_item("Redo", MyGui.MENU_REDO)
            edit_menu.add_separator()
            edit_menu.add_item("Preferences...", MyGui.MENU_PREFERENCES)

            view_menu = gui.Menu()
            # Scene display options
            view_menu.add_item("Show Axes", MyGui.MENU_SHOW_AXES)
            view_menu.set_checked(MyGui.MENU_SHOW_AXES, True)
            view_menu.add_item("Show Grid", MyGui.MENU_SHOW_GRID)
            view_menu.set_checked(MyGui.MENU_SHOW_GRID, True)
            ground_plane_menu = gui.Menu()
            ground_plane_menu.add_item("XY", 100)
            ground_plane_menu.add_item("XZ", 101)
            ground_plane_menu.add_item("YZ", 102)
            view_menu.add_menu("Ground Plane", ground_plane_menu)
            view_menu.add_separator()
            # Object display options
            view_menu.add_item("Show Model", MyGui.MENU_SHOW_MODEL)
            view_menu.set_checked(MyGui.MENU_SHOW_MODEL, True)
            view_menu.add_item("Show Point Clouds",
                               MyGui.MENU_SHOW_POINT_CLOUDS)
            view_menu.set_checked(MyGui.MENU_SHOW_POINT_CLOUDS, True)
            view_menu.add_item("Show Regions", MyGui.MENU_SHOW_REGIONS)
            view_menu.set_checked(MyGui.MENU_SHOW_REGIONS, True)
            view_menu.add_separator()
            # Panel display options
            view_menu.add_item("Viewpoint Generation",
                               MyGui.MENU_SHOW_VIEWPOINT)
            view_menu.set_checked(
                MyGui.MENU_SHOW_VIEWPOINT, True)
            view_menu.add_item("Lighting & Materials",
                               MyGui.MENU_SHOW_SETTINGS)
            view_menu.set_checked(
                MyGui.MENU_SHOW_SETTINGS, True)
            view_menu.add_item("Error Logging",
                               MyGui.MENU_SHOW_ERRORS)
            view_menu.set_checked(
                MyGui.MENU_SHOW_ERRORS, False)
            help_menu = gui.Menu()
            help_menu.add_item(
                "About", MyGui.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("View", view_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        # menu item is activated.
        w.set_on_menu_item_activated(
            MyGui.MENU_NEW, self._on_menu_new)
        w.set_on_menu_item_activated(
            MyGui.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(
            MyGui.MENU_SAVE, self._on_menu_save)
        w.set_on_menu_item_activated(
            MyGui.MENU_SAVE_AS, self._on_menu_save_as)
        w.set_on_menu_item_activated(MyGui.MENU_IMPORT_MODEL,
                                     self._on_menu_import_model)
        w.set_on_menu_item_activated(MyGui.MENU_IMPORT_PCD,
                                     self._on_menu_import_pcd)
        w.set_on_menu_item_activated(
            MyGui.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(
            MyGui.MENU_PREFERENCES, self._on_menu_preferences)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_AXES, self._on_menu_show_axes)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_GRID, self._on_menu_show_grid)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_MODEL, self._on_menu_show_model)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_POINT_CLOUDS, self._on_menu_show_point_clouds)
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_REGIONS, self._on_menu_show_regions)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_VIEWPOINT,
                                     self._on_menu_toggle_viewpoint_generation_panel)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(
            MyGui.MENU_ABOUT, self._on_menu_about)
        # ----

        self._reset_scene()

        self.scene_widget.setup_camera(
            60, self.scene_widget.scene.bounding_box, [1, 0, 0])
        self.scene_widget.set_view_controls(
            gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene_widget.scene.show_axes(False)
        self.scene_widget.scene.show_ground_plane(
            True, o3d.visualization.rendering.Scene.GroundPlane.XY)

        self.main_tabs.add_tab("3D Scene", self.scene_ribbon)
        self.main_tabs.add_tab("Monitor", self.monitor_ribbon)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.monitor_image_panel)
        self.window.add_child(self.main_tabs)
        self.window.add_child(self.stereo_camera_panel)
        self.window.add_child(self.part_frame_panel)
        self.window.add_child(self.camera_frame_panel)
        self.window.add_child(self.action_panel)
        self.window.add_child(self.log_panel)
        self.window.add_child(self.camera_config_panel)
        self.window.set_on_layout(self._on_layout)

    def _reset_scene(self):
        self.scene_widget.scene.clear_geometry()
        self.scene_widget.scene.add_geometry(
            'xy axes', self.xy_axes, self.line_material)
        if self.part_model is not None:
            self.scene_widget.scene.add_geometry(
                self.part_model_name, self.part_model, self.part_model_material)
        if self.part_point_cloud is not None:
            self.scene_widget.scene.add_geometry(
                self.part_point_cloud_name, self.part_point_cloud, self.part_point_cloud_material)
        self.scene_widget.setup_camera(
            60, self.scene_widget.scene.bounding_box, [0, 0, 0])

    def _send_transform(self, T, parent, child):
        self.ros_thread.send_transform(T, parent, child)

    def _on_load_part_config(self):
        self.config_file = os.path.expanduser(
            "~") + '/Inspection/Parts/config/blade.yaml'
        self.config_dict = yaml.load(
            open(self.config_file), Loader=yaml.FullLoader)

        # Model
        self.part_model_units = self.config_dict['part']['model_units']
        part_model_file = self.config_dict['part']['model']
        self.part_model_file_edit.text_value = part_model_file
        self._import_model(part_model_file)

        # Part Point Cloud
        self.part_point_cloud_units = self.config_dict['part']['point_cloud_units']
        part_pcd_file = self.config_dict['part']['point_cloud']
        self.part_pcd_file_edit.text_value = part_pcd_file
        self._import_point_cloud(part_pcd_file)

        self.defects = self.config_dict['defects']

        self.defect_selection.clear_items()
        for i in range(len(self.defects)):
            self.defect_selection.add_item(self.defects[i]['name'])

        # Send updated part frame
        self.part_frame_parent = self.config_dict['part']['frame']['parent']
        self.part_frame = self.config_dict['part']['frame']['child']
        x = self.config_dict['part']['frame']['x']
        y = self.config_dict['part']['frame']['y']
        z = self.config_dict['part']['frame']['z']
        roll = self.config_dict['part']['frame']['roll']
        pitch = self.config_dict['part']['frame']['pitch']
        yaw = self.config_dict['part']['frame']['yaw']

        self.part_x_edit.double_value = x
        self.part_y_edit.double_value = y
        self.part_z_edit.double_value = z
        self.part_roll_edit.double_value = roll
        self.part_pitch_edit.double_value = pitch
        self.part_yaw_edit.double_value = yaw

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.part_frame_parent, self.part_frame)

        # send updated camera frame
        self.camera_frame_parent = self.config_dict['camera']['frame']['parent']
        self.camera_frame = self.config_dict['camera']['frame']['child']
        x = self.config_dict['camera']['frame']['x']
        y = self.config_dict['camera']['frame']['y']
        z = self.config_dict['camera']['frame']['z']
        roll = self.config_dict['camera']['frame']['roll']
        pitch = self.config_dict['camera']['frame']['pitch']
        yaw = self.config_dict['camera']['frame']['yaw']

        self.camera_x_edit.double_value = x
        self.camera_y_edit.double_value = y
        self.camera_z_edit.double_value = z
        self.camera_roll_edit.double_value = roll
        self.camera_pitch_edit.double_value = pitch
        self.camera_yaw_edit.double_value = yaw

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.camera_frame_parent, self.camera_frame)

        self.fov_width_px = self.config_dict['camera']['fov']['width_px']
        self.fov_height_px = self.config_dict['camera']['fov']['height_px']
        self.fov_width_mm = self.config_dict['camera']['fov']['width_mm']
        self.fov_height_mm = self.config_dict['camera']['fov']['height_mm']
        self.roi_width = self.config_dict['camera']['roi']['width_px']
        self.roi_height = self.config_dict['camera']['roi']['height_px']
        self.dof = self.config_dict['camera']['dof_mm']
        self.focal_distance_mm = self.config_dict['camera']['focal_distance_mm']

        self.fov_width_mm_edit.double_value = self.fov_width_mm
        self.fov_height_mm_edit.double_value = self.fov_height_mm
        self.fov_width_px_edit.int_value = self.fov_width_px
        self.fov_height_px_edit.int_value = self.fov_height_px

        # PARTITIONER SETTINGS
        self.viewpoint_dict = None

        self.partitioner.fov_height = self.fov_height_mm * \
            (self.roi_height/self.fov_height_px) / 10

        self.partitioner.fov_width = self.fov_width_mm * \
            (self.roi_width/self.fov_width_px) / 10

        self.partitioner.focal_distance = self.focal_distance_mm / 10

    def _import_model(self, path):
        try:
            self.part_model = o3d.io.read_triangle_mesh(path)
        except:
            print("Error reading model file")
            return
        if self.part_model_units == 'mm':
            scale = 0.1
        elif self.part_model_units == 'cm':
            scale = 1.0
        elif self.part_model_units == 'm':
            scale = 100.0
        elif self.part_model_units == 'in':
            scale = 2.54
        self.part_model.scale(scale, center=(0, 0, 0))
        self.part_point_cloud = None
        self._reset_scene()

    def _import_point_cloud(self, path):
        try:
            self.part_point_cloud = o3d.io.read_point_cloud(path)
        except:
            print("Error reading point_cloud file")
            return
        if self.part_point_cloud_units == 'mm':
            scale = 0.1
        elif self.part_point_cloud_units == 'cm':
            scale = 1.0
        elif self.part_point_cloud_units == 'm':
            scale = 100.0
        elif self.part_point_cloud_units == 'in':
            scale = 2.54
        self.part_point_cloud.scale(scale, center=(0, 0, 0))
        self._reset_scene()

    def _get_and_send_part_frame_tf(self):
        x = self.part_x_edit.double_value
        y = self.part_y_edit.double_value
        z = self.part_z_edit.double_value
        roll = self.part_roll_edit.double_value
        pitch = self.part_pitch_edit.double_value
        yaw = self.part_yaw_edit.double_value

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.part_frame_parent, self.part_frame)

    def _get_and_send_camera_frame_tf(self):
        x = self.camera_x_edit.double_value
        y = self.camera_y_edit.double_value
        z = self.camera_z_edit.double_value
        roll = self.camera_roll_edit.double_value
        pitch = self.camera_pitch_edit.double_value
        yaw = self.camera_yaw_edit.double_value

        T = np.eye(4)
        T[0:3, 3] = [x, y, z]
        T[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [roll, pitch, yaw])

        self.ros_thread.send_transform(
            T, self.camera_frame_parent, self.camera_frame)

    def _on_menu_new(self):
        pass

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".yaml",
            "Config files (.yaml)")
        # dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_open_dialog_cancel)
        dlg.set_on_done(self._on_open_dialog_done)
        self.window.show_dialog(dlg)

    def _on_open_dialog_cancel(self):
        self.window.close_dialog()

    def _on_open_dialog_done(self, config_filename):
        self.window.close_dialog()
        self.load_config(config_filename)

    def load_config(path):
        pass

    def _on_menu_save(self):
        pass

    def _on_menu_save_as(self):
        pass

    def _on_menu_import_model(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                             self.window.theme)
        dlg.set_path(os.path.expanduser("~"))
        dlg.add_filter(
            ".obj .stl", "Triangle mesh (.obj, .stl)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_import_dialog_cancel)
        dlg.set_on_done(self._on_import_model_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_import_pcd(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file",
                             self.window.theme)
        dlg.set_path(os.path.expanduser("~"))
        dlg.add_filter(
            ".ply", "Point cloud data (.ply)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_import_dialog_cancel)
        dlg.set_on_done(self._on_import_pcd_dialog_done)
        self.window.show_dialog(dlg)

    def _on_import_dialog_cancel(self):
        self.window.close_dialog()

    def _on_import_model_dialog_done(self, path):
        self.part_model = o3d.io.read_triangle_mesh(path)
        self.part_model.scale(100, center=(0, 0, 0))
        self.window.close_dialog()

    def _on_import_pcd_dialog_done(self, path):
        self.part_point_cloud = o3d.io.read_point_cloud(path)
        self.part_point_cloud.scale(100, center=(0, 0, 0))
        self.window.close_dialog()

    def _on_menu_quit(self):
        pass

    def _on_menu_preferences(self):
        dlg = gui.Dialog("Preferences")

        em = self.window.theme.font_size

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.preferred_width = 50 * em

        # Interface Preferences

        dlg_layout.add_child(gui.Label("Interface Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_plot_style_select(item, is_double_click):
            plt.style.use(item)

        # plot_style_select = gui.ListView()
        plot_style_select = gui.Combobox()
        # plot_style_select.set_items(plt.style.available)
        styles = plt.style.available
        for style in styles:
            plot_style_select.add_item(style)
        # plot_style_select.set_max_visible_items(3)
        plot_style_select.set_on_selection_changed(on_plot_style_select)

        def on_cmap_select(item, is_double_click):
            self.plot_cmap = item

        # plot_style_select = gui.ListView()
        cmap_select = gui.Combobox()
        # plot_style_select.set_items(plt.style.available)
        cmaps = list(colormaps)
        for cmap in cmaps:
            cmap_select.add_item(cmap)
        # plot_style_select.set_max_visible_items(3)
        cmap_select.set_on_selection_changed(on_cmap_select)

        def on_panel_color_changed(c):
            self.panel_color = gui.Color(c.red, c.green, c.blue, c.alpha)
            self.stereo_camera_panel.background_color = self.panel_color
            self.main_tabs.background_color = self.panel_color
            # self.log_panel.background_color = self.panel_color
            self.camera_config_panel.background_color = self.panel_color
            self.action_panel.background_color = self.panel_color

        panel_color_edit = gui.ColorEdit()
        # Example default RGBA background color
        panel_color_edit.color_value = self.panel_color
        panel_color_edit.set_on_value_changed(on_panel_color_changed)

        def on_background_color_changed(c):
            self.scene_widget.scene.set_background(
                [c.red, c.green, c.blue, c.alpha])

        background_color_edit = gui.ColorEdit()
        bgc = self.scene_widget.scene.background_color
        # Example default RGBA background color
        background_color_edit.color_value = gui.Color(
            bgc[0], bgc[1], bgc[2], bgc[3])
        background_color_edit.set_on_value_changed(on_background_color_changed)

        grid.add_child(gui.Label("Plot Style"))
        grid.add_child(plot_style_select)
        grid.add_child(gui.Label("Color Map"))
        grid.add_child(cmap_select)
        grid.add_child(gui.Label("Panel Color"))
        grid.add_child(panel_color_edit)
        grid.add_child(gui.Label("Background Color"))
        grid.add_child(background_color_edit)

        dlg_layout.add_child(grid)

        # Model Preferences

        dlg_layout.add_child(gui.Label("Model Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_model_color_changed(c):
            self.part_model_material.base_color = [
                c.red, c.green, c.blue, c.alpha]
            # self.scene_widget.scene.modify_geometry_material(
            # self.part_model_name, self.part_model_material)

        model_color_edit = gui.ColorEdit()
        mmc = self.part_model_material.base_color
        model_color_edit.color_value = gui.Color(
            mmc[0], mmc[1], mmc[2], mmc[3])
        model_color_edit.set_on_value_changed(on_model_color_changed)

        grid.add_child(gui.Label("Object Color"))
        grid.add_child(model_color_edit)

        dlg_layout.add_child(grid)

        # Point Cloud Preferences

        dlg_layout.add_child(gui.Label("Point Cloud Preferences"))

        grid = gui.VGrid(2, 0.25 * em)

        def on_pcd_color_changed(c):
            self.live_point_cloud_material.base_color = [
                c.red, c.green, c.blue, c.alpha]
            # self.scene_widget.scene.modify_geometry_material(
            # self.live_point_cloud_name, self.live_point_cloud_material)

        pcd_color_edit = gui.ColorEdit()
        pcdmc = self.live_point_cloud_material.base_color
        pcd_color_edit.color_value = gui.Color(
            pcdmc[0], pcdmc[1], pcdmc[2], pcdmc[3])
        pcd_color_edit.set_on_value_changed(on_pcd_color_changed)

        def on_pcd_size_changed(value):
            self.live_point_cloud_material.point_size = value
            # self.scene_widget.scene.modify_geometry_material(
            # self.live_point_cloud_name, self.live_point_cloud_material)

        pcd_size_edit = gui.Slider(gui.Slider.INT)
        pcd_size_edit.int_value = int(
            self.live_point_cloud_material.point_size)
        pcd_size_edit.set_limits(1, 10)
        pcd_size_edit.set_on_value_changed(on_pcd_size_changed)

        grid.add_child(gui.Label("Point Size"))
        grid.add_child(pcd_size_edit)
        grid.add_child(gui.Label("Point Cloud Color"))
        grid.add_child(pcd_color_edit)

        dlg_layout.add_child(grid)

        # Optionally, connect signals to handle color changes
        # background_color_edit.set_on_value_changed(self._on_background_color_changed)
        # model_color_edit.set_on_value_changed(self._on_object_color_changed)

        # Add ColorEdit widgets to the dialog with labels

        # Create OK button and its callback
        ok_button = gui.Button("OK")

        def on_ok_clicked():
            # Implement saving changes or other actions here
            self.window.close_dialog()
        ok_button.set_on_clicked(on_ok_clicked)

        # Create Cancel button and its callback
        cancel_button = gui.Button("Cancel")

        def on_cancel_clicked():
            self.window.close_dialog()
        cancel_button.set_on_clicked(on_cancel_clicked)

        button_layout = gui.Horiz()

        # Add buttons to the layout
        button_layout.add_stretch()
        button_layout.add_child(ok_button)
        button_layout.add_child(cancel_button)

        # Add the button layout,

        dlg_layout.add_child(button_layout)

        # ... then add the layout as the child of the Dialog

        dlg.add_child(dlg_layout)

        # Show the dialog
        self.window.show_dialog(dlg)

    def _monitor_mouse_event(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            # Pan on
            if event.is_button_down(gui.MouseButton.RIGHT):
                self.pan_pos = (event.x, event.y)
                self.ros_thread.pan_pos = (event.x, event.y)
            # Orbit on
            elif event.is_button_down(gui.MouseButton.LEFT):
                self.orbit_pos = (event.x, event.y)
                self.ros_thread.orbit_pos = (event.x, event.y)
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            # Pan off
            if event.is_button_down(gui.MouseButton.RIGHT):
                self.pan_pos = None
                self.pan_goal = None
                self.ros_thread.pan_goal = self.ros_thread.pan_pos
            # Orbit off
            elif event.is_button_down(gui.MouseButton.LEFT):
                self.orbit_pos = None
                self.orbit_goal = None
                self.ros_thread.orbit_goal = self.ros_thread.orbit_pos
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.Type.DRAG:
            # Change pan goal
            if event.is_button_down(gui.MouseButton.RIGHT):
                self.pan_goal = (event.x, event.y)
                self.ros_thread.pan_goal = (event.x, event.y)
            # Change orbit goal
            if event.is_button_down(gui.MouseButton.LEFT):
                self.orbit_goal = (event.x, event.y)
                self.ros_thread.orbit_goal = (event.x, event.y)
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.Type.WHEEL:
            # Zoom
            if event.wheel_dy > 0:
                self.ros_thread.zoom(100)
            else:
                self.ros_thread.zoom(-100)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_menu_show_axes(self):
        self.scene_widget.scene.show_axes(True)

    def _on_menu_show_grid(self):
        pass

    def _on_menu_show_model(self):
        pass

    def _on_menu_show_point_clouds(self):
        pass

    def _on_menu_show_regions(self):
        pass

    def _on_menu_toggle_viewpoint_generation_panel(self):
        pass

    def _on_menu_toggle_settings_panel(self):
        pass

    def _on_menu_about(self):
        pass

    def _scene_mouse_event(self, event):
        pass

    def _exit(self):
        pass

    def set_panel_color(self, color):
        self.stereo_camera_panel.background_color = color

    def generate_point_cloud():
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    def update_point_cloud(self):
        rgb_image, annotated_rgb_image, depth_image, depth_intrinsic, illuminance_image, gphoto2_image, T = self.ros_thread.get_data()

        rgb_image_o3d = o3d.geometry.Image(rgb_image)
        annotated_rgb_image_o3d = o3d.geometry.Image(annotated_rgb_image)
        depth_image_o3d = o3d.geometry.Image(depth_image)
        illuminance_image_o3d = o3d.geometry.Image(
            cv2.cvtColor(illuminance_image, cv2.COLOR_GRAY2RGB))

        # Get data from ReconstructThread
        live_point_cloud = self.reconstruct_thread.live_point_cloud

        self.reconstruct_thread.depth_image_o3d = depth_image_o3d
        self.reconstruct_thread.rgb_image_o3d = rgb_image_o3d
        self.reconstruct_thread.depth_intrinsic = depth_intrinsic
        self.reconstruct_thread.T = T

        tab_index = self.main_tabs.selected_tab_index

        # REMOVE GEOMETRY ##########################################################

        self.scene_widget.scene.remove_geometry(self.live_point_cloud_name)
        self.scene_widget.scene.remove_geometry("stereo_camera")
        self.scene_widget.scene.remove_geometry("main_camera")
        self.scene_widget.scene.remove_geometry("light_ring")

        # UPDATE SCENE TAB #########################################################

        if tab_index == MyGui.SCENE_TAB:

            # Show/hide and enable/disable UI elements
            self.stereo_camera_panel.enabled = True
            self.stereo_camera_panel.visible = True
            self.camera_config_panel.enabled = False
            self.camera_config_panel.visible = False
            self.monitor_image_panel.enabled = False
            self.monitor_image_panel.visible = False
            self.monitor_image_widget.enabled = False
            self.scene_widget.scene.show_ground_plane(
                True, o3d.visualization.rendering.Scene.GroundPlane.XY)
            self.main_tabs.selected_tab_index = MyGui.SCENE_TAB

            # RGB TAB ################################################

            self.rgb_image.update_image(annotated_rgb_image_o3d)

            # ILLUMINANCE TAB ########################################

            self.illuminance_image.update_image(illuminance_image_o3d)

            # DEPTH TAB ##############################################

            depth_image[depth_image > self.depth_trunc] = 0
            ax = self.webcam_fig.add_subplot()
            pos = ax.imshow(depth_image, cmap=self.plot_cmap,
                            interpolation='none')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(pos, cax=cax)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            self.webcam_fig.clf()
            plot_depth_image_cv2 = cv2.imdecode(
                plot_image, cv2.IMREAD_UNCHANGED)
            plot_depth_image_cv2 = cv2.cvtColor(
                plot_depth_image_cv2, cv2.COLOR_BGR2RGB)
            # Replace black pixels with transparent pixels
            plot_depth_image_cv2[np.all(
                plot_depth_image_cv2 == [0, 0, 0], axis=-1)] = [255*self.stereo_camera_panel.background_color.red,
                                                                255*self.stereo_camera_panel.background_color.green,
                                                                255*self.stereo_camera_panel.background_color.blue]

            image_o3d = o3d.geometry.Image(plot_depth_image_cv2)
            self.depth_image.update_image(image_o3d)

            # 3D SCENE WIDGET ##########################################

            # Add Real-time PCD

            self.scene_widget.scene.add_geometry(
                self.live_point_cloud_name, live_point_cloud, self.live_point_cloud_material)

            # Add Camera

            # stereo_camera = o3d.geometry.LineSet().create_camera_visualization(
            #     depth_intrinsic, extrinsic=np.eye(4))
            # stereo_camera.scale(self.depth_trunc, center=np.array([0, 0, 0]))
            # stereo_camera.transform(T)
            # stereo_camera.scale(100.0, center=np.array([0, 0, 0]))
            # stereo_camera.paint_uniform_color(
            #     np.array([0/255, 255/255, 255/255]))

            main_camera = o3d.geometry.LineSet()
            fov_width_m = 0.001*self.fov_width_mm * \
                (self.roi_width/self.fov_width_px)
            fov_height_m = 0.001*self.fov_height_mm * \
                (self.roi_height/self.fov_height_px)
            focal_distance_m = 0.001*self.focal_distance_mm

            o = np.array([0, 0, 0])
            tl = [-fov_width_m/2, fov_height_m/2, focal_distance_m]
            tr = [fov_width_m/2, fov_height_m/2, focal_distance_m]
            bl = [-fov_width_m/2, -fov_height_m/2, focal_distance_m]
            br = [fov_width_m/2, -fov_height_m/2, focal_distance_m]

            main_camera.points = o3d.utility.Vector3dVector(
                np.array([o, tl, tr, bl, br]))
            main_camera.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]))
            main_camera.transform(T)
            main_camera.scale(100.0, center=np.array([0, 0, 0]))
            main_camera.paint_uniform_color(np.array([255/255, 0/255, 0/255]))

            # self.scene_widget.scene.add_geometry(
            # "stereo_camera", stereo_camera, self.line_material)
            self.scene_widget.scene.add_geometry(
                "main_camera", main_camera, self.line_material)

            # Add Light Ring

            # light_ring_mesh = o3d.geometry.TriangleMesh.create_cylinder(
            #     radius=0.1, height=0.01)
            # light_ring = o3d.geometry.LineSet.create_from_triangle_mesh(
            #     light_ring_mesh)
            # light_ring.transform(T)
            # light_ring.scale(100.0, center=np.array([0, 0, 0]))
            # light_ring.paint_uniform_color(np.array([255/255, 255/255, 0/255]))

            # self.scene_widget.scene.add_geometry(
            #     "light_ring", light_ring, self.line_material)

            # Partitioning Results
            if self.partitioner.is_running:
                progress = self.partitioner.progress
                self.viewpoint_progress_bar.value = progress
                if progress == 1.0:
                    # Enable UI Buttons
                    self.defect_selection.enabled = True
                    self.roi_width_edit.enabled = True
                    self.roi_height_edit.enabled = True
                    self.fov_height_mm_edit.enabled = True
                    self.fov_width_mm_edit.enabled = True
                    self.part_model_file_edit.enabled = True
                    self.part_pcd_file_edit.enabled = True
                    self.generate_viewpoints_button.enabled = True

                    self.partitioner.stop()
                    self.viewpoint_dict = self.partitioner.get_viewpoint_dict()
                    self.scene_widget.scene.remove_geometry(
                        self.part_point_cloud_name)

                    selected_index = self.partitioner.best_path[self.selected_viewpoint]
                    self.viewpoint_stack.selected_index = selected_index + 1
                    self.viewpoint_slider.enabled = True
                    self.viewpoint_slider.set_limits(
                        1, len(self.viewpoint_dict.keys()))

                    self._reset_scene()
                    self.scene_widget.scene.remove_geometry(
                        self.part_point_cloud_name)

                    for i, (region_name, region) in enumerate(self.viewpoint_dict.items()):

                        viewpoint_tf = region['viewpoint']
                        point_cloud = region['point_cloud']
                        origin = region['origin']
                        point = region['point']

                        if i == selected_index:
                            color = [0/255, 255/255, 0/255]
                        else:
                            color = region['color']

                        point_cloud.paint_uniform_color(color)
                        viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
                            radius=1)
                        viewpoint_geom.paint_uniform_color(color)
                        viewpoint_geom.transform(viewpoint_tf)

                        viewpoint_line = o3d.geometry.LineSet()
                        viewpoint_line.points = o3d.utility.Vector3dVector(
                            np.array([origin, point]))
                        viewpoint_line.lines = o3d.utility.Vector2iVector(
                            np.array([[0, 1]]))
                        viewpoint_line.paint_uniform_color(color)

                        self.scene_widget.scene.add_geometry(
                            f"{region_name}_viewpoint", viewpoint_geom, self.viewpoint_material)
                        self.scene_widget.scene.add_geometry(
                            region_name, point_cloud, self.part_point_cloud_material)
                        self.scene_widget.scene.add_geometry(
                            f"{region_name}_line", viewpoint_line, self.line_material)

        # UPDATE MONITOR TAB #########################################################

        elif tab_index == MyGui.MONITOR_TAB:

            # Show/hide and enable/disable UI elements

            self.stereo_camera_panel.enabled = False
            self.stereo_camera_panel.visible = False
            self.camera_config_panel.enabled = True
            self.camera_config_panel.visible = True
            # self.monitor_image_panel.enabled = True
            self.monitor_image_panel.visible = True
            self.scene_widget.scene.show_ground_plane(
                False, o3d.visualization.rendering.Scene.GroundPlane.XY)
            self.main_tabs.selected_tab_index = MyGui.MONITOR_TAB

            # MONITOR TAB ############################################################

            r = self.window.content_rect
            # Scale image evenly to fill the window
            scale = max(r.width / gphoto2_image.shape[0],
                        r.height / gphoto2_image.shape[1])
            gphoto2_image = cv2.resize(
                gphoto2_image, (0, 0), fx=scale, fy=scale)

            if self.pan_pos is not None:
                frame_origin = (self.monitor_image_widget.frame.get_left(),
                                self.monitor_image_widget.frame.get_top())

                frame_width = self.monitor_image_widget.frame.width
                frame_height = self.monitor_image_widget.frame.height

                pos = ((self.pan_pos[0] - frame_origin[0])/frame_width,
                       (self.pan_pos[1] - frame_origin[1])/frame_height)

                width = gphoto2_image.shape[1]
                height = gphoto2_image.shape[0]
                p = (int(pos[0]*width), int(pos[1]*height))

                if self.pan_goal is not None:
                    goal = ((self.pan_goal[0] - frame_origin[0])/frame_width,
                            (self.pan_goal[1] - frame_origin[1])/frame_height)

                    g = (int(goal[0]*width), int(goal[1]*height))
                    gphoto2_image = cv2.arrowedLine(
                        gphoto2_image, p, g, (200, 200, 200), 10)
                # Draw circle at pan_pos
                gphoto2_image = cv2.circle(
                    gphoto2_image, p, 20, (200, 200, 200), -1)

            gphoto2_image_o3d = o3d.geometry.Image(gphoto2_image)

            self.monitor_image_widget.update_image(gphoto2_image_o3d)

        # Update log
        self.log_list = self.ros_thread.read_log()
        # self.log_list.insert(0, "Log " + str(np.random.randint(1000)))
        # self.log_list = self.log_list[:1000]

        self.ros_log_text.set_items(self.log_list)
        self.ros_log_text.selected_index = 0

        return True

    def startThread(self):
        if self.update_delay >= 0:
            threading.Thread(target=self.update_thread).start()

    def update_thread(self):
        def do_update():
            return self.update_point_cloud()

        while not self.is_done:
            time.sleep(self.update_delay)
            print("update_thread")
            with self.lock:
                if self.is_done:  # might have changed while sleeping.
                    break
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update_point_cloud)

    def on_main_window_closing(self):
        with self.lock:
            self.is_done = True

        self.ros_thread.stop()
        self.reconstruct_thread.stop()

        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        # print("tick")
        return self.update_point_cloud()

    def _on_layout(self, layout_context):

        em = self.window.theme.font_size

        r = self.window.content_rect

        self.main_tabs.frame = r
        self.scene_widget.frame = r

        tab_frame_top = 1.5*em
        tab_frame_height = 3.5*em

        self.main_tabs.frame = gui.Rect(
            0, tab_frame_top, r.width, tab_frame_top + tab_frame_height)

        main_frame_top = 2*tab_frame_top + tab_frame_height

        self.scene_widget.frame = r

        # Stereo Camera Panel

        width = 30 * layout_context.theme.font_size
        height = self.stereo_camera_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height < 10 * em:
            width = 7.5 * em

        self.stereo_camera_panel.frame = gui.Rect(
            0, main_frame_top + 2*em, width, height)

        # Camera Config Panel

        self.camera_config_panel.frame = gui.Rect(
            0, main_frame_top + 2*em, width, height)

        # Part Frame Panel

        width = self.part_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        height = self.part_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height < 10 * em:
            width = 7.5 * em

        top = main_frame_top + 2*em
        left = r.width - width

        self.part_frame_panel.frame = gui.Rect(left, top, width, height)

        # camera Frame Panel

        width = self.camera_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        height = self.camera_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height < 10 * em:
            width = 7.5 * em

        top = self.part_frame_panel.frame.get_bottom() + 2*em
        left = r.width - width

        self.camera_frame_panel.frame = gui.Rect(left, top, width, height)

        # Log Panel

        max_height = 10 * em
        height = min(self.log_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height, max_height)

        log_panel_width = r.width
        log_panel_height = height
        self.log_panel.frame = gui.Rect(
            0, r.height - log_panel_height + 1.5*em, log_panel_width, log_panel_height)

        # Action Panel

        action_panel_height = 4*em
        action_panel_width = log_panel_width

        top = self.log_panel.frame.get_top() - action_panel_height
        left = 0

        self.action_panel.frame = gui.Rect(
            left, top, action_panel_width, action_panel_height)

        # Monitor Image Panel
        height = self.monitor_image_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        height = self.action_panel.frame.get_top() - main_frame_top

        self.monitor_image_panel.frame = gui.Rect(
            0, main_frame_top, r.width, height)


def main(args=None):
    rclpy.init(args=args)
    print(args)

    gui.Application.instance.initialize()

    thread_delay = 0.1
    use_tick = -1

    dpcApp = MyGui(use_tick)
    dpcApp.startThread()

    gui.Application.instance.run()

    rclpy.shutdown()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()
