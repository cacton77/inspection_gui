#!/usr/bin/env python3
import rclpy

import os
import yaml
import platform
import cv2  # OpenCV library
import datetime
import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
from inspection_gui.assets.materials import Materials
from inspection_gui.focus_monitor import FocusMonitor
from inspection_gui.threads.ros import RosThread
from inspection_gui.threads.reconstruct import ReconstructThread
from inspection_gui.threads.partitioner import Partitioner, NPCD
from inspection_gui.threads.plotting import PlottingThread
from inspection_gui.threads.lighting import LightMap
# from inspection_gui.threads.moveit import MoveItThread

# Custom UI Panel definitions
from inspection_gui.panels.focus_monitor import FocusMonitorPanel

plt.style.use('dark_background')
# plt.style.use('bmh')

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
    MENU_SHOW_PATH = 19
    MENU_ABOUT = 21

    # Main Tabs
    SCENE_TAB = 0
    MONITOR_TAB = 1

    # Stereo Camera Tabs
    STEREO_RGB_TAB = 0
    STEREO_DEPTH_TAB = 1
    STEREO_ILLUMINANCE_TAB = 2

    background_color = Materials.background_color
    panel_color = Materials.panel_color
    header_footer_color = Materials.header_footer_color

    part_model_material = Materials.part_model_material
    viewpoint_material = Materials.viewpoint_material
    selected_viewpoint_material = Materials.selected_viewpoint_material
    line_material = Materials.line_material
    selected_line_material = Materials.selected_line_material
    part_point_cloud_material = Materials.part_point_cloud_material
    live_point_cloud_material = Materials.live_point_cloud_material
    best_path_material = Materials.best_path_material
    axes_line_material = Materials.axes_line_material
    grid_line_material = Materials.grid_line_material
    ground_plane_material = Materials.ground_plane_material
    camera_line_material = Materials.camera_line_material

    # Flags
    moving_to_viewpoint_flag = False

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

        ###############################

        self.plot_cmap = 'Greys'
        self.webcam_fig = plt.figure()

        # Threads
        self.ros_thread = RosThread(stream_id=0)  # 0 id for main camera
        # self.moveit_thread = MoveItThread(name='moveit_py_planning_scene')
        self.reconstruct_thread = ReconstructThread(rate=20)
        self.plotting_thread = PlottingThread()

        light_locations = np.loadtxt(
            self.inspection_root_path + '/Lights/led_positions.csv', delimiter=',')
        shape_mm = (200, 200)
        dpmm = 10
        self.light_map = LightMap(shape_mm, dpmm, light_locations)

        self.ros_thread.start()  # processing frames in input stream
        self.reconstruct_thread.start()  # processing frames in input stream
        self.plotting_thread.start()
        self.light_map.start()
        # self.moveit_thread.start()

        # 3D SCENE ################################################################
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.scene_widget.scene.set_background(self.background_color)
        self.scene_widget.enable_scene_caching(False)

        # MAIN TABS ################################################################

        self.main_tabs = gui.TabControl()
        self.main_tabs.background_color = self.header_footer_color

        # 3D SCENE TAB ####################

        # Add XY Axes

        self.xy_axes = o3d.geometry.LineSet()
        self.xy_axes.points = o3d.utility.Vector3dVector(
            np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]))
        self.xy_axes.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1], [2, 3]]))
        self.xy_axes.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0]]))

        # Part Model
        self.part_model_name = "Part Model"
        self.part_model = None
        self.part_model_units = self.config_dict['part']['model_units']

        part_model_file = self.config_dict['part']['model']

        # Part Point Cloud
        self.part_point_cloud_name = "Part Point Cloud"
        self.part_point_cloud = None
        self.part_point_cloud_units = self.config_dict['part']['point_cloud_units']

        part_pcd_file = self.config_dict['part']['point_cloud']

        # Viewpoints etc.

        # Live Point Cloud
        self.live_point_cloud_name = "Point Cloud"

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
        scene_ribbon_grid = gui.VGrid(3, 0.25 * em)

        part_horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self.config_switch = False
        load_part_button = gui.Button("Load Part")
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

        part_horiz.add_child(gui.Label("Part: "))
        part_horiz.add_child(load_part_button)
        part_horiz.add_fixed(0.5 * em)
        part_horiz.add_child(grid)

        # Scene ribbon defect panels

        defect_horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        def _on_defect_select(defect_name, i):
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

            self.partitioner.fov_height = self.fov_height_mm * \
                (self.roi_height/self.fov_height_px) / 10

            self.partitioner.fov_width = self.fov_width_mm * \
                (self.roi_width/self.fov_width_px) / 10

            self.partitioner.focal_distance = self.focal_distance_mm / 10

            defect_dir = self.inspection_root_path + '/Parts/' + \
                self.part_model_name + '/Defects/' + defect_name

            # Save the viewpoint dictionary to a yaml file
            viewpoint_dict_path = defect_dir + '/viewpoint_dict.yaml'
            if os.path.exists(viewpoint_dict_path):
                self.load_viewpoints(viewpoint_dict_path)

        self.defects = self.config_dict['defects']

        self.defect_selection = gui.Combobox()
        for i in range(len(self.defects)):
            self.defect_selection.add_item(self.defects[i]['name'])

        self.defect_selection.set_on_selection_changed(_on_defect_select)

        defect_horiz.add_child(gui.Label("Defect Selection:"))
        defect_horiz.add_child(self.defect_selection)
        defect_horiz.add_fixed(10 * em)

        scene_ribbon_grid.add_child(part_horiz)
        scene_ribbon_grid.add_child(defect_horiz)

        self.scene_ribbon.add_child(scene_ribbon_grid)

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

        # LIGHTS PANEL ###############################################################

        self.light_panel = gui.CollapsableVert("Light Control", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.light_panel.background_color = self.panel_color
        self.light_panel.set_is_open(False)
        self.num_leds = 148

        def _on_light_intensity_changed(intensity):
            self.light_map.set_intensity(intensity)
            self.set_pixels()

        def _on_light_position_x_changed(x):
            self.light_map.set_mu_x(x)
            self.set_pixels()

        def _on_light_position_y_changed(y):
            self.light_map.set_mu_y(y)
            self.set_pixels()

        def _on_light_sigma_changed(sigma):
            self.light_map.set_sigma(sigma)
            self.set_pixels()

        # Light map
        # Import light positions from led_positions.csv
        self.light_map_image = gui.ImageWidget()

        self.light_intensity_slider = gui.Slider(gui.Slider.INT)
        self.light_intensity_slider.set_limits(0, 255)
        self.light_intensity_slider.set_on_value_changed(
            _on_light_intensity_changed)

        self.light_sigma_slider = gui.Slider(gui.Slider.DOUBLE)
        self.light_sigma_slider.set_limits(0.1, 100)
        self.light_sigma_slider.set_on_value_changed(
            _on_light_sigma_changed)
        self.light_position_x_slider = gui.Slider(gui.Slider.DOUBLE)
        self.light_position_x_slider.set_limits(-0.5, 0.5)
        self.light_position_x_slider.set_on_value_changed(
            _on_light_position_x_changed)
        self.light_position_y_slider = gui.Slider(gui.Slider.DOUBLE)
        self.light_position_y_slider.set_limits(-0.5, 0.5)
        self.light_position_y_slider.set_on_value_changed(
            _on_light_position_y_changed)

        self.light_panel.add_child(gui.Label("Light Map: "))
        self.light_panel.add_child(self.light_map_image)

        self.light_panel.add_child(gui.Label("Intensity: "))
        self.light_panel.add_child(self.light_intensity_slider)
        self.light_panel.add_child(gui.Label("Sigma: "))
        self.light_panel.add_child(self.light_sigma_slider)
        self.light_panel.add_child(gui.Label("X: "))
        self.light_panel.add_child(self.light_position_x_slider)
        self.light_panel.add_child(gui.Label("Y: "))
        self.light_panel.add_child(self.light_position_y_slider)

        # VIEWPOINT GENERATION PANEL ###############################################

        self.viewpoint_generation_panel = gui.CollapsableVert("Viewpoints", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.viewpoint_generation_panel.background_color = self.panel_color
        self.viewpoint_generation_panel.set_is_open(False)

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

        self.segmentation_method_select = gui.Combobox()
        self.segmentation_method_select.add_item("Nandagopal Method")

        segmentation_grid = gui.VGrid(2, 0.25 * em)
        segmentation_grid.add_child(gui.Label("Partitioning Method: "))
        segmentation_grid.add_child(self.segmentation_method_select)

        self.generate_viewpoints_button = gui.Button("Generate Viewpoints")
        self.generate_viewpoints_button.set_on_clicked(
            _on_viewpoint_generation_button_clicked)

        self.viewpoint_generation_panel.add_child(segmentation_grid)
        self.viewpoint_generation_panel.add_fixed(0.5 * em)
        self.viewpoint_generation_panel.add_child(
            self.generate_viewpoints_button)

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

        self.tf_part_to_world = T

        self.ros_thread.send_transform(
            self.tf_part_to_world, self.part_frame_parent, self.part_frame)

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

        self.tf_camera_to_tool = T

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

        # INSPECTION ACTION PANEL ##################################################

        self.footer_panel = gui.Vert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.footer_panel.background_color = self.header_footer_color

        # VIEWPOINT GENERATION ######################

        # self.partitioning_progress_queue = Queue()
        # self.partitioning_results_queue = Queue()

        # npcd = NPCD.from_o3d_point_cloud(o3d.geometry.PointCloud())

        # self.partitioning_process = Process(
        #     target=self.partitioner.rg_not_smart_partition_worker, args=(npcd,
        #                                                                  self.partitioning_progress_queue,
        #                                                                  self.partitioning_results_queue))

        self.viewpoint_stack = gui.StackedWidget()
        self.viewpoint_progress_bar = gui.ProgressBar()
        self.viewpoint_progress_bar.background_color = gui.Color(0, 1, 0, 0.8)

        self.selected_viewpoint = 0
        self.viewpoint_slider = gui.Slider(gui.Slider.INT)
        self.viewpoint_slider.set_limits(0, 100)
        self.viewpoint_slider.enabled = False
        self.viewpoint_slider.set_on_value_changed(self.select_viewpoint)
        self.viewpoint_stack.add_child(self.viewpoint_progress_bar)
        self.viewpoint_stack.add_child(self.viewpoint_slider)
        self.viewpoint_stack.selected_index = 0
        action_grid = gui.Vert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        horiz.add_child(self.viewpoint_stack)
        action_grid.add_child(horiz)

        self.move_button = gui.Button("Move")

        def _on_move_button_clicked():
            self.move_to_selected_viewpoint()

        self.move_button.set_on_clicked(_on_move_button_clicked)

        # Image capture
        self.image_count = 0

        def _on_capture_image():
            # Get date and time and use as filename
            file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            file_path = self.inspection_root_path + \
                '/Images/' + file_name + '.jpg'
            self.ros_thread.capture_image(file_path)
            self.image_count += 1

        self.capture_image_button = gui.Button("Capture")
        self.capture_image_button.set_on_clicked(_on_capture_image)
        # action_grid.add_child(horiz)

        def _on_go_button_clicked():
            self.t0 = time.time()
            self.running = not self.running
            if self.running:
                self.go_button.text = " Stop "
                self.go_button.background_color = gui.Color(0.8, 0.0, 0.0, 1.0)
            else:
                self.go_button.text = " Go "
                self.go_button.background_color = gui.Color(0.0, 0.8, 0.0, 1.0)

        self.t0 = 0
        self.running = False
        self.go_button = gui.Button(" Go ")
        self.go_button.background_color = gui.Color(0.0, 0.8, 0.0, 1.0)
        self.go_button.toggleable = True
        self.go_button.set_on_clicked(_on_go_button_clicked)

        horiz = gui.Horiz(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        horiz.add_child(action_grid)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(self.move_button)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(gui.Button("Focus"))
        horiz.add_fixed(0.5 * em)
        horiz.add_child(self.capture_image_button)
        horiz.add_fixed(0.5 * em)
        horiz.add_child(self.go_button)

        self.footer_panel.add_child(horiz)

        # LOG PANEL ################################################################

        # Plotting focus metrics

        self.log_panel = gui.CollapsableVert("Log", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.log_panel.background_color = gui.Color(
            36/255, 36/255, 36/255, 1.0)
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

        self.monitor_image_panel = gui.Vert(0, gui.Margins(
            0, 0, 0, 0))

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

        # RIGHT PANEL ###############################################################

        self.right_panel = gui.CollapsableVert("", 0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self.right_panel.background_color = gui.Color(0, 0, 0, 0.8)

        # Tabs
        self.right_panel_tabs = gui.TabControl()

        # MAIN CAMERA PANEL ########################

        # MAIN CAMERA PANEL #########################################################

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

        svert = gui.ScrollableVert(0.25 * em)
        svert.background_color = self.panel_color

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

        svert.add_child(gui.Label('FOV:'))
        svert.add_child(fov_px_grid)
        svert.add_child(fov_mm_grid)
        svert.add_child(gui.Label('ROI:'))
        svert.add_child(roi_grid)
        svert.add_child(gui.Label('Focal Depth:'))
        svert.add_child(dof_focal_distance_grid)

        # Camera settings

        camera_params = self.ros_thread.read_camera_params()

        def on_camera_param_changed(value):
            self.ros_thread.set_camera_params()

        def _on_aperture_changed(value, i):
            self.ros_thread.set_camera_param(self.aperture_name, value)

        def _on_shutterspeed_changed(value, i):
            self.ros_thread.set_camera_param(self.shutterspeed_name, value)

        def _on_iso_changed(value, i):
            self.ros_thread.set_camera_param(self.iso_name, value)

        # Order the camera parameters

        param_names_ordered = camera_params.keys()

        grid = gui.VGrid(2, 0.25 * em)

        # Loop through camera_params and add widgets to grid
        for name in param_names_ordered:

            # Check if parameter is ISO, Shutterspeed, or Aperture
            # If parameter name contains 'iso'
            if 'iso' in name:
                self.iso_select = gui.Combobox()
                description = camera_params[name]
                for choice in description['choices']:
                    self.iso_select.add_item(choice)

                self.iso_name = name
                self.iso_select.set_on_selection_changed(
                    _on_iso_changed)

                grid.add_child(gui.Label("ISO: "))
                grid.add_child(self.iso_select)

            elif 'shutterspeed' in name:
                self.shutterspeed_select = gui.Combobox()
                description = camera_params[name]
                for choice in description['choices']:
                    self.shutterspeed_select.add_item(choice)

                self.shutterspeed_name = name
                self.shutterspeed_select.set_on_selection_changed(
                    _on_shutterspeed_changed)

                grid.add_child(gui.Label("Shutterspeed: "))
                grid.add_child(self.shutterspeed_select)
            elif 'aperture' in name or 'fstop' in name or 'f-stop' in name or 'f/stop' in name or 'f-number' in name or 'F-Number' in name:
                self.aperture_select = gui.Combobox()
                description = camera_params[name]
                for choice in description['choices']:
                    self.aperture_select.add_item(choice)

                self.aperture_name = name
                self.aperture_select.set_on_selection_changed(
                    _on_aperture_changed)

                grid.add_child(gui.Label("Aperture: "))
                grid.add_child(self.aperture_select)
            else:
                continue

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
            # self.main_camera_panel.add_child(grid)

        svert.add_child(gui.Label('Camera Settings:'))
        svert.add_child(grid)

        self.right_panel_tabs.add_tab("Camera", svert)

        # METRIC PANEL ########################

        focus_vert = gui.ScrollableVert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        grid = gui.VGrid(2, 0.25 * em)
        self.focus_metric_select = gui.Combobox()
        focus_metric_names = FocusMonitor.get_metrics()
        for metric in focus_metric_names:
            self.focus_metric_select.add_item(metric)

        def on_focus_metric_changed(name, i):
            self.ros_thread.set_focus_metric(name)

        self.focus_metric_select.set_on_selection_changed(
            on_focus_metric_changed)

        grid.add_child(gui.Label("Focus Metric: "))
        grid.add_child(self.focus_metric_select)

        self.focus_metric_figure = plt.figure()
        self.focus_metric_plot_image = gui.ImageWidget()

        self.focus_image_figure = plt.figure()
        self.focus_metric_image = gui.ImageWidget()

        focus_vert.add_child(grid)
        focus_vert.add_child(self.focus_metric_plot_image)
        focus_vert.add_child(self.focus_metric_image)

        self.right_panel_tabs.add_tab("Focus", focus_vert)

        # STEREO CAMERA TAB #########################################################

        stereo_vert = gui.ScrollableVert(0, gui.Margins(
            0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

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

        self.stereo_camera_tabs = gui.TabControl()
        # tabs.add_tab("RGB", rgb_grid)
        self.stereo_camera_tabs.add_tab("RGB", self.rgb_image)
        self.stereo_camera_tabs.add_tab("Depth", depth_grid)
        self.stereo_camera_tabs.add_tab("Illuminance", illuminance_grid)
        self.stereo_camera_tabs.add_child(gui.TabControl())

        stereo_vert.add_child(self.stereo_camera_tabs)
        self.right_panel_tabs.add_tab("Stereo Camera", stereo_vert)
        self.right_panel.add_child(self.right_panel_tabs)

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
            view_menu.add_item("Show Path", MyGui.MENU_SHOW_PATH)
            view_menu.set_checked(MyGui.MENU_SHOW_PATH, False)
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
        w.set_on_menu_item_activated(
            MyGui.MENU_SHOW_PATH, self._on_menu_show_path)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_VIEWPOINT,
                                     self._on_menu_toggle_viewpoint_generation_panel)
        w.set_on_menu_item_activated(MyGui.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(
            MyGui.MENU_ABOUT, self._on_menu_about)
        # ----

        self._reset_scene()

        defect_name = self.defect_selection.selected_text
        defect_dir = self.inspection_root_path + '/Parts/' + \
            self.part_model_name + '/Defects/' + defect_name
        viewpoint_dict_path = defect_dir + '/viewpoint_dict.yaml'
        if os.path.exists(viewpoint_dict_path):
            self.load_viewpoints(viewpoint_dict_path)

        # Setup the Camera
        self.scene_widget.setup_camera(
            60, self.scene_widget.scene.bounding_box, [1, 0, 0])
        # self.scene_widget.set_view_controls(
        #     gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.scene_widget.set_view_controls(
            gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
        self.scene_widget.scene.show_axes(False)
        # self.scene_widget.scene.show_skybox(True)
        # self.show_ground_plane = True
        # self.scene_widget.scene.show_ground_plane(
        # self.show_ground_plane, o3d.visualization.rendering.Scene.GroundPlane.XY)

        self.main_tabs.add_tab("3D Scene", self.scene_ribbon)
        self.main_tabs.add_tab("Monitor", self.monitor_ribbon)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.monitor_image_panel)
        self.window.add_child(self.main_tabs)
        self.window.add_child(self.part_frame_panel)
        self.window.add_child(self.camera_frame_panel)
        self.window.add_child(self.footer_panel)
        self.window.add_child(self.log_panel)
        self.window.add_child(self.viewpoint_generation_panel)
        self.window.add_child(self.light_panel)
        self.window.add_child(self.right_panel)
        self.window.set_on_layout(self._on_layout)

        self.last_draw_time = time.time()

    def _reset_scene(self):
        self.scene_widget.scene.clear_geometry()

        if self.part_model is not None:
            self.scene_widget.scene.add_geometry(
                self.part_model_name, self.part_model, self.part_model_material)
        if self.part_point_cloud is not None:
            self.scene_widget.scene.add_geometry(
                self.part_point_cloud_name, self.part_point_cloud, self.part_point_cloud_material)

        # Find bounding box region geometry
        if self.part_model is not None:
            bbox = self.part_model.get_axis_aligned_bounding_box()
            # self.scene_widget.scene.add_geometry(
            # 'bounding box', bbox, self.axes_line_material)
            # scale xy_axes by the bounding box diagonal
            diag = bbox.get_max_bound() - bbox.get_min_bound()
            diag = np.linalg.norm(np.asarray(diag))

            # XY Axes

            self.xy_axes = o3d.geometry.LineSet()

            points = [[-10, 0, 0], [10, 0, 0], [0, -10, 0], [0, 10, 0]]
            lines = [[0, 1], [2, 3]]
            colors = [[1, 0, 0], [0, 1, 0]]

            x0, y0, _ = bbox.get_min_bound()
            # Round x0 and y0 to the nearest 10
            x0 = 10 * (round(x0 / 10) - 1)
            y0 = 10 * (round(y0 / 10) - 1)
            x1, y1, _ = bbox.get_max_bound()
            x1 = 10 * (round(x1 / 10) + 1)
            y1 = 10 * (round(y1 / 10) + 1)

            self.xy_axes.points = o3d.utility.Vector3dVector(
                np.array(points))
            self.xy_axes.lines = o3d.utility.Vector2iVector(
                np.array(lines))
            self.xy_axes.colors = o3d.utility.Vector3dVector(
                np.array(colors))

            # Grid

            self.grid = o3d.geometry.LineSet()
            points = []
            lines = []
            colors = []

            for x in np.linspace(x0, x1, round((x1 - x0) / 10) + 1):
                points.append([x, y0, 0])
                points.append([x, y1, 0])
                lines.append([len(points)-2, len(points)-1])
                colors.append([0.3, 0.3, 0.3])
            for y in np.linspace(y0, y1, round((y1 - y0) / 10) + 1):
                points.append([x0, y, 0])
                points.append([x1, y, 0])
                lines.append([len(points)-2, len(points)-1])
                colors.append([0.3, 0.3, 0.3])

            self.grid.points = o3d.utility.Vector3dVector(
                np.array(points))
            self.grid.lines = o3d.utility.Vector2iVector(
                np.array(lines))
            self.grid.colors = o3d.utility.Vector3dVector(
                np.array(colors))

            ground_plane = o3d.geometry.TriangleMesh()
            ground_plane.vertices = o3d.utility.Vector3dVector(
                np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]))
            ground_plane.triangles = o3d.utility.Vector3iVector(
                np.array([[0, 1, 2], [0, 2, 3]]))
            ground_plane.vertex_normals = o3d.utility.Vector3dVector(
                np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]))
            self.scene_widget.scene.add_geometry(
                'ground plane', ground_plane, self.ground_plane_material)

        self.scene_widget.scene.add_geometry(
            'xy axes', self.xy_axes, self.axes_line_material)
        self.scene_widget.scene.add_geometry(
            'grid', self.grid, self.grid_line_material)

        self.scene_widget.setup_camera(
            60, self.scene_widget.scene.bounding_box, [0, 0, 0])

        self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()

        # Add a skybox made from a sphere
        # skybox_material = o3d.visualization.rendering.MaterialRecord()
        # skybox_material.base_color = [0.5, 0.5, 0.5, 1.0]
        # skybox_material.shader = "defaultLit"
        # sphere_box = o3d.geometry.TriangleMesh.create_box(1000, 1000, 1000)
        # sphere_box.compute_vertex_normals()
        # sphere_box.paint_uniform_color([0.5, 0.5, 0.5])
        # sphere_box.translate([-500, -500, -500])
        # self.scene_widget.scene.add_geometry(
        #     'skybox', sphere_box, skybox_material)

    def _send_transform(self, T, parent, child):
        self.ros_thread.send_transform(T, parent, child)

    def _on_load_part_config(self):
        self.config_switch = not self.config_switch

        if self.config_switch:
            self.config_file = os.path.expanduser(
                "~") + '/Inspection/Parts/config/blade.yaml'
        else:
            self.config_file = os.path.expanduser(
                "~") + '/Inspection/Parts/config/default.yaml'
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

        self.tf_part_to_world = T

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
        self.roi_height_edit.int_value = self.roi_height
        self.roi_width_edit.int_value = self.roi_width
        self.dof_edit.double_value = self.dof
        self.focal_distance_edit.double_value = self.focal_distance_mm

        # PARTITIONER SETTINGS

        self.partitioner.fov_height = self.fov_height_mm * \
            (self.roi_height/self.fov_height_px) / 10

        self.partitioner.fov_width = self.fov_width_mm * \
            (self.roi_width/self.fov_width_px) / 10

        self.partitioner.dof = self.dof / 10
        self.partitioner.focal_distance = self.focal_distance_mm / 10

        # Move selected index to 0
        self.selected_viewpoint = 0

        # Regions and Viewpoints

        defect_name = self.defect_selection.selected_text
        defect_dir = self.inspection_root_path + '/Parts/' + \
            self.part_model_name + '/Defects/' + defect_name
        viewpoint_dict_path = defect_dir + '/viewpoint_dict.yaml'
        if os.path.exists(viewpoint_dict_path):
            self.load_viewpoints(viewpoint_dict_path)
        else:
            self.viewpoint_dict = None

    def _import_model(self, path):
        try:
            self.part_model = o3d.io.read_triangle_mesh(path)
            self.part_model.compute_vertex_normals()
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

        # Create a directory for the part model
        self.part_model_name = os.path.basename(path).split('.')[0]
        part_path = self.inspection_root_path + '/Parts/' + self.part_model_name
        if not os.path.exists(part_path):
            os.makedirs(part_path, exist_ok=True)

        # Save the part model to the part directory
        o3d.io.write_triangle_mesh(
            part_path + '/' + self.part_model_name + '.stl', self.part_model)

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

        self.tf_part_to_world = T

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

    def set_pixels(self):
        # Make every tenth pixel white
        pixel_values = self.light_map.get_pixel_values()
        pixel_colors = []
        for value in pixel_values:
            pixel_colors.append((value, value, value))
        self.ros_thread.pixels_to(pixel_colors)

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
        # End threads
        self.set_lights(0)

        gui.Application.instance.quit()

        rclpy.shutdown()
        time.sleep(1)
        exit()

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
            self.main_tabs.background_color = self.panel_color
            # self.log_panel.background_color = self.panel_color
            self.footer_panel.background_color = self.panel_color

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
                self.ros_thread.zoom(500)
            else:
                self.ros_thread.zoom(-500)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_menu_show_axes(self):
        self.scene_widget.scene.show_axes(True)

    def _on_menu_show_grid(self):
        self.scene_widget.scene.show_ground_plane(
            not self.show_ground_plane, o3d.visualization.rendering.Scene.GroundPlane.XY)
        self.show_ground_plane = not self.show_ground_plane

    def _on_menu_show_model(self):
        pass

    def _on_menu_show_point_clouds(self):
        pass

    def _on_menu_show_regions(self):
        pass

    def _on_menu_show_path(self):
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
        pass

    def generate_point_cloud():
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    def save_viewpoints(self):
        self.viewpoint_dict = self.partitioner.get_viewpoint_dict()
        defect_name = self.defect_selection.selected_text

        # If there is not a folder for the defect, create one
        defect_dir = self.inspection_root_path + '/Parts/' + \
            self.part_model_name + '/Defects/' + defect_name
        viewpoint_dir = defect_dir + '/Viewpoints'
        if not os.path.exists(defect_dir):
            os.makedirs(viewpoint_dir, exist_ok=True)

        # Delete all items in viewpoint directory
        for item in os.listdir(viewpoint_dir):
            os.remove(os.path.join(viewpoint_dir, item))

        # Save the region pcd to the viewpoint directory and remove PointCloud from dictionary
        for i, (region_name, region) in enumerate(self.viewpoint_dict['regions'].items()):
            region_pcd_path = viewpoint_dir + '/' + region_name + '.ply'
            region['point_cloud_path'] = region_pcd_path
            o3d.io.write_point_cloud(region_pcd_path, region['point_cloud'])
            del region['point_cloud']

        # Save the viewpoint dictionary to a yaml file
        viewpoint_dict_path = defect_dir + '/viewpoint_dict.yaml'
        with open(viewpoint_dict_path, 'w') as file:
            yaml.dump(self.viewpoint_dict, file)

        return viewpoint_dict_path

    def load_viewpoints(self, viewpoint_dict_path):
        # Load the viewpoint dictionary from a yaml file
        # with open(viewpoint_dict_path, 'r') as file:
        # self.viewpoint_dict = yaml.load(file, Loader=yaml.FullLoader)

        self.viewpoint_dict = yaml.load(
            open(viewpoint_dict_path), Loader=yaml.FullLoader)

        best_path = self.viewpoint_dict['best_path']

        for region_name, region in self.viewpoint_dict['regions'].items():
            region_pcd_path = region['point_cloud_path']
            region['point_cloud'] = o3d.io.read_point_cloud(region_pcd_path)

        self.scene_widget.scene.remove_geometry(self.part_point_cloud_name)
        self._reset_scene()
        self.scene_widget.scene.remove_geometry(
            self.part_point_cloud_name)

        fov_height_m = 0.001*self.fov_height_mm * \
            (self.roi_height/self.fov_height_px)
        viewpoint_marker_size = fov_height_m/0.3048

        for path_step, i in enumerate(best_path):

            region_name = f"region_{i}"
            region = self.viewpoint_dict['regions'][region_name]

            viewpoint_tf = region['viewpoint']
            point_cloud = region['point_cloud']
            origin = region['origin']
            point = region['point']

            viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
                radius=viewpoint_marker_size)
            viewpoint_geom.transform(viewpoint_tf)

            viewpoint_material = self.viewpoint_material
            val = path_step/len(best_path)
            cmap = colormaps[self.plot_cmap]
            color = list(cmap(val))[:3]
            region['color'] = color
            viewpoint_geom.paint_uniform_color(color)

            point_cloud.paint_uniform_color(color)

            viewpoint_line = o3d.geometry.LineSet()
            viewpoint_line.points = o3d.utility.Vector3dVector(
                np.array([origin, point]))
            viewpoint_line.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1]]))

            self.scene_widget.scene.add_geometry(
                f"{region_name}_viewpoint", viewpoint_geom, viewpoint_material)
            self.scene_widget.scene.add_geometry(
                region_name, point_cloud, self.part_point_cloud_material)

        # Set selected viewpoint

        selected_index = best_path[0]

        self.viewpoint_stack.selected_index = 1
        self.viewpoint_slider.enabled = True
        self.viewpoint_slider.set_limits(
            1, len(self.viewpoint_dict['regions'].keys()))
        self.viewpoint_slider.int_value = selected_index + 1

        # self.selected_viewpoint = best_path[0]
        self.viewpoint_slider.int_value = 1
        self.select_viewpoint(1)

        # Generate path line

        cmap = colormaps[self.plot_cmap]
        color = list(cmap(0.5))[:3]

        path_line = o3d.geometry.LineSet()
        path_points = []
        for i in self.viewpoint_dict['best_path']:
            path_points.append(
                self.viewpoint_dict['regions'][f'region_{i}']['point'])
        path_line.points = o3d.utility.Vector3dVector(np.array(path_points))
        path_line.lines = o3d.utility.Vector2iVector(
            np.array([[i, i+1] for i in range(len(path_points)-1)]))

        self.scene_widget.scene.add_geometry(
            'viewpoint_path', path_line, self.best_path_material)

    def select_viewpoint(self, value):
        old_region_name = f"region_{self.selected_viewpoint}"
        self.scene_widget.scene.remove_geometry(old_region_name)
        self.scene_widget.scene.remove_geometry(f"{old_region_name}_line")
        self.scene_widget.scene.remove_geometry(
            f"{old_region_name}_viewpoint")

        old_region = self.viewpoint_dict['regions'][old_region_name]

        viewpoint_tf = old_region['viewpoint']
        point_cloud = old_region['point_cloud']
        origin = old_region['origin']
        point = old_region['point']
        color = old_region['color']

        fov_height_m = 0.001*self.fov_height_mm * \
            (self.roi_height/self.fov_height_px)
        viewpoint_marker_size = fov_height_m/0.3048

        point_cloud.paint_uniform_color(color)
        viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
            radius=viewpoint_marker_size)
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
        # self.scene_widget.scene.add_geometry(
        # f"{old_region_name}_line", viewpoint_line, self.selected_line_material)

        # get new selected_viewpoint
        self.selected_viewpoint = self.viewpoint_dict['best_path'][int(
            value-1)]

        new_region_name = f"region_{self.selected_viewpoint}"
        self.scene_widget.scene.remove_geometry(new_region_name)
        self.scene_widget.scene.remove_geometry(f"{new_region_name}_line")
        self.scene_widget.scene.remove_geometry(
            f"{new_region_name}_viewpoint")

        new_region = self.viewpoint_dict['regions'][new_region_name]

        viewpoint_tf = new_region['viewpoint']
        point_cloud = new_region['point_cloud']
        origin = new_region['origin']
        point = new_region['point']
        color = [0.0, 1.0, 0.0]

        # Generate UVs from selected viewpoint
        def project_texture(mesh, texture, intrinsic, extrinsic):
            vertices = np.asarray(mesh.vertices)
            vertices_homogeneous = np.hstack(
                (vertices, np.ones((vertices.shape[0], 1))))
            projected_vertices = intrinsic.intrinsic_matrix @ (
                extrinsic @ vertices_homogeneous.T)[:3, :]
            projected_vertices /= projected_vertices[2, :]
            uvs = projected_vertices[:2, :].T / \
                np.array([texture.shape[1], texture.shape[0]])
            uvs = np.clip(uvs, 0, 1)
            return uvs

        # uvs = project_texture(self.part_model, texture, intrinsic, T)

        viewpoint_marker_size = 2 * viewpoint_marker_size

        point_cloud.paint_uniform_color(color)
        viewpoint_geom = o3d.geometry.TriangleMesh.create_sphere(
            radius=viewpoint_marker_size)
        # viewpoint_geom.paint_uniform_color(color)
        viewpoint_geom.transform(viewpoint_tf)

        viewpoint_line = o3d.geometry.LineSet()
        viewpoint_line.points = o3d.utility.Vector3dVector(
            np.array([origin, point]))
        viewpoint_line.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1]]))
        viewpoint_line.paint_uniform_color(color)

        self.scene_widget.scene.add_geometry(
            f"{new_region_name}_viewpoint", viewpoint_geom, self.selected_viewpoint_material)
        self.scene_widget.scene.add_geometry(
            new_region_name, point_cloud, self.part_point_cloud_material)
        # self.scene_widget.scene.add_geometry(
        # f"{new_region_name}_line", viewpoint_line, self.selected_line_material)

    def move_to_selected_viewpoint(self):
        self.moving_to_viewpoint_flag = True

        # Get selected viewpoint
        selected_region = self.viewpoint_dict['regions'][f'region_{self.selected_viewpoint}']
        viewpoint = np.array(selected_region['viewpoint'])

        # Convert position to meters
        viewpoint[0, 3] = viewpoint[0, 3] / 100
        viewpoint[1, 3] = viewpoint[1, 3] / 100
        viewpoint[2, 3] = viewpoint[2, 3] / 100

        # Transform viewpoint to world frame
        tf_part_to_world = np.linalg.inv(self.tf_part_to_world)
        viewpoint = np.dot(self.tf_part_to_world, viewpoint)

        # Calculate tool0 frame from camera frame
        # tf_camera_to_tool = np.linalg.inv(self.tf_camera_to_tool)
        # viewpoint = np.dot(self.tf_camera_to_tool, viewpoint)

        # Move to viewpoint
        self.ros_thread.move_to_pose(viewpoint, frame_id='world')

    def update_scene(self):

        rgb_image, annotated_rgb_image, depth_image, depth_intrinsic, illuminance_image, gphoto2_image, T = self.ros_thread.get_data()

        if (T == np.eye(4)).all() and self.viewpoint_dict is not None:
            selected_region_name = f"region_{self.selected_viewpoint}"
            selected_region = self.viewpoint_dict['regions'][selected_region_name]

            T = np.array(selected_region['viewpoint'])
            T[0, 3] = 0.01*T[0, 3]
            T[1, 3] = 0.01*T[1, 3]
            T[2, 3] = 0.01*T[2, 3]

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

        # Content

        r = self.window.content_rect

        # UPDATE LIGHT PANEL ####################################################

        if self.light_panel.get_is_open():

            light_map_cv2 = self.light_map.get_map_image()

            image_o3d = o3d.geometry.Image(light_map_cv2)
            self.light_map_image.update_image(image_o3d)

        # UPDATE STEREO PANEL ####################################################

        if self.stereo_camera_panel.get_is_open():

            stereo_tab = self.stereo_camera_tabs.selected_tab_index

            # RGB TAB ################################################
            if stereo_tab == MyGui.STEREO_RGB_TAB:
                self.rgb_image.update_image(annotated_rgb_image_o3d)

            # DEPTH TAB ##############################################

            elif stereo_tab == MyGui.STEREO_DEPTH_TAB:
                depth_image[depth_image > self.depth_trunc] = 0

                self.plotting_thread.update_depth_image(depth_image)
                depth_image_cv2 = self.plotting_thread.get_depth_image()

                # Replace black pixels with transparent pixels
                depth_image_o3d = o3d.geometry.Image(depth_image_cv2)
                self.depth_image.update_image(depth_image_o3d)

            # ILLUMINANCE TAB ########################################

            elif stereo_tab == MyGui.STEREO_ILLUMINANCE_TAB:
                self.illuminance_image.update_image(illuminance_image_o3d)

        tab_index = self.main_tabs.selected_tab_index

        # REMOVE GEOMETRY ##########################################################

        self.scene_widget.scene.remove_geometry(self.live_point_cloud_name)
        self.scene_widget.scene.remove_geometry("stereo_camera")
        self.scene_widget.scene.remove_geometry("main_camera")
        self.scene_widget.scene.remove_geometry("light_ring")

        # UPDATE SCENE TAB #########################################################

        if tab_index == MyGui.SCENE_TAB:

            # Show/hide and enable/disable UI elements
            self.monitor_image_panel.enabled = False
            self.monitor_image_panel.visible = False
            self.monitor_image_widget.enabled = False
            # self.right_panel.enabled = False
            # self.right_panel.visible = False
            self.main_tabs.selected_tab_index = MyGui.SCENE_TAB

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
            focal_distance_m_l = 0.001*self.focal_distance_mm - 0.001*self.dof/2
            focal_distance_m_h = 0.001*self.focal_distance_mm + 0.001*self.dof/2

            o = np.array([0, 0, 0])
            tl_l = [-fov_width_m/2, fov_height_m/2, focal_distance_m_l]
            tr_l = [fov_width_m/2, fov_height_m/2, focal_distance_m_l]
            bl_l = [-fov_width_m/2, -fov_height_m/2, focal_distance_m_l]
            br_l = [fov_width_m/2, -fov_height_m/2, focal_distance_m_l]
            tl_h = [-fov_width_m/2, fov_height_m/2, focal_distance_m_h]
            tr_h = [fov_width_m/2, fov_height_m/2, focal_distance_m_h]
            bl_h = [-fov_width_m/2, -fov_height_m/2, focal_distance_m_h]
            br_h = [fov_width_m/2, -fov_height_m/2, focal_distance_m_h]

            main_camera.points = o3d.utility.Vector3dVector(
                np.array([o, tl_l, tr_l, bl_l, br_l, tl_h, tr_h, bl_h, br_h]))
            main_camera.lines = o3d.utility.Vector2iVector(
                np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1],
                          [5, 6], [6, 8], [8, 7], [7, 5], [5, 1], [6, 2], [8, 4], [7, 3]]))
            main_camera.transform(T)
            main_camera.scale(100.0, center=np.array([0, 0, 0]))

            # self.scene_widget.scene.add_geometry(
            # "stereo_camera", stereo_camera, self.line_material)
            self.scene_widget.scene.add_geometry(
                "main_camera", main_camera, self.camera_line_material)

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

            if self.moving_to_viewpoint_flag:
                if not self.ros_thread.moving_to_viewpoint_flag:
                    selected_viewpoint_inspectable = self.ros_thread.last_move_successful
                    print(
                        f"Selected viewpoint inspectable: {selected_viewpoint_inspectable}")

                    self.moving_to_viewpoint_flag = False
                    self.t0 = time.time()

            elif self.running:
                t1 = time.time()
                if t1 - self.t0 >= 0.1:
                    self.viewpoint_slider.int_value = self.viewpoint_slider.int_value + 1 if self.viewpoint_slider.int_value < len(
                        self.viewpoint_dict['regions'].keys()) else 1
                    self.select_viewpoint(self.viewpoint_slider.int_value)
                    self.move_to_selected_viewpoint()
                    self.t0 = t1

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

                    viewpoint_dict_path = self.save_viewpoints()
                    self.load_viewpoints(viewpoint_dict_path)

        # UPDATE MONITOR TAB #########################################################

        elif tab_index == MyGui.MONITOR_TAB:

            # Show/hide and enable/disable UI elements

            self.right_panel.enabled = True
            self.right_panel.visible = True
            # self.monitor_image_panel.enabled = True
            self.monitor_image_panel.visible = True
            # self.scene_widget.scene.show_ground_plane(
            # False, o3d.visualization.rendering.Scene.GroundPlane.XY)
            self.main_tabs.selected_tab_index = MyGui.MONITOR_TAB

            # MONITOR TAB ############################################################

            width_u = self.monitor_image_panel.frame.width
            height_u = self.monitor_image_panel.frame.height
            width_px = gphoto2_image.shape[1]
            height_px = gphoto2_image.shape[0]
            scale_w = width_u/width_px
            scale_h = height_u/height_px
            scale = scale_w

            # crop the top and bottom of the image to fit the widget
            if height_px*scale_w > height_u:
                crop_u = (height_px*scale - height_u)
                crop_px = int(crop_u/scale/2)+1
                gphoto2_image = gphoto2_image[crop_px+1:height_px - crop_px, :]
                scale = scale_w
            # elif width_px*scale_h > width_u:
            #     crop_u = (width_px*scale - width_u)
            #     crop_px = int(crop_u/scale/2)
            #     print(f'crop_u: {crop_u}, crop_px: {crop_px}')
            #     gphoto2_image = gphoto2_image[:, crop_px:width_px - crop_px]
            #     scale = scale_h

            gphoto2_image = cv2.resize(
                gphoto2_image, (0, 0), fx=scale, fy=scale)

            # Draw ROI on image
            # roi_width = int(scale*self.roi_width)
            # roi_height = int(scale*self.roi_height)
            # roi_x = int((gphoto2_image.shape[1] - roi_width)/2)
            # roi_y = int((gphoto2_image.shape[0] - roi_height)/2)
            # gphoto2_image = cv2.rectangle(
            #     gphoto2_image, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (255, 255, 255), 2)
            # Draw a circle in the center of the image
            gphoto2_image = cv2.circle(
                gphoto2_image, (int(gphoto2_image.shape[1]/2), int(gphoto2_image.shape[0]/2)), 10, (255, 255, 255), 1)

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
                        gphoto2_image, p, g, (255, 255, 255), 5)
                # Draw circle at pan_pos
                gphoto2_image = cv2.circle(
                    gphoto2_image, p, 10, (255, 255, 255), -1)

            gphoto2_image_o3d = o3d.geometry.Image(gphoto2_image)

            self.monitor_image_widget.update_image(gphoto2_image_o3d)

        # METRIC PANEL ###########################################################

        if self.right_panel.get_is_open():

            focus_metric_data = self.ros_thread.focus_metric_dict['metrics']['sobel']['value']
            focus_metric_time = self.ros_thread.focus_metric_dict['metrics']['sobel']['time']
            focus_metric_image = self.ros_thread.focus_metric_dict['metrics']['sobel']['image']

            self.plotting_thread.update_focus_metric(
                focus_metric_time, focus_metric_data, focus_metric_image)
            focus_metric_plot_cv2 = self.plotting_thread.get_focus_metric_plot()
            focus_metric_image_cv2 = self.plotting_thread.get_focus_metric_image()

            focus_metric_plot_o3d = o3d.geometry.Image(focus_metric_plot_cv2)
            self.focus_metric_plot_image.update_image(focus_metric_plot_o3d)

            focus_metric_image_o3d = o3d.geometry.Image(focus_metric_image_cv2)
            self.focus_metric_image.update_image(focus_metric_image_o3d)

            # Update log
        self.log_list = self.ros_thread.read_log()
        # self.log_list.insert(0, "Log " + str(np.random.randint(1000)))
        # self.log_list = self.log_list[:1000]

        self.ros_log_text.set_items(self.log_list)
        self.ros_log_text.selected_index = 0

        this_draw_time = time.time()
        fps = 1.0 / (this_draw_time - self.last_draw_time)
        self.last_draw_time = this_draw_time
        # print(f'FPS: {fps}')

        return True

    def startThread(self):
        if self.update_delay >= 0:
            threading.Thread(target=self.update_thread).start()

    def update_thread(self):
        def do_update():
            return self.update_scene()

        while not self.is_done:
            time.sleep(self.update_delay)
            print("update_thread")
            with self.lock:
                if self.is_done:  # might have changed while sleeping.
                    break
                gui.Application.instance.post_to_main_thread(
                    self.window, self.update_scene)

    def on_main_window_closing(self):
        with self.lock:
            self.is_done = True

        self.pixels_to(self.num_pixels*[0])
        time.sleep(0.2)

        gui.Application.instance.quit()

        rclpy.shutdown()
        time.sleep(1)
        exit()

        return True  # False would cancel the close

    def on_main_window_tick_event(self):
        # print("tick")
        return self.update_scene()

    def _on_layout(self, layout_context):

        em = self.window.theme.font_size

        r = self.window.content_rect

        self.main_tabs.frame = r
        self.scene_widget.frame = r

        tab_frame_top = 1.5*em
        tab_frame_height = 4*em

        self.main_tabs.frame = gui.Rect(
            0, tab_frame_top, r.width, tab_frame_top + tab_frame_height)

        main_frame_top = 2*tab_frame_top + tab_frame_height

        self.scene_widget.frame = r

        # Viewpoints Panel

        width = self.viewpoint_generation_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        height = self.viewpoint_generation_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        top = self.footer_panel.frame.get_top() - 2*em - height
        if height < 4 * em:
            width = 7.5 * em

        self.viewpoint_generation_panel.frame = gui.Rect(
            0, top, width, height)

        # Light Control Panel

        top = self.main_tabs.frame.get_bottom() + 2*em

        width = self.light_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        width = 25 * em
        height = self.light_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height

        if not self.light_panel.get_is_open():
            width = 7.5 * em
        else:
            height = 40 * em

        self.light_panel.frame = gui.Rect(
            0, top, width, height)

        # Log Panel

        max_height = 10 * em
        height = min(self.log_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height, max_height)

        log_panel_width = r.width
        log_panel_height = height
        self.log_panel.frame = gui.Rect(
            0, r.height - log_panel_height + 1.5*em, log_panel_width, log_panel_height)

        # Action Panel

        action_panel_height = 3.5*em
        action_panel_width = log_panel_width

        top = self.log_panel.frame.get_top() - action_panel_height
        left = 0

        self.footer_panel.frame = gui.Rect(
            left, top, action_panel_width, action_panel_height)

        # Monitor Image Panel
        height = self.monitor_image_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        height = self.footer_panel.frame.get_top() - main_frame_top

        self.monitor_image_panel.frame = gui.Rect(
            0, main_frame_top, r.width, height)

        # Metric Panel

        # width = self.right_panel.calc_preferred_size(
        # layout_context, gui.Widget.Constraints()).width
        # Set width to 1/5 the width of the window
        width = r.width/5
        height = self.right_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        height = self.footer_panel.frame.get_top() - main_frame_top
        if not self.right_panel.get_is_open():
            width = 3 * em

        top = main_frame_top + 2*em
        top = main_frame_top
        left = r.width - width

        self.right_panel.frame = gui.Rect(left, top, width, height)

        # Part Frame Panel

        width = self.part_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        height = self.part_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height < 10 * em:
            width = 7.5 * em

        top = main_frame_top + 2*em
        left = self.right_panel.frame.get_left() - width - em

        self.part_frame_panel.frame = gui.Rect(left, top, width, height)

        # camera Frame Panel

        width = self.camera_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).width
        height = self.camera_frame_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints()).height
        if height < 10 * em:
            width = 7.5 * em

        top = self.part_frame_panel.frame.get_bottom() + 2*em
        left = self.right_panel.frame.get_left() - width - em

        self.camera_frame_panel.frame = gui.Rect(left, top, width, height)


def main(args=None):
    rclpy.init(args=args)
    print(args)

    gui.Application.instance.initialize()

    thread_delay = 0.1
    use_tick = -1

    dpcApp = MyGui(use_tick)
    dpcApp.startThread()

    gui.Application.instance.run()


if __name__ == '__main__':
    print("Open3D version:", o3d.__version__)
    main()
