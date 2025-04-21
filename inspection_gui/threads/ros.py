#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import os
import json
import cv2  # OpenCV library
import open3d as o3d
import numpy as np
import datetime
import threading
import asyncio
import time
import message_filters
from ultralytics import YOLO
import pytransform3d.rotations as pr
from scipy.spatial.transform import Rotation as R

import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, Joy
from geometry_msgs.msg import Twist, TwistStamped, Pose, PoseStamped
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import ListParameters, DescribeParameters, GetParameters, SetParameters

from inspection_gui.threads.tf2_message_filter import Tf2MessageFilter
from inspection_gui.focus_monitor import FocusMonitor
from inspection_gui.threads.lighting import LightMap
from inspection_msgs.msg import PixelStrip, FocusValue
from inspection_srvs.srv import CaptureImage, MoveToPose, SetFocusMetric
from std_msgs.msg import Float64, ColorRGBA, String
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController

kv = 0.6

OFF = 0
ON = 1

IDLE = 0
AUTOFOCUS = 1
SAVING_DATA = 2
MOVING = 3
RESETTING = 4
CAPTURING_IMAGE = 5
STARTING_SERVO = 6
STOPPING_SERVO = 7

DYNAMIC_AUTOFOCUS = 1
HILLCLIMB_AUTOFOCUS = 2
AUTOFOCUS_IN_PROGRESS = 0
AUTOFOCUS_SUCCESS = 1
AUTOFOCUS_FAILURE = 2

# DYNAMIC_AUTOFOCUS_SPEED = 0.01
# HILLCLIMB_AUTOFOCUS_SPEED = 0.1
AUTOFOCUS_SPEED = 0.25


class RosThread(Node):

    last_move_successful = False

    servo_state = ON
    state = IDLE
    state_flags = [False]*8

    focus_metric = 'sobel'
    focus_value_alpha = 0.5
    filtered_focus_value = 0.0

    # SERVO
    m = 5
    k_p = 0.02
    c_p = 45.0
    k_o = 0.1
    c_o = 0.1

    joy_axes = [0., 0., 1., 0., 0., 1., 0., 0.]
    joy_buttons = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    pan_pos = (0., 0.)
    pan_vel = (0., 0.)
    pan_vel_min = (0.001, 0.001)
    pan_vel_max = (1., 1.)
    pan_goal = (0., 0.)

    orbit_pos = (0., 0., 0.)
    orbit_vel = (0., 0., 0.)
    orbit_vel_min = (0.5, 0.5, 0.5)
    orbit_scaling = (1., 1., 1.)
    orbit_vel_max = (1., 1., 1.)
    orbit_goal = (0., 0., 0.)

    home_position = (0., 0., 0.)
    home_orientation = (0., 0., 0., 1.)

    position = (0., 0., 0.)
    orientation = (0., 0., 0., 1.)

    velocity = (0., 0., 0., 0., 0., 0.)

    zoom_pos = 0.0
    zoom_vel = 0.0
    zoom_vel_min = 0.001
    zoom_vel_max = 1.0
    zoom_goal = 0.0

    servo_twist = TwistStamped()
    servo_twist.header.frame_id = 'tool0'
    servo_twist_pub_timer_period = 0.1

    af_type = 'dynamic'
    af_distance = 0.04
    af_dFV_threshold = 50
    af_ddFV_threshold = -30

    data_path = '/root/Inspection/data/'
    last_data_path = '/root/Inspection/data/'

    last_process_time = time.time()
    macro_image_fps = 0
    focus_measurement_time = 0

    image_width = 640
    image_height = 480

    roi_width = 100
    roi_height = 100

    illuminance_resolution = 40

    # initialization method
    def __init__(self, stream_id=0):
        super().__init__('gui_node')
        self.start_measure()

        self.log = []

        self.bridge = CvBridge()

        yolov8 = YOLO('yolov8n-seg.pt')
        # self.yolov8_seg = yolov8
        # yolov8.export(format='openvino')
        # self.yolov8_seg = YOLO("yolov8n-seg_openvino_model/")
        self.yolov8_seg = yolov8

        self.stopped = True        # thread instantiation
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        self.frame_id = None
        self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

        self.T_wt = np.eye(4)

        self.annotated_rgb_image = np.zeros(
            (480, 640, 3), dtype=np.uint8)
        self.rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth_image = np.zeros((480, 640, 1), dtype=np.float32)

        # TF2 #########################################################################

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.get_tf_frames()
        self.staticTfBroadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # MACRO CAMERA ####################################################################

        self.camera_frame_tf = np.eye(4)

        macro_camera_cb_group = MutuallyExclusiveCallbackGroup()

        self.focus_monitor = FocusMonitor(
            0.5, 0.5, self.roi_width, self.roi_height, 'sobel')
        self.macro_image = np.zeros((480, 620, 3), dtype=np.uint8)
        self.display_image = np.zeros((480, 620, 3), dtype=np.uint8)
        self.cropped_image = np.zeros(
            (self.roi_height, self.roi_width, 3), dtype=np.uint8)

        image_topic = '/image_raw/compressed'
        image_sub = self.create_subscription(
            CompressedImage, image_topic, self.compressed_image_callback, 10, callback_group=macro_camera_cb_group)

        # TELEOP

        teleop_cb_group = MutuallyExclusiveCallbackGroup()
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10, callback_group=teleop_cb_group)

        # FOCUS #########################################################################

        # Generate self.autofocus_data_dict
        self.reset_autofocus_data()

        self.focus_state = IDLE
        focus_cb_group = MutuallyExclusiveCallbackGroup()

        def change_metric_callback(request, response):
            success = self.set_focus_metric(request.metric)
            print(f'Success: {success}')
            response.success = success
            return response

        def auto_focus_callback(request, response):
            self.reset_autofocus_data()
            self.state = DYNAMIC_AUTOFOCUS
            response.success = True
            return response

        self.change_metric_service = self.create_service(
            SetFocusMetric, '/set_focus_metric', change_metric_callback, callback_group=focus_cb_group)

        self.auto_focus_service = self.create_service(Trigger, '/auto_focus', auto_focus_callback,
                                                      callback_group=focus_cb_group)

        self.focus_pub = self.create_publisher(
            FocusValue, image_topic + '/focus_value', 10)

        # MACRO SETTINGS ######################################################################

        self.camera_params = {}

        self.get_logger().info('Connecting to camera1 node...')
        camera_node_name = 'camera1'
        param_names = []
        self.camera_node_list_parameters_cli = self.create_client(
            ListParameters, camera_node_name + '/list_parameters', callback_group=macro_camera_cb_group)
        if not self.camera_node_list_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('List parameters service not available, waiting again...')
        else:
            self.get_logger().info('Connected!')

            req = ListParameters.Request()
            self.get_logger().info('Sending list parameters request...')
            future = self.camera_node_list_parameters_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            self.get_logger().info('Got parameters.')
            for param_name in resp.result.names:
                param_names.append(param_name)

        self.camera_node_describe_parameters_cli = self.create_client(
            DescribeParameters, camera_node_name + '/describe_parameters', callback_group=macro_camera_cb_group)
        if not self.camera_node_describe_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Describe parameters service not available, waiting again...')
        else:
            req = DescribeParameters.Request()
            req.names = param_names
            self.get_logger().info('Sending describe parameters request...')
            future = self.camera_node_describe_parameters_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            for param in resp.descriptors:
                self.camera_params[param.name] = {}
                self.camera_params[param.name]['type'] = param.type
                self.camera_params[param.name]['description'] = param.description
                self.camera_params[param.name]['choices'] = param.additional_constraints.split(
                    '\n')
                self.camera_params[param.name]['read_only'] = param.read_only

        self.camera_node_get_parameters_cli = self.create_client(
            GetParameters, camera_node_name + '/get_parameters', callback_group=macro_camera_cb_group)
        if not self.camera_node_get_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get parameters service not available, waiting again...')
        else:
            req = GetParameters.Request()
            req.names = param_names
            self.get_logger().info('Sending get parameters request...')
            future = self.camera_node_get_parameters_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            for i in range(len(param_names)):
                if resp.values[i].type == rclpy.Parameter.Type.BOOL:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].bool_value
                elif resp.values[i].type == rclpy.Parameter.Type.BOOL_ARRAY:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].bool_array_value
                elif resp.values[i].type == rclpy.Parameter.Type.BYTE_ARRAY:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].byte_array_value
                elif resp.values[i].type == rclpy.Parameter.Type.DOUBLE:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].double_value
                elif resp.values[i].type == rclpy.Parameter.Type.DOUBLE_ARRAY:
                    self.camera_params[param_names[i]
                                       ['value']] = resp.values[i].double_array_value
                elif resp.values[i].type == 2:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].integer_value
                elif resp.values[i].type == rclpy.Parameter.Type.INTEGER_ARRAY:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].integer_array_value
                elif resp.values[i].type == 4:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].string_value
                elif resp.values[i].type == rclpy.Parameter.Type.STRING_ARRAY:
                    self.camera_params[param_names[i]
                                       ]['value'] = resp.values[i].string_array_value

        # Set Parameters Client
        self.set_camera_params_cli = self.create_client(
            SetParameters, 'camera1/set_parameters', callback_group=macro_camera_cb_group)
        if not self.set_camera_params_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set parameters service not available, waiting again...')

        # Capture Image Client
        self.capture_image_cli = self.create_client(
            CaptureImage, '/capture_image', callback_group=macro_camera_cb_group)
        if not self.capture_image_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('capture image service not available, waiting again...')
        else:
            self.get_logger().info('Connected to capture image service!')

        # STEREO CAMERA ##################################################################

        stereo_camera_cb_group = MutuallyExclusiveCallbackGroup()

        self.depth_intrinsic_sub = self.create_subscription(
            Image, "/camera/camera/depth/camera_info", self.depth_intrinsic_callback, 10, callback_group=stereo_camera_cb_group)
        # self.depth_image_sub = self.create_subscription(
        # Image, "/camera/camera/depth/image_rect_raw", self.depth_image_callback, 10)

        self.depth_trunc = 1.0

        depth_image_sub = message_filters.Subscriber(self,
                                                     Image, "/camera/camera/depth/image_rect_raw")
        rgb_image_sub = message_filters.Subscriber(self,
                                                   Image, "/camera/camera/color/image_rect_raw")
        # macro_image_sub = message_filters.Subscriber(self,
        #    Image, "/camera1/image_raw")
        ts = Tf2MessageFilter(self, [depth_image_sub, rgb_image_sub], 'part_frame',
                              'camera_depth_optical_frame', queue_size=1000)
        ts.registerCallback(self.depth_image_callback)

        # Inference

        inference_timer_period = 0.1
        # self.inference_timer = self.create_timer(
        # inference_timer_period, self.inference_timer_callback)

        # LIGHTS #########################################################################

        shape_mm = (200, 200)
        dpmm = 10
        light_locations = np.loadtxt(
            '/root/Inspection/Lights/led_positions.csv', delimiter=',')
        self.light_map = LightMap(shape_mm, dpmm, light_locations)
        self.light_map.start()

        lights_cb_group = MutuallyExclusiveCallbackGroup()

        self.capture_image_future = None
        self.pixel_strip_msg = PixelStrip()
        self.pixel_count = 148
        self.wb = [1.0, 1.0, 1.0]
        self.pixel_strip_msg.pixel_colors = self.pixel_count * \
            [ColorRGBA(r=0.0, g=0.0, b=0.0)]
        pixel_pub_timer_period = 0.05
        self.pixel_pub = self.create_publisher(PixelStrip, '/pixel_strip', 10)
        self.pixel_pub_timer = self.create_timer(
            pixel_pub_timer_period, self.pixel_pub_timer_callback, callback_group=lights_cb_group)

        self.get_logger().info('Connecting to light node...')
        light_node_name = '/pixel_strip'
        param_names = []
        self.light_node_get_parameters_cli = self.create_client(
            GetParameters, light_node_name + '/get_parameters', callback_group=lights_cb_group)
        if not self.light_node_get_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('get parameters service not available, waiting again...')
        else:
            req = GetParameters.Request()
            req.names = param_names
            self.get_logger().info('Connected to light node!')
            self.get_logger().info('Sending get parameters request...')
            future = self.light_node_get_parameters_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            for i in range(len(param_names)):
                if resp.values[i].type == rclpy.Parameter.Type.INTEGER:
                    self.get_logger().info(
                        f'{param_names[i]}: {resp.values[i].integer_value}')

        # SERVO #########################################################################

        servo_cb_group = MutuallyExclusiveCallbackGroup()

        self.controller_manager_cli = self.create_client(
            SwitchController, '/controller_manager/switch_controller', callback_group=servo_cb_group)
        if not self.controller_manager_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('controller manager service not available, waiting again...')
        else:
            self.get_logger().info('Connected to controller manager!')

        # Call /servo_node/start_servo service
        self.get_logger().info('Connecting to servo node...')
        self.stop_servo_cli = self.create_client(
            Trigger, '/servo_node/stop_servo', callback_group=servo_cb_group)
        if not self.stop_servo_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('start servo service not available, waiting again...')
        else:
            self.get_logger().info('Connected to servo node!')
            self.stop_servo()

        self.start_servo_cli = self.create_client(
            Trigger, '/servo_node/start_servo', callback_group=servo_cb_group)
        if not self.start_servo_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('start servo service not available, waiting again...')
        else:
            self.get_logger().info('Connected to servo node!')
            self.start_servo()

        self.start_servo_control()

        self.servo_twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.servo_twist_pub_timer = self.create_timer(
            self.servo_twist_pub_timer_period, self.servo_twist_pub_timer_callback, callback_group=servo_cb_group)

        light_ring = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.1, height=0.01)
        self.light_ring = o3d.geometry.LineSet.create_from_triangle_mesh(
            light_ring)
        self.camera = o3d.geometry.LineSet().create_camera_visualization(
            self.depth_intrinsic, extrinsic=np.eye(4))

        self.geom_pcd = self.generate_point_cloud()

        # MoveIt #########################################################################

        moveit_cb_group = MutuallyExclusiveCallbackGroup()

        self.target_pose_publisher = self.create_publisher(
            PoseStamped, 'move_to_pose_target', 10)
        self.move_to_pose_cli = self.create_client(
            MoveToPose, 'inspection/move_to_pose', callback_group=moveit_cb_group)
        if not self.move_to_pose_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('moveit path planning service not available, waiting again...')
        else:
            self.get_logger().info('Connected to moveit path planning service!')

        # PLOTTING ######################################################################
        self.plot_timer_period = 0.05
        self.plot_timer = self.create_timer(
            self.plot_timer_period, self.plot_timer_callback)

        # ILLUMINANCE ####################################################################
        self.illuminance_plot = np.zeros((480,
                                          620, 3), dtype=np.uint8)

        self.illuminance_timer_period = 0.1
        self.illuminance_timer = self.create_timer(
            self.illuminance_timer_period, self.illuminance_timer_callback, callback_group=lights_cb_group)

        # MAIN LOOP #####################################################################

        main_callback_group = MutuallyExclusiveCallbackGroup()

        main_loop_period = 0.01
        self.main_loop_timer = self.create_timer(
            main_loop_period, self.main_loop, callback_group=main_callback_group)

    def main_loop(self):
        # Check state
        if self.state == IDLE:
            self.velocity = self.get_teleop_velocity()
            if self.state_flags[SAVING_DATA]:
                self.state = SAVING_DATA
                self.save_focus_data()
                self.state_flags[SAVING_DATA] = False
                self.state = IDLE
        elif self.state == AUTOFOCUS:
            autofocus_status = AUTOFOCUS_IN_PROGRESS

            # Get velocity based on autofocus type
            if self.af_type == 'dynamic':
                self.velocity, autofocus_status = self.get_dynamic_autofocus_velocity()
            elif self.af_type == 'hillclimb':
                self.velocity, autofocus_status = self.get_hillclimb_autofocus_velocity()

            # If autofocus is successful, move to max focus position
            if autofocus_status == AUTOFOCUS_SUCCESS:
                self.get_logger().info('Autofocus successful!')
                if self.af_type == 'hillclimb':
                    tf_max = np.eye(4)
                    rotation = R.from_quat(
                        self.autofocus_data_dict['orientation_max'])
                    rotation_matrix = rotation.as_matrix()
                    tf_max[:3, :3] = rotation_matrix
                    tf_max[:3, 3] = self.autofocus_data_dict['position_max']

                    self.move_to_pose(tf_max, 'world')

                self.state_flags[SAVING_DATA] = True
            elif autofocus_status == AUTOFOCUS_FAILURE:
                self.get_logger().info('Autofocus failed!')
                self.reset_autofocus_data()

        elif self.state == STOPPING_SERVO:
            pass
        elif self.state == STARTING_SERVO:
            pass
        elif self.state == MOVING:
            pass
        elif self.state == RESETTING:
            pass
        elif self.state == CAPTURING_IMAGE:
            pass
        elif self.state == SAVING_DATA:
            pass

    def start_measure(self):
        self.t0 = time.time()

    def stop_measure(self):
        self.get_logger().info(f'Measurement time: {time.time() - self.t0}')

    def start_servo_control(self):
        self.state = STARTING_SERVO
        self.activate_forward_position_controller()

    def stop_servo_control(self):
        self.state = STOPPING_SERVO
        self.activate_joint_trajectory_controller()

    def activate_forward_position_controller(self):
        self.state = STARTING_SERVO
        req = SwitchController.Request()
        req.start_controllers = ['forward_position_controller']
        req.stop_controllers = ['joint_trajectory_controller']
        self.get_logger().info('Activating forward position controller...')
        future = self.controller_manager_cli.call_async(req)
        future.add_done_callback(
            self.activate_forward_position_controller_callback)

    def activate_forward_position_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate forward position controller response: %s' % resp.ok)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        if resp.ok:
            self.start_servo()

    def start_servo(self):
        # Call /servo_node/start_servo service
        req = Trigger.Request()
        self.get_logger().info('Sending start servo request...')
        future = self.start_servo_cli.call_async(req)
        future.add_done_callback(self.start_servo_callback)
        self.get_logger().info('Servo node started!')

    def start_servo_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Start servo response: %s' % resp.success)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        self.servo_state = ON
        self.state = IDLE

    def activate_joint_trajectory_controller(self):
        self.state = STOPPING_SERVO

        req = SwitchController.Request()
        req.start_controllers = ['joint_trajectory_controller']
        req.stop_controllers = ['forward_position_controller']
        self.get_logger().info('Activating joint trajectory controller...')

        future = self.controller_manager_cli.call_async(req)
        future.add_done_callback(
            self.activate_joint_trajectory_controller_callback)

    def activate_joint_trajectory_controller_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Activate joint trajectory controller response: %s' % resp.ok)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        if resp.ok:
            self.stop_servo()
        else:
            self.state = IDLE

    def stop_servo(self):
        # Call /servo_node/stop_servo service
        req = Trigger.Request()
        self.get_logger().info('Sending stop servo request...')

        future = self.stop_servo_cli.call_async(req)
        future.add_done_callback(self.stop_servo_callback)

        self.get_logger().info('Servo node stopped!')
        self.servo_state = OFF

    def stop_servo_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Stop servo response: %s' % resp.success)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        self.servo_state = OFF
        self.state = MOVING

    def align_z_axis_down(self, position, orientation_quat):
        # Convert the quaternion to a rotation matrix
        current_rotation = R.from_quat(orientation_quat)
        rotation_matrix = current_rotation.as_matrix()

        # Extract the current z-axis direction
        current_z_axis = rotation_matrix[:, 2]

        # Define the desired z-axis direction (pointing downwards in world coordinates)
        desired_z_axis = np.array([0, 0, -1])

        # Calculate the rotation needed to align the current z-axis with the desired z-axis
        v = np.cross(current_z_axis, desired_z_axis)
        s = np.linalg.norm(v)
        c = np.dot(current_z_axis, desired_z_axis)
        if s == 0:
            # The current z-axis is already aligned with the desired z-axis
            rotation_needed = R.from_quat([0, 0, 0, 1])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            rotation_matrix_needed = np.eye(
                3) + vx + vx @ vx * ((1 - c) / (s ** 2))
            rotation_needed = R.from_matrix(rotation_matrix_needed)

        # Apply the rotation to the current orientation quaternion
        new_rotation = rotation_needed * current_rotation

        return new_rotation.as_matrix()

    def joy_callback(self, msg):
        self.joy_buttons = msg.buttons
        if self.joy_buttons[-1] == 1:
            position = self.autofocus_data_dict['position'][-1]
            orientation = self.autofocus_data_dict['orientation'][-1]
            new_rotation = self.align_z_axis_down(
                position, orientation)
            new_tf = np.eye(4)
            new_tf[:3, :3] = new_rotation
            new_tf[0, 3] = position[0]
            new_tf[1, 3] = position[1]
            new_tf[2, 3] = position[2]
            self.move_to_pose(new_tf, 'world')
        elif self.joy_buttons[0] == 1:
            # Autofocus
            if self.state == IDLE:
                self.af_type = 'dynamic'
                self.start_autofocus()
        elif self.joy_buttons[1] == 1:
            self.capture_image(self.last_data_path + 'image.jpg')
        elif self.joy_buttons[2] == 1:
            # Autofocus
            if self.state == IDLE:
                self.af_type = 'hillclimb'
                self.start_autofocus()

        elif self.joy_buttons[6]:
            self.set_home_position()
        elif self.joy_buttons[8]:
            self.go_to_home_position()

        if self.joy_buttons[4] == 1:
            self.light_map.set_sigma(int(100*(msg.axes[5]+1)/2)+0.01)
            self.light_map.set_mu_x(-msg.axes[0]/2)
            self.light_map.set_mu_y(msg.axes[1]/2)
            # self.light_map.set_sigma(sigma)
            self.set_pixels()
        else:
            self.joy_axes = msg.axes

    def set_home_position(self):
        self.home_position = self.position
        self.home_orientation = self.orientation

    def go_to_home_position(self):
        tf = np.eye(4)
        tf[:3, 3] = self.home_position
        tf[:3, :3] = R.from_quat(self.home_orientation).as_matrix()
        self.move_to_pose(tf, 'world')

    def start_autofocus(self):
        self.state = RESETTING
        self.reset_autofocus_data()
        self.state = AUTOFOCUS

    def move_to_pose(self, tf, frame_id):
        self.stop_servo_control()
        while not self.state == MOVING:
            print(self.state)
            self.get_logger().info('Waiting for state to be MOVING...')
            time.sleep(0.1)

        # Turn homogeneous tf into position and quaternion
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = frame_id
        pose_stamped.pose.position.x = tf[0, 3]
        pose_stamped.pose.position.y = tf[1, 3]
        pose_stamped.pose.position.z = tf[2, 3]
        q = pr.quaternion_from_matrix(tf[:3, :3])
        pose_stamped.pose.orientation.x = q[1]
        pose_stamped.pose.orientation.y = q[2]
        pose_stamped.pose.orientation.z = q[3]
        pose_stamped.pose.orientation.w = q[0]
        self.target_pose_publisher.publish(pose_stamped)

        self.state = MOVING
        req = MoveToPose.Request()
        req.target_pose = pose_stamped
        self.get_logger().info('Sending move to pose request...')

        future = self.move_to_pose_cli.call_async(req)
        future.add_done_callback(self.move_to_pose_callback)

        self.get_logger().info('Move to pose request complete!')

    def move_to_pose_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Move to pose response: %s' % resp.done)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        self.start_servo_control()

    def pixels_to(self, rgb_list):
        pixel_colors = []
        for i in range(len(rgb_list)):
            color = ColorRGBA()
            color.r = self.wb[0]*float(rgb_list[i][0])
            color.g = self.wb[1]*float(rgb_list[i][1])
            color.b = self.wb[2]*float(rgb_list[i][2])
            pixel_colors.append(color)
        self.pixel_strip_msg.pixel_colors = pixel_colors

    def pixel_pub_timer_callback(self):
        self.pixel_pub.publish(self.pixel_strip_msg)

    def capture_image(self, file_path):
        self.state = CAPTURING_IMAGE
        self.get_logger().info(f'Capturing image to: {file_path}')
        req = CaptureImage.Request()
        req.file_path = file_path
        print(req.file_path)
        future = self.capture_image_cli.call_async(req)
        future.add_done_callback(self.capture_image_callback)

    def capture_image_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Capture image response: %s' % resp.done)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        self.state = IDLE

    def get_tf_frames(self):
        return self.tfBuffer.all_frames_as_yaml()

    def send_transform(self, T, parent_frame, child_frame):
        t = tf2_ros.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = T[0, 3]
        t.transform.translation.y = T[1, 3]
        t.transform.translation.z = T[2, 3]
        q = pr.quaternion_from_matrix(T[:3, :3])
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]
        t.transform.rotation.w = q[0]
        self.staticTfBroadcaster.sendTransform(t)

    def set_camera_param(self, name, value):
        self.camera_params[name]['value'] = value

        req = SetParameters.Request()

        type = self.camera_params[name]['type']
        parameter = Parameter()
        parameter.name = name

        if type == rclpy.Parameter.Type.BOOL:
            parameter.value.bool_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.BOOL_ARRAY:
            parameter.value.bool_array_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.BYTE_ARRAY:
            parameter.value.byte_array_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.DOUBLE:
            parameter.value.double_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.DOUBLE_ARRAY:
            parameter.value.double_array_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.INTEGER:
            parameter.value.integer_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.INTEGER_ARRAY:
            parameter.value.integer_array_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.STRING:
            parameter.value.string_value = self.camera_params[name]['value']
        elif type == rclpy.Parameter.Type.STRING_ARRAY:
            parameter.value.string_array_value = self.camera_params[name]['value']

        req.parameters = [parameter]

        self.get_logger().info(
            f'Sending set parameters request for {name} to {value}...')
        future = self.set_camera_params_cli.call_async(req)
        future.add_done_callback(self.set_camera_params_callback)

    def set_camera_params(self):

        req = SetParameters.Request()
        req.parameters = []

        # Loop through items in self.camera_params and append to request
        for name in self.camera_params:

            type = self.camera_params[name]['type']
            parameter = Parameter()
            parameter.name = name

            if type == rclpy.Parameter.Type.BOOL:
                parameter.value.bool_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.BOOL_ARRAY:
                parameter.value.bool_array_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.BYTE_ARRAY:
                parameter.value.byte_array_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.DOUBLE:
                parameter.value.double_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.DOUBLE_ARRAY:
                parameter.value.double_array_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.INTEGER:
                parameter.value.integer_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.INTEGER_ARRAY:
                parameter.value.integer_array_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.STRING:
                parameter.value.string_value = self.camera_params[name]['value']
            elif type == rclpy.Parameter.Type.STRING_ARRAY:
                parameter.value.string_array_value = self.camera_params[name]['value']

            req.parameters.append(parameter)

        self.get_logger().info('Sending set parameters request...')
        future = self.set_camera_params_cli.call_async(req)
        future.add_done_callback(self.set_camera_params_callback)

    def set_camera_params_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Parameters set!')
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

    def inference_timer_callback(self):
        original_size = self.rgb_image.shape

        img = self.rgb_image.copy()

        scale_percent = 100  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        rgb_image_resized = cv2.resize(
            img, dim, interpolation=cv2.INTER_AREA).astype(np.uint8)
        results = self.yolov8_seg(img.astype(np.uint8), verbose=False)
        # self.annotated_rgb_image = self.rgb_image
        self.annotated_rgb_image = results[0].plot().astype(np.uint8)

    def zoom(self, zoom_vel):
        self.zoom_goal = self.zoom_pos + zoom_vel

    def autofocus_on(self):
        self.state = DYNAMIC_AUTOFOCUS

    def autofocus_off(self):
        self.state = IDLE

    def servo_twist_pub_timer_callback(self):
        # Publish servo twist
        if self.state == MOVING:
            return

        self.servo_twist.twist.linear.x = self.velocity[0]
        self.servo_twist.twist.linear.y = self.velocity[1]
        self.servo_twist.twist.linear.z = self.velocity[2]
        self.servo_twist.twist.angular.x = self.velocity[3]
        self.servo_twist.twist.angular.y = self.velocity[4]
        self.servo_twist.twist.angular.z = self.velocity[5]

        msg_time = self.get_clock().now().to_msg()
        self.servo_twist.header.stamp = msg_time
        self.servo_twist_pub.publish(self.servo_twist)

        timestamp = msg_time.sec + msg_time.nanosec / 1e9

        # If data length is greater than max length, remove first element

        if self.state == DYNAMIC_AUTOFOCUS or self.state == HILLCLIMB_AUTOFOCUS:
            pass
        elif len(self.autofocus_data_dict['velocity_data']['time']) > self.autofocus_data_dict['buffer_size']:
            self.autofocus_data_dict['velocity_data']['time'].pop(0)
            self.autofocus_data_dict['velocity_data']['velocity'].pop(0)

        # Add time and velocity data to dictionary
        self.autofocus_data_dict['velocity_data']['time'].append(timestamp)
        self.autofocus_data_dict['velocity_data']['velocity'].append(
            self.velocity)

    def generate_point_cloud(self):
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame
        # self.t2.start()

    def start_async_loop(self):
        asyncio.set_event_loop()

    def update(self):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()

    def depth_intrinsic_callback(self, msg):
        # Convert ROS message to Open3D camera intrinsic
        self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            msg.width, msg.height, msg.K[0], msg.K[4], msg.K[2], msg.K[5])

    def depth_image_callback(self, dmap_msg, rgb_msg, tf_msg):
        depth_image = self.bridge.imgmsg_to_cv2(
            dmap_msg, desired_encoding="passthrough").astype(np.float32) / 1000.0
        rgb_image = self.bridge.imgmsg_to_cv2(
            rgb_msg, desired_encoding="passthrough").astype(np.uint8)
        # macro_image = self.bridge.imgmsg_to_cv2(
        # gphoto2_msg, desired_encoding="passthrough").astype(np.uint8)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV).astype(np.uint8)
        # Set all pixels in image above 1000 to 0
        depth_image_m = depth_image / 1.0
        # depth_image_m[depth_image_m > self.depth_trunc] = 0
        # Apply gaussian blur to depth image
        # depth_image_cm = cv2.GaussianBlur(
        # depth_image_cm, (7, 7), 0, 0, cv2.BORDER_DEFAULT)

        trans = tf_msg.transform.translation
        quat = tf_msg.transform.rotation
        R = o3d.geometry.get_rotation_matrix_from_quaternion(
            [quat.w, quat.x, quat.y, quat.z])

        # Combine trans and R into a 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = trans.x
        T[1, 3] = trans.y
        T[2, 3] = trans.z

        self.depth_image = depth_image_m
        self.rgb_image = rgb_image
        self.T_wt = T
        self.illuminance_image = hsv_image[:, :, 2]

    def set_focus_metric(self, name):
        self.save_focus_data()
        self.focus_metric = name
        return self.focus_monitor.set_metric(name)

    def get_teleop_velocity(self):
        # Get last velocity
        if len(self.autofocus_data_dict['velocity_data']['velocity']) > 0:
            v_last = self.autofocus_data_dict['velocity_data']['velocity'][-1]
        else:
            v_last = (0., 0., 0., 0., 0., 0.)
        # Round to 3 decimal places
        vx_curr = -self.joy_axes[0]
        vx_curr = 0.8 * vx_curr + 0.2 * v_last[0]
        vx_curr = vx_curr if abs(vx_curr) > self.pan_vel_min[0] else 0.0

        vy_curr = (-self.joy_axes[5]+1)/2 - (-self.joy_axes[2]+1)/2
        vy_curr = 0.8 * vy_curr + 0.2 * v_last[1]
        vy_curr = vy_curr if abs(vy_curr) > self.zoom_vel_min else 0.0

        vz_curr = -self.joy_axes[1]
        vz_curr = 0.8 * vz_curr + 0.2 * v_last[2]
        vz_curr = vz_curr if abs(vz_curr) > self.pan_vel_min[1] else 0.0

        wx_curr = self.orbit_scaling[0] * self.joy_axes[4]
        wx_curr = 0.8 * wx_curr + 0.2 * v_last[3]
        wx_curr = wx_curr if abs(wx_curr) > self.orbit_vel_min[0] else 0.0

        # wy_curr = -self.joy_axes[3]
        # wy_curr = 0.9 * wy_curr + 0.1 * v_last[4]
        # wy_curr = wy_curr if abs(wy_curr) > self.orbit_vel_min[1] else 0.0
        wy_curr = 0.0

        wz_curr = -self.orbit_scaling[2] * self.joy_axes[3]
        wz_curr = 0.8 * wz_curr + 0.2 * v_last[5]
        wz_curr = wz_curr if abs(wz_curr) > self.orbit_vel_min[2] else 0.0

        return (round(vx_curr, 3), -round(vy_curr, 3), round(vz_curr, 3), round(wx_curr, 3), round(wy_curr, 3), round(wz_curr, 3))

        # Do not publish if all twist values are zero
        px0 = self.pan_pos[0]
        gx = self.pan_goal[0]
        vx0 = self.pan_vel[0]
        ax = (self.k_p * (gx - px0) - self.c_p * vx0) / self.m
        vx1 = vx0 + ax * self.servo_twist_pub_timer_period
        if abs(vx1) < self.pan_vel_min[0]:
            vx1 = 0.0
        elif vx1 > 0.0:
            vx1 = min(round(vx1, 3), self.pan_vel_max[0])
        else:
            vx1 = max(round(vx1, 3), -self.pan_vel_max[0])
        px1 = px0 + vx1 * self.servo_twist_pub_timer_period

        py0 = self.pan_pos[1]
        gy = self.pan_goal[1]
        vy0 = self.pan_vel[1]
        ay = (self.k_p * (gy - py0) - self.c_p * vy0) / self.m
        vy1 = vy0 + ay * self.servo_twist_pub_timer_period
        if abs(vy1) < self.pan_vel_min[1]:
            vy1 = 0.0
        elif vy1 > 0.0:
            vy1 = min(round(vy1, 3), self.pan_vel_max[1])
        else:
            vy1 = max(round(vy1, 3), -self.pan_vel_max[1])
        py1 = py0 + vy1 * self.servo_twist_pub_timer_period

        py1 = round(py1, 3)

        pz0 = self.zoom_pos
        gz = self.zoom_goal
        vz0 = self.zoom_vel
        az = (self.k_p * (gz - pz0) - self.c_p * vz0) / self.m
        vz1 = vz0 + az * self.servo_twist_pub_timer_period
        if abs(vz1) < self.zoom_vel_min:
            vz1 = 0.0
        elif vz1 > 0.0:
            vz1 = min(round(vz1, 3), self.zoom_vel_max)
        else:
            vz1 = max(round(vz1, 3), -self.zoom_vel_max)
        pz1 = pz0 + vz1 * self.servo_twist_pub_timer_period
        # Round to 3 decimal places

        px1 = round(px1, 3)
        py1 = round(py1, 3)
        pz1 = round(pz1, 3)

        self.pan_pos = (px1, py1)
        self.pan_vel = (vx1, vy1)

        self.zoom_pos = pz1
        self.zoom_vel = vz1
        self.zoom_goal = pz1

        if abs(vx1) < 0.25:
            vx1 = 0.0
        if abs(vy1) < 0.25:
            vy1 = 0.0
        if abs(vz1) < 0.25:
            vz1 = 0.0

        return (-vx1, vz1, vy1)

    def get_hillclimb_autofocus_velocity(self):

        # Check current position against first position. If distance is greater than self.af_distance, return
        if len(self.autofocus_data_dict['position']) > 1:
            p0 = self.autofocus_data_dict['position'][0]
            p1 = self.autofocus_data_dict['position'][-1]
            distance = np.linalg.norm(np.array(p1) - np.array(p0))

            if distance > self.af_distance:
                return (0., 0., 0., 0., 0., 0.), AUTOFOCUS_SUCCESS

        speed = AUTOFOCUS_SPEED

        return (0., -speed, 0., 0., 0., 0.), AUTOFOCUS_IN_PROGRESS

    def get_dynamic_autofocus_velocity(self):

        # Check current position against first position. If distance is greater than self.af_distance, return
        if len(self.autofocus_data_dict['position']) > 1:
            p0 = self.autofocus_data_dict['position'][0]
            p1 = self.autofocus_data_dict['position'][-1]
            distance = np.linalg.norm(np.array(p1) - np.array(p0))

            if abs(distance) > self.af_distance:
                return (0., 0., 0., 0., 0., 0.), AUTOFOCUS_FAILURE

        speed = AUTOFOCUS_SPEED

        if len(self.autofocus_data_dict['time']) == 0:
            ratio = 0
            dFV = 0
            ddFV = 0
            smooth_ddFV = 0
        elif len(self.autofocus_data_dict['time']) == 1:
            ratio = 0
            dFV = self.autofocus_data_dict['focus_value_dema'][-1]
            ddFV = 0
            smooth_ddFV = 0
        elif len(self.autofocus_data_dict['time']) == 2:
            ratio = self.autofocus_data_dict['focus_value_dema'][-1] / \
                self.autofocus_data_dict['focus_value_dema'][-2]
            dFV = self.autofocus_data_dict['focus_value_dema'][-1] - \
                self.autofocus_data_dict['focus_value_dema'][-2]
            ddFV = 0
            smooth_ddFV = 0
        else:
            ratio = self.autofocus_data_dict['focus_value_dema'][-1] / \
                self.autofocus_data_dict['focus_value_dema'][-2]
            dFV = self.autofocus_data_dict['focus_value_dema'][-1] - \
                self.autofocus_data_dict['focus_value_dema'][-2]
            try:
                ddFV = self.autofocus_data_dict['dFV'][-1] - \
                    self.autofocus_data_dict['dFV'][-2]
            except:
                print(self.autofocus_data_dict)

            K_smooth = 2 / (3 + 1)
            smooth_ddFV = (
                K_smooth * (ddFV - self.autofocus_data_dict['smooth_ddFV'][-1])) + self.autofocus_data_dict['smooth_ddFV'][-1]

            if smooth_ddFV < self.af_ddFV_threshold:
                print(f'dFV: {dFV}')
                print(f'ddFV: {smooth_ddFV}')

            # Calculate speed

            if abs(dFV) < self.af_dFV_threshold and smooth_ddFV < self.af_ddFV_threshold:
                speed = 0
                self.get_logger().info('Completed autofocus')
                self.state = IDLE
                return (0., 0., 0., 0., 0., 0.), AUTOFOCUS_SUCCESS
            if smooth_ddFV < -0.1 and dFV > 0:
                speed = kv/ratio
            elif smooth_ddFV < -0.1 and dFV > 0:
                speed = -kv/ratio
            elif smooth_ddFV > 0.1 and dFV < 0:
                speed = -kv*(ratio-0.5)
            else:
                speed = kv*(ratio-0.5)

            # Bound speed between -1 and 1
            speed = min(max(speed, -1.), 1.)

        # Update data dictionary
        self.autofocus_data_dict['ratio'].append(ratio)
        self.autofocus_data_dict['dFV'].append(dFV)
        self.autofocus_data_dict['ddFV'].append(ddFV)
        self.autofocus_data_dict['smooth_ddFV'].append(smooth_ddFV)

        return (0., -speed, 0., 0., 0., 0.), AUTOFOCUS_IN_PROGRESS

    def set_material(self, material):
        self.autofocus_data_dict['material'] = material

    def reset_autofocus_data(self):

        # Create a blank image
        height, width = 400, 800
        image = np.ones((height, width, 3), dtype=np.uint8) * 0

        # Define the bounding box
        bbox_top_left = (50, 50)
        bbox_bottom_right = (750, 350)
        cv2.rectangle(image, bbox_top_left,
                      bbox_bottom_right, (255, 255, 255), 2)

        velocity_plot = image.copy()

        margin = 10
        cv2.rectangle(image, (50+margin, 50+margin),
                      (300+margin, 150+margin), (0, 0, 0), -1)
        cv2.rectangle(image, (50+margin, 50+margin),
                      (300+margin, 150+margin), (255, 255, 255), 2)

        cv2.rectangle(velocity_plot, (50+margin, 50+margin),
                      (195+margin, 195+margin), (0, 0, 0), -1)
        cv2.rectangle(velocity_plot, (50+margin, 50+margin),
                      (195+margin, 195+margin), (255, 255, 255), 2)

        self.fv_ratio_plot = image.copy()

        self.autofocus_data_dict = {
            'material': None,
            'metric': self.focus_metric,
            'af_type': self.af_type,
            'roi_width': self.roi_width,
            'roi_height': self.roi_height,
            'buffer_size': 200,
            'focus_value_max': 0.1,
            'position_max': [0., 0., 0.],
            'orientation_max': [0., 0., 0., 1.],
            'time': [],
            'focus_value': [],
            'focus_value_ema': [],
            'focus_value_ema2': [],
            'focus_value_dema': [],
            'dFV': [],
            'ddFV': [],
            'smooth_ddFV': [],
            'ratio': [],
            'image': [],
            'focus_image': [],
            'position': [],
            'orientation': [],
            'velocity_data': {'time': [], 'velocity': []},
            'focus_value_plot': image,
            'velocity_plot': velocity_plot,
            'illuminance_map': np.zeros((480, 620), dtype=np.uint8),
        }

        self.state = IDLE

    def plot_timer_callback(self):
        if self.state == RESETTING:
            return
        elif len(self.autofocus_data_dict['focus_value']) <= 1:
            return

        focus_value_plot = self.plot_focus_metrics()
        velocity_plot = self.plot_twist_velocity()
        self.illuminance_plot = self.plot_illuminance()
        self.fv_ratio_plot = self.plot_fv_ratio()

        self.autofocus_data_dict['focus_value_plot'] = focus_value_plot
        self.autofocus_data_dict['velocity_plot'] = velocity_plot

    def illuminance_timer_callback(self):
        if self.state == RESETTING:
            return
        elif len(self.autofocus_data_dict['focus_value']) <= 1:
            return

        # Conver tthe macro image to LAB colorspace
        lab = cv2.cvtColor(self.macro_image, cv2.COLOR_BGR2LAB)
        # Extract the L channel
        l = lab[:, :, 0]

        height, width = l.shape

        # Apply a gaussian blur to the value channel
        v = cv2.GaussianBlur(l, (5, 5), 0, 0, cv2.BORDER_DEFAULT)

        # Downsample the hsv image
        illuminance_map = cv2.resize(
            l, (width//self.illuminance_resolution, height//self.illuminance_resolution), interpolation=cv2.INTER_AREA)

        self.autofocus_data_dict['illuminance_map'] = illuminance_map

    def plot_focus_metrics(self):
        # Plot focus metric data
        fv_data = self.autofocus_data_dict['focus_value']
        fv_dema_data = self.autofocus_data_dict['focus_value_dema']

        # If data is shorter than the buffer size, pad with zeros
        if len(fv_data) < self.autofocus_data_dict['buffer_size']:
            fv_data = [0] * \
                (self.autofocus_data_dict['buffer_size'] -
                 len(fv_data)) + fv_data
            fv_dema_data = [0] * \
                (self.autofocus_data_dict['buffer_size'] -
                 len(fv_dema_data)) + fv_dema_data
        # If data is longer than buffer size, truncate to last buffer_size elements
        elif len(fv_data) > self.autofocus_data_dict['buffer_size']:
            fv_data = fv_data[-self.autofocus_data_dict['buffer_size']:]
            fv_dema_data = fv_dema_data[-self.autofocus_data_dict['buffer_size']:]

        # Current focus value
        focus_value_curr = fv_data[-1]

        # Set the maximum value of the plot
        max_curr = self.autofocus_data_dict['focus_value_max']
        if max_curr == 0:
            max_curr = 1
        max_curr = max(max_curr, max(fv_dema_data))

        # Scale data to fit within the bounding box
        fv_data = [100 * x / max_curr
                   for x in fv_data]
        fv_dema_data = [100 * x / max_curr
                        for x in fv_dema_data]

        # Create a blank image
        height, width = 400, 800
        image = np.ones((height, width, 3), dtype=np.uint8) * 0

        # Define the bounding box
        bbox_top_left = (50, 50)
        bbox_bottom_right = (750, 350)
        cv2.rectangle(image, bbox_top_left,
                      bbox_bottom_right, (255, 255, 255), 2)

        # Plot the data as vertical bars
        bar_width = (bbox_bottom_right[0] - bbox_top_left[0]) // len(fv_data)
        for i in range(len(fv_data)):
            fv_value = fv_data[i]
            fv_dema_value = fv_dema_data[i]
            x1 = bbox_top_left[0] + i * bar_width
            y1 = bbox_bottom_right[1] - \
                int((bbox_bottom_right[1] -
                     bbox_top_left[1]) * (fv_value / 100))
            x2 = x1 + bar_width - 1
            y2 = bbox_bottom_right[1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
            y1 = bbox_bottom_right[1] - \
                int((bbox_bottom_right[1] -
                     bbox_top_left[1]) * (fv_dema_value / 100))
            y2 = y1 + 4*(x1 - x2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), -1)

        # Draw a box with white border and black transparent fill in the upper left corner of the image
        margin = 10
        cv2.rectangle(image, (50+margin, 50+margin),
                      (300+margin, 150+margin), (0, 0, 0), -1)
        cv2.rectangle(image, (50+margin, 50+margin),
                      (300+margin, 150+margin), (255, 255, 255), 2)
        # Add text to the box. First line "FV: " followed by the last focus value
        cv2.putText(image, 'FV: ' + str(int(focus_value_curr)),
                    (60+margin, 90+margin), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Add text to the box. Second line "FV MAX: " followed by the max focus value
        cv2.putText(image, 'FV MAX: ' + str(int(self.autofocus_data_dict['focus_value_max'])),
                    (60+margin, 130+margin), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def plot_fv_ratio(self):
        # Plot focus metric data
        data = self.autofocus_data_dict['ratio']

        # If data is shorter than the buffer size, pad with zeros
        if len(data) < self.autofocus_data_dict['buffer_size']:
            data = [0] * \
                (self.autofocus_data_dict['buffer_size'] - len(data)) + data
        # If data is longer than buffer size, truncate to last buffer_size elements
        elif len(data) > self.autofocus_data_dict['buffer_size']:
            data = data[-self.autofocus_data_dict['buffer_size']:]

        # Create a blank image
        height, width = 400, 800
        image = np.ones((height, width, 3), dtype=np.uint8) * 0

        # Define the bounding box
        bbox_top_left = (50, 50)
        bbox_bottom_right = (750, 350)
        cv2.rectangle(image, bbox_top_left,
                      bbox_bottom_right, (255, 255, 255), 2)

        # Plot the data as vertical bars
        bar_width = (bbox_bottom_right[0] - bbox_top_left[0]) // len(data)
        for i, value in enumerate(data):
            x1 = bbox_top_left[0] + i * bar_width
            y1 = bbox_bottom_right[1] - \
                int((bbox_bottom_right[1] -
                     bbox_top_left[1]) * (value))
            x2 = x1 + bar_width - 1
            y2 = bbox_bottom_right[1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        return image

    def plot_twist_velocity(self):
        # Plot twist velocity data for pan and zoom
        data = self.autofocus_data_dict['velocity_data']['velocity']

        # If data is shorter than the buffer size, pad beginning with zeros
        if len(data) < self.autofocus_data_dict['buffer_size']:
            data = [(0., 0., 0.)] * \
                (self.autofocus_data_dict['buffer_size'] - len(data)) + data
        # If data is longer than buffer size, truncate to last buffer_size elements
        elif len(data) > self.autofocus_data_dict['buffer_size']:
            data = data[-self.autofocus_data_dict['buffer_size']:]

        # Create a blank image
        height, width = 400, 800
        image = np.ones((height, width, 3), dtype=np.uint8) * 0

        # Define the bounding box
        margins = 50
        bbox_height = 300
        bbox_width = 700
        bbox_top_left = (margins, margins)
        bbox_bottom_right = (margins+bbox_width, margins+bbox_height)

        # Plot zoom velocity data as vertical bars. 0 is in the middle of the bounding box
        bar_width = (bbox_bottom_right[0] - bbox_top_left[0]) // len(data)
        for i, v in enumerate(data):
            vx = (bbox_height/2)*v[0]/self.pan_vel_max[0]
            vy = (bbox_height/2)*v[1]/self.zoom_vel_max
            vz = (bbox_height/2)*v[2]/self.pan_vel_max[1]

            x1 = bbox_top_left[0] + i * bar_width
            y1 = bbox_bottom_right[1] - int(vx) - int(bbox_height/2)
            x2 = x1 + bar_width - 1
            y2 = bbox_bottom_right[1] - int(bbox_height/2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), -1)

            x1 = bbox_top_left[0] + i * bar_width
            y1 = bbox_bottom_right[1] - int(vy) - int(bbox_height/2)
            x2 = x1 + bar_width - 1
            y2 = bbox_bottom_right[1] - int(bbox_height/2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)

            x1 = bbox_top_left[0] + i * bar_width
            y1 = bbox_bottom_right[1] - int(vz) - int(bbox_height/2)
            x2 = x1 + bar_width - 1
            y2 = bbox_bottom_right[1] - int(bbox_height/2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)

        # Draw horizontal line in the middle of the bounding box
        cv2.line(image, (bbox_top_left[0], (bbox_top_left[1] + bbox_bottom_right[1]) // 2),
                 (bbox_bottom_right[0], (bbox_top_left[1] + bbox_bottom_right[1]) // 2), (55, 55, 55), 2)

        # Draw a box with white border and black transparent fill in the upper left corner of the image
        margin = 10
        cv2.rectangle(image, (50+margin, 50+margin),
                      (195+margin, 195+margin), (0, 0, 0), -1)
        cv2.rectangle(image, (50+margin, 50+margin),
                      (195+margin, 195+margin), (255, 255, 255), 2)
        # Add text to the box. First line "X: " followed by the  x velocity
        cv2.putText(image, 'X: ' + str(round(self.velocity[0], 2)),
                    (60+margin, 90+margin), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
        # Add text to the box. Second line "Y: " followed by the y velocity
        cv2.putText(image, 'Y: ' + str(round(self.velocity[1], 2)),
                    (60+margin, 130+margin), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
        # Add text to the box. Third line "Z: " followed by the z velocity
        cv2.putText(image, 'Z: ' + str(round(self.velocity[2], 2)),
                    (60+margin, 170+margin), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw bounding box last so it is on top of the bars
        cv2.rectangle(image, bbox_top_left,
                      bbox_bottom_right, (255, 255, 255), 2)

        return image

    def set_intensity(self, intensity):
        self.light_map.set_intensity(intensity)
        self.set_pixels()

    def set_mu_x(self, x):
        self.light_map.set_mu_x(x)
        self.set_pixels()

    def set_mu_y(self, y):
        self.light_map.set_mu_y(y)
        self.set_pixels()

    def set_sigma(self, sigma):
        self.light_map.set_sigma(sigma)
        self.set_pixels()

    def set_pixels(self):
        # Make every tenth pixel white
        pixel_values = self.light_map.get_pixel_values()
        pixel_colors = []
        for value in pixel_values:
            pixel_colors.append((value, value, value))
        self.pixels_to(pixel_colors)

    def plot_illuminance(self):

        # Subtract the mean value from the illuminance map.
        illuminance_map = self.autofocus_data_dict['illuminance_map']
        # illuminance_map = illuminance_map - np.mean(illuminance_map)

        # Make sure the illuminance map is CV_8UC1
        # illuminance_map = cv2.normalize(
        # illuminance_map, None, 0, 255, cv2.NORM_MINMAX)
        illuminance_map = illuminance_map.astype(np.uint8)

        # Positive values are green, negative values are red
        illuminance_plot = cv2.applyColorMap(
            illuminance_map, cv2.COLORMAP_TWILIGHT)

        # Convert from BGR to RGB
        illuminance_plot = cv2.cvtColor(illuminance_plot, cv2.COLOR_BGR2RGB)

        # Upscale the illuminance map to the same size as the focus image
        illuminance_plot = cv2.resize(
            illuminance_plot, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

        # Draw ROI rectangle
        cx = self.focus_monitor.cx
        cy = self.focus_monitor.cy
        w = self.focus_monitor.w
        h = self.focus_monitor.h
        x0 = int(cx*self.image_width - w/2)
        y0 = int(cy*self.image_height - h/2)
        x1 = int(cx*self.image_width + w/2)
        y1 = int(cy*self.image_height + h/2)
        cv2.rectangle(illuminance_plot, (x0, y0), (x1, y1), (255, 255, 255), 2)

        return illuminance_plot

    def save_focus_data(self):

        # Copy the dictionary to a new dictionary
        autofocus_data_dict = self.autofocus_data_dict.copy()

        # Get current time to use as name for the folder
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = self.data_path + dt_string + '/'
        os.makedirs(file_path, exist_ok=True)
        self.last_data_path = file_path

        # Remove image and focus_image from dictionary
        images = autofocus_data_dict.pop('image')
        focus_images = autofocus_data_dict.pop('focus_image')
        focus_value_plot = autofocus_data_dict.pop('focus_value_plot')
        velocity_plot = autofocus_data_dict.pop('velocity_plot')
        illuminance_map = autofocus_data_dict.pop('illuminance_map')

        # Save the dictionary to a file
        with open(file_path + 'data.json', 'w') as f:
            json.dump(autofocus_data_dict, f)
        # Restore image and focus_image to dictionary
        # Save images to video
        out = cv2.VideoWriter(file_path + 'video.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (images[0].shape[1], images[0].shape[0]))
        for i in range(len(images)):
            out.write(images[i])
        out.release()

    def compressed_image_callback(self, msg):
        if self.state == RESETTING:
            return
        elif self.state == SAVING_DATA:
            return
        elif self.state == CAPTURING_IMAGE:
            return

        self.start_measure()

        msg_time = msg.header.stamp

        macro_image = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="rgb8").astype(np.uint8)

        height, width, _ = macro_image.shape
        self.image_height = height
        self.image_width = width

        cx = self.focus_monitor.cx
        cy = self.focus_monitor.cy
        w = self.focus_monitor.w
        h = self.focus_monitor.h
        x0 = int(cx*width - w/2)
        y0 = int(cy*height - h/2)
        x1 = int(cx*width + w/2)
        y1 = int(cy*height + h/2)

        t0 = time.time()

        focus_value, focus_image, cropped_image = self.focus_monitor.measure_focus(
            macro_image)

        self.focus_measurement_time = time.time() - t0

        self.filtered_focus_value = self.focus_value_alpha * focus_value + \
            (1 - self.focus_value_alpha) * self.filtered_focus_value

        # Publish focus value
        # focus_msg = FocusValue(
        #     header=msg.header, metric=self.focus_monitor.metric, data=self.filtered_focus_value, raw_data=focus_value)
        # self.focus_pub.publish(focus_msg)

        position = (0., 0., 0.)
        orientation = (0., 0., 0., 1.)
        velocity = (0., 0., 0., 0., 0., 0.)

        # Get the transform from world to tool0
        try:
            # Attempt to get the transform at the exact requested time
            tf_wt = self.tfBuffer.lookup_transform(
                'world', 'tool0', msg_time, rclpy.time.Duration(seconds=0.1))
            position = (tf_wt.transform.translation.x,
                        tf_wt.transform.translation.y, tf_wt.transform.translation.z)
            orientation = (tf_wt.transform.rotation.x, tf_wt.transform.rotation.y,
                           tf_wt.transform.rotation.z, tf_wt.transform.rotation.w)

            # Combine position and R into a 4x4 transformation matrix
            T = np.eye(4)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(
                [orientation[3], orientation[0], orientation[1], orientation[2]])
            T[:3, :3] = R
            T[0, 3] = position[0]
            T[1, 3] = position[1]
            T[2, 3] = position[2]

            self.position = position
            self.orientation = orientation
            self.T_wt = T

        except tf2_ros.TransformException as ex:
            # Fallback to the latest available transform within a 1-second duration
            print(ex)

        # Check max focus value
        if focus_value > self.autofocus_data_dict['focus_value_max']:
            self.autofocus_data_dict['focus_value_max'] = focus_value
            self.autofocus_data_dict['position_max'] = position
            self.autofocus_data_dict['orientation_max'] = orientation

        # Convert Time msg to float value
        self.timestamp = msg_time.sec + msg_time.nanosec / 1e9

        # Filter focus value
        # Calculate the EMA
        N_ema = 20  # Doesn't work consistently with 15
        if len(self.autofocus_data_dict['time']) <= 1:
            focus_value_ema = focus_value
            focus_value_ema2 = focus_value
            focus_value_dema = focus_value
            previous_focus_value_dema = focus_value
        else:
            focus_value_ema = self.autofocus_data_dict['focus_value_ema'][-1]
            focus_value_ema2 = self.autofocus_data_dict['focus_value_ema2'][-1]
            focus_value_dema = self.autofocus_data_dict['focus_value_dema'][-1]
            previous_focus_value_dema = self.autofocus_data_dict['focus_value_dema'][-1]

        K = 2 / (N_ema + 1)  # EMA smoothing factor for the last 15 periods

        focus_value_ema = K * (focus_value - focus_value_ema) + focus_value_ema
        focus_value_ema2 = K * \
            (focus_value_ema - focus_value_ema2) + focus_value_ema2
        focus_value_dema = max(0.1, 2 * focus_value_ema - focus_value_ema2)

        # Update autofocus data dictionary

        if self.state == DYNAMIC_AUTOFOCUS or self.state == HILLCLIMB_AUTOFOCUS:
            # self.get_dynamic_autofocus_velocity(focus_msg)
            pass
        else:
            if len(self.autofocus_data_dict['time']) > self.autofocus_data_dict['buffer_size']:
                self.autofocus_data_dict['time'].pop(0)
                self.autofocus_data_dict['focus_value_ema'].pop(0)
                self.autofocus_data_dict['focus_value_ema2'].pop(0)
                self.autofocus_data_dict['focus_value'].pop(0)
                self.autofocus_data_dict['position'].pop(0)
                self.autofocus_data_dict['orientation'].pop(0)
                self.autofocus_data_dict['image'].pop(0)
                self.autofocus_data_dict['focus_image'].pop(0)

        self.autofocus_data_dict['time'].append(self.timestamp)
        self.autofocus_data_dict['focus_value_ema'].append(focus_value_ema)
        self.autofocus_data_dict['focus_value_ema2'].append(focus_value_ema2)
        self.autofocus_data_dict['focus_value_dema'].append(focus_value_dema)
        self.autofocus_data_dict['focus_value'].append(focus_value)
        self.autofocus_data_dict['image'].append(cropped_image)
        self.autofocus_data_dict['focus_image'].append(focus_image)
        self.autofocus_data_dict['position'].append(position)
        self.autofocus_data_dict['orientation'].append(orientation)

        this_process_time = time.time()
        macro_image_fps = 1 / (this_process_time - self.last_process_time)
        self.macro_image_fps = 0.5 * macro_image_fps + \
            0.5 * self.macro_image_fps
        self.macro_image_fps = round(self.macro_image_fps, 2)
        self.last_process_time = this_process_time

        display_image = macro_image.copy()
        cv2.putText(display_image, 'FPS: ' + str(self.macro_image_fps),
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # draw the ROI rectangle
        cv2.rectangle(display_image, (x0, y0), (x1, y1), (255, 255, 255), 2)
        self.display_image = display_image

        self.macro_image = macro_image
        self.cropped_image = cropped_image

    def trigger_focus_experiment(self):
        future = self.focus_experiment_cli.call_async(
            self.focus_experiment_trigger)
        future.add_done_callback(self.focus_experiment_callback)

    def focus_experiment_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Focus experiment response: %s' % resp.done)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

    def get_data(self):
        light_map = self.light_map.get_map_image()

        # Look up transform from part_frame to tool0
        try:
            tf_pc = self.tfBuffer.lookup_transform(
                'part_frame', 'camera_link', rclpy.time.Time(seconds=self.timestamp))
            position = (tf_pc.transform.translation.x,
                        tf_pc.transform.translation.y, tf_pc.transform.translation.z)
            orientation = (tf_pc.transform.rotation.x, tf_pc.transform.rotation.y,
                           tf_pc.transform.rotation.z, tf_pc.transform.rotation.w)
            # Combine position and R into a 4x4 transformation matrix
            T = np.eye(4)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(
                [orientation[3], orientation[0], orientation[1], orientation[2]])
            T[:3, :3] = R
            T[0, 3] = position[0]
            T[1, 3] = position[1]
            T[2, 3] = position[2]
        except:
            print('Could not get transform from part_frame to tool0')
            T = np.eye(4)

        return self.rgb_image, self.annotated_rgb_image, self.depth_image, self.depth_intrinsic, self.illuminance_plot, self.display_image, T, light_map

    def read_rgb_image(self):
        return self.annotated_rgb_image

    def read_depth_image(self):
        return self.depth_image.copy()

    def read_point_cloud(self):
        return self.geom_pcd

    def read_camera(self):
        return self.camera

    def read_camera_params(self):
        return self.camera_params

    def read_light_ring(self):
        return self.light_ring

    def read_log(self):
        return self.log

    # method to stop reading frames
    def stop(self):
        print("Stopping ROS thread...")
        self.stopped = True
        self.t.join()
        print("ROS thread stopped")
        # self.t2.join()
