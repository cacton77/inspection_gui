#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import cv2  # OpenCV library
import open3d as o3d
import numpy as np
import threading
import asyncio
import time
import message_filters
from ultralytics import YOLO
import pytransform3d.rotations as pr

import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import ListParameters, DescribeParameters, GetParameters, SetParameters

from inspection_gui.threads.tf2_message_filter import Tf2MessageFilter
from inspection_gui.focus_monitor import FocusMonitor
from inspection_msgs.msg import PixelStrip, FocusValue
from inspection_srvs.srv import CaptureImage
from inspection_srvs.srv import MoveToPose
from std_msgs.msg import Float64, ColorRGBA
from std_srvs.srv import Trigger


class RosThread(Node):

    moving_to_viewpoint_flag = False
    last_move_successful = False

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

        self.T = np.eye(4)

        self.annotated_rgb_image = np.zeros(
            (480, 640, 3), dtype=np.uint8)
        self.rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth_image = np.zeros((480, 640, 1), dtype=np.float32)
        self.illuminance_image = np.zeros((480, 640, 1), dtype=np.uint8)

        # MACRO CAMERA ####################################################################

        macro_camera_cb_group = MutuallyExclusiveCallbackGroup()

        self.focus_monitor = FocusMonitor(0.5, 0.5, 300, 300, 'sobel')
        self.gphoto2_image = np.zeros((576, 1024, 3), dtype=np.uint8)
        self.focus_metric_dict = {}
        self.focus_metric_dict['buffer_size'] = 100
        self.focus_metric_dict['metrics'] = {}
        self.focus_metric_dict['metrics']['sobel'] = {}
        self.focus_metric_dict['metrics']['sobel']['filtered_value'] = []
        self.focus_metric_dict['metrics']['sobel']['raw_value'] = []
        self.focus_metric_dict['metrics']['sobel']['time'] = []
        self.focus_metric_dict['metrics']['sobel']['image'] = np.zeros(
            (200, 200))

        image_topic = '/image_raw/compressed'
        image_sub = self.create_subscription(
            CompressedImage, image_topic, self.compressed_image_callback, 10, callback_group=macro_camera_cb_group)

        # FOCUS #########################################################################

        self.filtered_focus_value = 0.0
        self.focus_value_alpha = 0.5
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
        # gphoto2_image_sub = message_filters.Subscriber(self,
        #    Image, "/camera1/image_raw")
        ts = Tf2MessageFilter(self, [depth_image_sub, rgb_image_sub], 'part_frame',
                              'camera_depth_optical_frame', queue_size=1000)
        ts.registerCallback(self.depth_image_callback)

        # Inference

        self.twist = TwistStamped()
        self.twist.header.frame_id = 'tool0'
        inference_timer_period = 0.1
        # self.inference_timer = self.create_timer(
        # inference_timer_period, self.inference_timer_callback)

        # LIGHTS #########################################################################

        lights_cb_group = MutuallyExclusiveCallbackGroup()

        self.capture_image_future = None
        self.pixel_strip_msg = PixelStrip()
        self.pixel_count = 148
        self.wb = [1.0, 1.0, 1.0]
        self.pixel_strip_msg.pixel_colors = self.pixel_count * \
            [ColorRGBA(r=0.0, g=0.0, b=0.0)]
        pixel_pub_timer_period = 0.1
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

        self.m = 5
        self.k_p = 0.02
        self.c_p = 45.0
        self.k_o = 0.1
        self.c_o = 0.1

        self.pan_pos = (0., 0., 0.)
        self.pan_vel = (0., 0., 0.)
        self.pan_vel_max = (1, 1, 1)
        self.pan_goal = (0., 0., 0.)

        self.orbit_pos = (0., 0., 0.)
        self.orbit_vel = (0., 0., 0.)
        self.orbit_vel_max = (0.1, 0.1, 0.1)
        self.orbit_goal = (0., 0., 0.)

        self.zoom_pos = 0.0
        self.zoom_vel = 0.0
        self.zoom_vel_max = 1.0
        self.zoom_goal = 0.0

        # Call /servo_node/start_servo service
        self.get_logger().info('Connecting to servo node...')
        self.start_servo_cli = self.create_client(
            Trigger, '/servo_node/start_servo', callback_group=servo_cb_group)
        if not self.start_servo_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('start servo service not available, waiting again...')
        else:
            req = Trigger.Request()
            self.get_logger().info('Connected to servo node!')
            self.get_logger().info('Sending start servo request...')
            future = self.start_servo_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            self.get_logger().info('Servo node started!')

        self.twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.twist_pub_timer_period = 0.1
        self.twist_pub_timer = self.create_timer(
            self.twist_pub_timer_period, self.twist_pub_timer_callback, callback_group=servo_cb_group)

        light_ring = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.1, height=0.01)
        self.light_ring = o3d.geometry.LineSet.create_from_triangle_mesh(
            light_ring)
        self.camera = o3d.geometry.LineSet().create_camera_visualization(
            self.depth_intrinsic, extrinsic=np.eye(4))

        self.geom_pcd = self.generate_point_cloud()

        # TF2 #########################################################################

        self.tfBuffer = tf2_ros.Buffer()
        self.staticTfBroadcaster = tf2_ros.StaticTransformBroadcaster(self)

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

        # FOCUS EXPERIMENT

        # Run focus test
        self.focus_experiment_trigger = Trigger.Request()
        self.focus_experiment_cli = self.create_client(
            Trigger, '/inspection/count_callback', callback_group=None)

    def start_measure(self):
        self.t0 = time.time()

    def stop_measure(self):
        self.get_logger().info(f'Measurement time: {time.time() - self.t0}')

    def move_to_pose(self, tf, frame_id):
        self.moving_to_viewpoint = True

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

        req = MoveToPose.Request()
        req.target_pose = pose_stamped
        self.get_logger().info('Sending move to pose request...')
        future = self.move_to_pose_cli.call_async(req)
        future.add_done_callback(self.move_to_pose_callback)
        # rclpy.spin_until_future_complete(self, future)
        # resp = future.result()
        # self.get_logger().info('Move to pose request complete!')

    def move_to_pose_callback(self, future):
        try:
            resp = future.result()
            self.get_logger().info('Move to pose response: %s' % resp.done)
        except Exception as e:
            self.get_logger().info(
                'Service call failed %r' % (e,))

        self.moving_to_viewpoint = False
        self.last_move_successful = resp.done

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
        self.get_logger().info(f'Capturing image to: {file_path}')
        req = CaptureImage.Request()
        req.file_path = file_path
        self.capture_image_future = self.capture_image_cli.call_async(req)

        # rclpy.spin_until_future_complete(self, future)
        # resp = future.result()

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

    def twist_pub_timer_callback(self):
        # Do not publish if all twist values are zero
        px0 = self.pan_pos[0]
        gx = self.pan_goal[0]
        vx0 = self.pan_vel[0]
        ax = (self.k_p * (gx - px0) - self.c_p * vx0) / self.m
        vx1 = vx0 + ax * self.twist_pub_timer_period
        if abs(vx1) < 0.01:
            vx1 = 0.0
        elif vx1 > 0.0:
            vx1 = min(round(vx1, 3), self.pan_vel_max[0])
        else:
            vx1 = max(round(vx1, 3), -self.pan_vel_max[0])
        px1 = px0 + vx1 * self.twist_pub_timer_period

        py0 = self.pan_pos[1]
        gy = self.pan_goal[1]
        vy0 = self.pan_vel[1]
        ay = (self.k_p * (gy - py0) - self.c_p * vy0) / self.m
        vy1 = vy0 + ay * self.twist_pub_timer_period
        if abs(vy1) < 0.01:
            vy1 = 0.0
        elif vy1 > 0.0:
            vy1 = min(round(vy1, 3), self.pan_vel_max[1])
        else:
            vy1 = max(round(vy1, 3), -self.pan_vel_max[1])
        py1 = py0 + vy1 * self.twist_pub_timer_period

        py1 = round(py1, 3)

        pz0 = self.zoom_pos
        gz = self.zoom_goal
        vz0 = self.zoom_vel
        az = (self.k_p * (gz - pz0) - self.c_p * vz0) / self.m
        vz1 = vz0 + az * self.twist_pub_timer_period
        if abs(vz1) < 0.01:
            vz1 = 0.0
        elif vz1 > 0.0:
            vz1 = min(round(vz1, 3), self.zoom_vel_max)
        else:
            vz1 = max(round(vz1, 3), -self.zoom_vel_max)
        pz1 = pz0 + vz1 * self.twist_pub_timer_period
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

        # Publish twist if any values are non-zero
        if vx1 != 0.0 or vy1 != 0.0 or vz1 != 0.0:
            self.twist.twist.linear.x = -vx1
            self.twist.twist.linear.y = vz1
            self.twist.twist.linear.z = vy1

            self.twist.header.stamp = self.get_clock().now().to_msg()
            self.twist_pub.publish(self.twist)

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
        # gphoto2_image = self.bridge.imgmsg_to_cv2(
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
        self.T = T
        self.illuminance_image = hsv_image[:, :, 2]
        # self.gphoto2_image = gphoto2_image

    def gphoto2_image_callback(self, msg):
        self.gphoto2_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="rgb8").astype(np.uint8)

        height, width, _ = self.gphoto2_image.shape

        cx = self.focus_monitor.cx
        cy = self.focus_monitor.cy
        w = self.focus_monitor.w
        h = self.focus_monitor.h
        x0 = int(cx*width - w/2)
        y0 = int(cy*height - h/2)
        x1 = int(cx*width + w/2)
        y1 = int(cy*height + h/2)

        focus_value, focus_image = self.focus_monitor.measure_focus(
            self.gphoto2_image)

        # self.focus_metric_dict['sobel']['buffer'].append(metrics.)

        if len(self.focus_metric_dict['metrics']['sobel']['value']) > self.focus_metric_dict['buffer_size']:
            self.focus_metric_dict['metrics']['sobel']['value'].pop(0)
            self.focus_metric_dict['metrics']['sobel']['time'].pop(0)
        self.focus_metric_dict['metrics']['sobel']['time'].append(time.time())
        self.focus_metric_dict['metrics']['sobel']['value'].append(focus_value)
        self.focus_metric_dict['metrics']['sobel']['image'] = focus_image

        cv2.rectangle(self.gphoto2_image, (x0, y0), (x1, y1),
                      color=(204, 108, 231), thickness=2)
        #   color=(255, 255, 255), thickness=2)

    def set_focus_metric(self, name):
        self.focus_monitor.set_metric(name)

    def compressed_image_callback(self, msg):
        # self.stop_measure()
        # self.start_measure()
        self.gphoto2_image = self.bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding="rgb8").astype(np.uint8)

        height, width, _ = self.gphoto2_image.shape

        cx = self.focus_monitor.cx
        cy = self.focus_monitor.cy
        w = self.focus_monitor.w
        h = self.focus_monitor.h
        x0 = int(cx*width - w/2)
        y0 = int(cy*height - h/2)
        x1 = int(cx*width + w/2)
        y1 = int(cy*height + h/2)

        focus_value, focus_image = self.focus_monitor.measure_focus(
            self.gphoto2_image)

        self.filtered_focus_value = self.focus_value_alpha * focus_value + \
            (1 - self.focus_value_alpha) * self.filtered_focus_value

        # Publish focus value
        self.focus_pub.publish(FocusValue(
            header=msg.header, metric=self.focus_monitor.metric, data=self.filtered_focus_value, raw_data=focus_value))

        # self.focus_metric_dict['sobel']['buffer'].append(metrics.)

        if len(self.focus_metric_dict['metrics']['sobel']['filtered_value']) > self.focus_metric_dict['buffer_size']:
            self.focus_metric_dict['metrics']['sobel']['time'].pop(0)
            self.focus_metric_dict['metrics']['sobel']['filtered_value'].pop(0)
            self.focus_metric_dict['metrics']['sobel']['raw_value'].pop(0)
        self.focus_metric_dict['metrics']['sobel']['time'].append(time.time())
        self.focus_metric_dict['metrics']['sobel']['filtered_value'].append(
            self.filtered_focus_value)
        self.focus_metric_dict['metrics']['sobel']['raw_value'].append(
            focus_value)
        self.focus_metric_dict['metrics']['sobel']['image'] = focus_image

        # cv2.rectangle(self.gphoto2_image, (x0, y0), (x1, y1),
        #   color=(204, 108, 231), thickness=2)
        #   color=(255, 255, 255), thickness=2)

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
        return self.rgb_image, self.annotated_rgb_image, self.depth_image, self.depth_intrinsic, self.illuminance_image, self.gphoto2_image, self.T

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
