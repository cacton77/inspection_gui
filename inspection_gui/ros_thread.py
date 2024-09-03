#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2  # OpenCV library
import open3d as o3d
import numpy as np
import threading
import time
import json
import matplotlib.pyplot as plt
import message_filters
from io import BytesIO
from math import pi
from ultralytics import YOLO
import pytransform3d.rotations as pr

import tf2_ros
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import ListParameters, DescribeParameters, GetParameters, SetParameters

from inspection_gui.tf2_message_filter import Tf2MessageFilter
from inspection_gui.focus_monitor import FocusMonitor
from inspection_srvs.srv import CaptureImage
from std_msgs.msg import ColorRGBA


class RosThread(Node):
    # initialization method
    def __init__(self, stream_id=0):
        super().__init__('scanner_node')

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

        # Main Camera
        self.focus_monitor = FocusMonitor(0.5, 0.5, 100, 100)
        self.focus_monitor.state = 'fft'
        self.gphoto2_image = np.zeros((576, 1024, 3), dtype=np.uint8)
        self.focus_metric_dict = {}
        self.focus_metric_dict['buffer_size'] = 1000
        self.focus_metric_dict['metrics'] = {}
        self.focus_metric_dict['metrics']['sobel'] = {}
        self.focus_metric_dict['metrics']['sobel']['value'] = 1000*[0]
        self.focus_metric_dict['metrics']['sobel']['time'] = 1000*[0]

        self.depth_intrinsic_sub = self.create_subscription(
            Image, "/camera/camera/depth/camera_info", self.depth_intrinsic_callback, 10)
        # self.depth_image_sub = self.create_subscription(
        # Image, "/camera/camera/depth/image_rect_raw", self.depth_image_callback, 10)

        self.depth_trunc = 1.0

        depth_image_sub = message_filters.Subscriber(self,
                                                     Image, "/camera/camera/depth/image_rect_raw")
        rgb_image_sub = message_filters.Subscriber(self,
                                                   Image, "/camera/camera/color/image_rect_raw")
        # gphoto2_image_sub = message_filters.Subscriber(self,
        #    Image, "/camera1/image_raw")
        gphoto2_image_sub = self.create_subscription(
            Image, "/camera1/image_raw", self.gphoto2_image_callback, 10)

        ts = Tf2MessageFilter(self, [depth_image_sub, rgb_image_sub], 'part_frame',
                              'camera_depth_optical_frame', queue_size=1000)
        ts.registerCallback(self.depth_image_callback)

        # Inference

        self.twist = TwistStamped()
        inference_timer_period = 0.1
        # self.inference_timer = self.create_timer(
        # inference_timer_period, self.inference_timer_callback)

        # Lights

        self.capture_image_future = None
        self.pixel_color = ColorRGBA()
        pixel_pub_timer_period = 0.1
        self.pixel_pub = self.create_publisher(ColorRGBA, '/pixel_strip', 10)
        self.pixel_pub_timer = self.create_timer(
            pixel_pub_timer_period, self.pixel_pub_timer_callback)

        # Send moveit servo command

        self.m = 10
        self.k_p = 0.01
        self.c_p = 25.0
        self.k_o = 0.1
        self.c_o = 0.1

        self.pan_pos = (0., 0., 0.)
        self.pan_vel = (0., 0., 0.)
        self.pan_vel_max = (0.1, 0.1, 0.1)
        self.pan_goal = (0., 0., 0.)

        self.orbit_pos = (0., 0., 0.)
        self.orbit_vel = (0., 0., 0.)
        self.orbit_vel_max = (0.1, 0.1, 0.1)
        self.orbit_goal = (0., 0., 0.)

        self.zoom_pos = 0.0
        self.zoom_vel = 0.0
        self.zoom_vel_max = 0.1
        self.zoom_goal = 0.0

        self.twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.twist_pub_timer_period = 0.1
        self.twist_pub_timer = self.create_timer(
            self.twist_pub_timer_period, self.twist_pub_timer_callback)

        # Generate first point cloud
        light_ring = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.1, height=0.01)
        # o3d.io.read_triangle_mesh(
        # "/home/col/Inspection/dev_ws/src/inspection_gui/inspection_gui/light_ring.stl")
        self.light_ring = o3d.geometry.LineSet.create_from_triangle_mesh(
            light_ring)
        self.camera = o3d.geometry.LineSet().create_camera_visualization(
            self.depth_intrinsic, extrinsic=np.eye(4))

        self.geom_pcd = self.generate_point_cloud()
        self.camera_params = {}

        # Macro Camera
        self.get_logger().info('Connecting to camera1 node...')
        camera_node_name = 'camera1'
        param_names = []
        self.camera_node_list_parameters_cli = self.create_client(
            ListParameters, camera_node_name + '/list_parameters')
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
            DescribeParameters, camera_node_name + '/describe_parameters')
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
            GetParameters, camera_node_name + '/get_parameters')
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
        self.camera_node_set_parameters_cli = self.create_client(
            SetParameters, 'camera1/set_parameters')
        if not self.camera_node_set_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set parameters service not available, waiting again...')

        # Get all frames in tf tree
        self.tfBuffer = tf2_ros.Buffer()
        self.staticTfBroadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Capture Image Client
        self.capture_image_cli = self.create_client(
            CaptureImage, '/capture_image')
        if not self.capture_image_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('capture image service not available, waiting again...')
        else:
            self.get_logger().info('Connected to capture image service!')

    def lights_on(self, value):
        self.pixel_color.r = float(value)
        self.pixel_color.g = float(value)
        self.pixel_color.b = float(value)

    def lights_off(self):
        self.pixel_color.r = 0.0
        self.pixel_color.g = 0.0
        self.pixel_color.b = 0.0

    def pixel_pub_timer_callback(self):
        self.pixel_pub.publish(self.pixel_color)

    def capture_image(self, file_path):
        self.lights_on(100.0)
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
        future = self.camera_node_set_parameters_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        self.get_logger().info('Parameters set!')

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

        self.twist.twist.linear.x = vx1
        self.twist.twist.linear.y = vy1
        self.twist.twist.linear.z = vz1

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

    def update(self):
        rclpy.spin(self)

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
