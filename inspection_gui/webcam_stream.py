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

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rcl_interfaces.srv import ListParameters, DescribeParameters, GetParameters, SetParameters

from inspection_gui.tf2_message_filter import Tf2MessageFilter


class WebcamStream(Node):
    # initialization method
    def __init__(self, stream_id=0):
        super().__init__('scanner_node')

        self.log = []

        self.bridge = CvBridge()

        yolov8 = YOLO('yolov8n-seg.pt')
        yolov8.export(format='openvino')
        self.yolov8_seg = YOLO("yolov8n-seg_openvino_model/")

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

        self.depth_intrinsic_sub = self.create_subscription(
            Image, "/camera/camera/depth/camera_info", self.depth_intrinsic_callback, 10)
        # self.depth_image_sub = self.create_subscription(
        # Image, "/camera/camera/depth/image_rect_raw", self.depth_image_callback, 10)

        self.depth_trunc = 1.0

        depth_image_sub = message_filters.Subscriber(self,
                                                     Image, "/camera/camera/depth/image_rect_raw")
        rgb_image_sub = message_filters.Subscriber(self,
                                                   Image, "/camera/camera/color/image_rect_raw")
        ts = Tf2MessageFilter(self, [depth_image_sub, rgb_image_sub], 'part_frame',
                              'camera_depth_optical_frame', queue_size=1000)
        ts.registerCallback(self.depth_image_callback)

        # Inference

        inference_timer_period = 0.1
        self.inference_timer = self.create_timer(
            inference_timer_period, self.inference_timer_callback)

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

        # Macro Camera
        self.get_logger().info('Connecting to camera1 node...')
        camera_node_name = 'camera1'
        self.camera_node_list_parameters_cli = self.create_client(
            ListParameters, camera_node_name + '/list_parameters')
        while not self.camera_node_list_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('Connected!')

        req = ListParameters.Request()
        self.get_logger().info('Sending list parameters request...')
        future = self.camera_node_list_parameters_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        self.get_logger().info('Got parameters.')
        param_names = []
        for param_name in resp.result.names:
            self.get_logger().info(param_name)
            param_names.append(param_name)

        self.camera_node_describe_parameters_cli = self.create_client(
            DescribeParameters, camera_node_name + '/describe_parameters')
        while not self.camera_node_describe_parameters_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        req = DescribeParameters.Request()
        req.names = param_names
        self.get_logger().info('Sending describe parameters request...')
        future = self.camera_node_describe_parameters_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        self.get_logger().info(f'{camera_node_name} parameters:')
        self.camera_params = {}
        for param in resp.descriptors:
            self.camera_params[param.name] = {}
            self.camera_params[param.name]['type'] = param.type
            self.camera_params[param.name]['description'] = param.description
            self.camera_params[param.name]['choices'] = param.additional_constraints.split(
                '\n')
        pretty = json.dumps(self.camera_params, indent=4)
        print(pretty)

    def inference_timer_callback(self):
        original_size = self.rgb_image.shape

        img = self.rgb_image.copy()

        scale_percent = 100  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        rgb_image_resized = cv2.resize(
            img, dim, interpolation=cv2.INTER_AREA).astype(np.uint8)
        results = self.yolov8_seg(img.astype(np.uint8))
        # self.annotated_rgb_image = self.rgb_image
        self.annotated_rgb_image = results[0].plot().astype(np.uint8)

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

    def get_data(self):
        return self.rgb_image, self.annotated_rgb_image, self.depth_image, self.depth_intrinsic, self.illuminance_image, self.T

    def read_rgb_image(self):
        return self.annotated_rgb_image

    def read_depth_image(self):
        return self.depth_image.copy()

    def read_point_cloud(self):
        return self.geom_pcd

    def read_camera(self):
        return self.camera

    def read_light_ring(self):
        return self.light_ring

    def read_log(self):
        return self.log

    # method to stop reading frames
    def stop(self):
        self.stopped = True
        self.t.join()
        # self.t2.join()
