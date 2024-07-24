#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2  # OpenCV library
import open3d as o3d
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import message_filters
from io import BytesIO
from math import pi
from ultralytics import YOLO

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from inspection_gui.tf2_message_filter import Tf2MessageFilter


class WebcamStream(Node):
    # initialization method
    def __init__(self, stream_id=0):
        super().__init__('scanner_node')

        self.bridge = CvBridge()

        yolov8 = YOLO('yolov8n-seg.pt')
        yolov8.export(format='openvino')
        self.yolov8_seg = YOLO("yolov8n-seg_openvino_model/")

        self.stopped = True        # thread instantiation
        self.t = threading.Thread(target=self.update, args=())
        self.t2 = threading.Thread(
            target=self.inference_timer_callback, args=())
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

        # Voxel grid construction
        self.buffer_length = 10
        self.geom_pcd_buffer = []
        self.voxel_size = 0.1
        self.voxel_grid = o3d.geometry.VoxelGrid()

        # Generate an np array of green colors with length of geom_pcd.points
        green_color = np.array([0, 1, 0])
        # Expand array to match the length of geom_pcd.points
        green_color = np.expand_dims(green_color, axis=0)
        self.geom_pcd.colors = o3d.utility.Vector3dVector(
            green_color)

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

    def update2(self):
        # Combine all point clouds in buffer and compute voxel grid at 1 Hz
        while not self.stopped:
            t0 = time.time()
            self.compute_voxel_grid_from_buffer()
            t1 = time.time()
            print("Time to compute voxel grid: ", t1 - t0)
            time.sleep(1 - (t1 - t0))

    def compute_voxel_grid_from_buffer(self):
        # Combine all point clouds in buffer
        combined_pcd = o3d.geometry.PointCloud()
        if len(self.geom_pcd_buffer) > 0:
            for pcd in self.geom_pcd_buffer:
                combined_pcd += pcd
        # Create a voxel grid from combined pcd
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            combined_pcd, 0.01)

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

        # Add geom_pcd to buffer. If buffer is full, remove oldest pcd
        # self.geom_pcd_buffer.append(self.geom_pcd)
        # if len(self.geom_pcd_buffer) > self.buffer_length:
        # self.geom_pcd_buffer.pop(0)

    def get_data(self):
        return self.rgb_image, self.annotated_rgb_image, self.depth_image, self.depth_intrinsic, self.illuminance_image, self.T

    def read_rgb_image(self):
        return self.annotated_rgb_image

    def read_depth_image(self):
        return self.depth_image.copy()

    def read_point_cloud(self):
        return self.geom_pcd

    def read_voxel_grid(self):
        return self.voxel_grid

    def read_camera(self):
        return self.camera

    def read_light_ring(self):
        return self.light_ring

    # method to stop reading frames
    def stop(self):
        self.stopped = True
        self.t.join()
        # self.t2.join()
