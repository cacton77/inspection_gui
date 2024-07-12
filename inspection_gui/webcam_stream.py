#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2  # OpenCV library
import open3d as o3d
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from io import BytesIO
import message_filters

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from inspection_gui.tf2_message_filter import Tf2MessageFilter


class WebcamStream(Node):
    # initialization method
    def __init__(self, stream_id=0):
        super().__init__('scanner_node')

        self.bridge = CvBridge()

        self.stopped = True        # thread instantiation
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

        self.frame_id = None
        self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        self.depth_image = np.zeros((480, 640), dtype=np.uint16)

        self.depth_intrinsic_sub = self.create_subscription(
            Image, "/camera/camera/depth/camera_info", self.depth_intrinsic_callback, 10)
        # self.depth_image_sub = self.create_subscription(
        # Image, "/camera/camera/depth/image_rect_raw", self.depth_image_callback, 10)
        depth_image_sub = message_filters.Subscriber(self,
                                                     Image, "/camera/camera/depth/image_rect_raw")
        ts = Tf2MessageFilter(depth_image_sub, 'world',
                              'camera_link', queue_size=1000)
        ts.registerCallback(self.depth_image_callback)

        # Generate first point cloud
        self.geom_pcd = self.generate_point_cloud()

        # Generate an np array of green colors with length of geom_pcd.points
        green_color = np.array([0, 1, 0])
        # Expand array to match the length of geom_pcd.points
        green_color = np.expand_dims(green_color, axis=0)
        self.geom_pcd.colors = o3d.utility.Vector3dVector(
            green_color)

    def generate_point_cloud(self):
        new_pcd = o3d.geometry.PointCloud()
        points = np.random.rand(100, 3)
        new_pcd.points = o3d.utility.Vector3dVector(points)
        return new_pcd

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame

    def update(self):
        rclpy.spin(self)

    def depth_intrinsic_callback(self, msg):
        # Convert ROS message to Open3D camera intrinsic
        self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            msg.width, msg.height, msg.K[0], msg.K[4], msg.K[2], msg.K[5])

    def depth_image_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough").astype(np.float32) / 10.0
        # Set all pixels in image above 1000 to 0
        depth_image_cm = self.depth_image / 1.0
        depth_image_cm[depth_image_cm > 30] = 0
        # Apply gaussian blur to depth image
        depth_image_cm = cv2.GaussianBlur(
            depth_image_cm, (7, 7), 0, 0, cv2.BORDER_DEFAULT)

        self.geom_pcd = o3d.geometry.PointCloud().create_from_depth_image(o3d.geometry.Image(depth_image_cm),
                                                                          intrinsic=self.depth_intrinsic, extrinsic=np.eye(4), depth_scale=1.0, depth_trunc=250.0)

    def read_depth_image(self):
        return self.depth_image.copy()

    def read_point_cloud(self):
        return self.geom_pcd

    # method to stop reading frames
    def stop(self):
        self.stopped = True
        self.t.join()
