#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import io
import cv2
import threading
import numpy as np
import open3d as o3d  # . . . . . . . . . . . . . . . Open3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from PIL import Image as PILImage


class ScannerNode(Node):
    def __init__(self):
        super().__init__('scanner_node')
        self.bridge = CvBridge()

        self.dmap_fig = plt.figure()
        self.dmap_plt_cv2 = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_image = None

        self.dmap_sub = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.dmap_callback, 10)

    def dmap_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")

    def get_dmap(self):
        return self.depth_image.copy()


def update_figure(fig, scanner_node, img_plot):
    dmap = scanner_node.get_dmap()
    print(np.max(dmap))
    if dmap is not None:
        if colorbar is None:
            # Display the image
            im = img_plot.set_data(dmap)
            # Create colorbar if it doesn't exist
            ax = fig.gca()
            colorbar = fig.colorbar(im, ax=ax)
        else:
            # Update the image data
            img_plot.set_data(dmap)
            # Update the colorbar limits based on the new data
            colorbar.set_clim(vmin=np.min(dmap), vmax=np.max(dmap))
            colorbar.draw_all()
    return img_plot, colorbar


def main():
    rclpy.init()

    scanner_node = ScannerNode()

    scan_thread = threading.Thread(target=rclpy.spin, args=(scanner_node,))

    scan_thread.start()

    fig, ax = plt.subplots()
    # Adjust the shape as needed
    img_plot = ax.imshow(np.zeros((480, 640)), cmap='gray')
    ani = animation.FuncAnimation(fig, update_figure, fargs=(
        scanner_node, img_plot), interval=100, blit=True)

    plt.show()

    scan_thread.join()

    rclpy.shutdown()
