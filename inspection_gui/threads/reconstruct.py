import time
import threading
import numpy as np
import open3d as o3d


class ReconstructThread:
    def __init__(self, rate=10):

        self.period = 1/rate

        self.stopped = True
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True

        self.depth_trunc = 1.0
        self.rgb_image_o3d = o3d.geometry.Image(
            np.zeros((480, 640, 3), dtype=np.uint8))
        self.depth_image_o3d = o3d.geometry.Image(
            np.zeros((480, 640, 1), dtype=np.float32))
        self.depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        self.live_point_cloud = o3d.geometry.PointCloud()
        self.T = np.eye(4)

    def start(self):
        self.stopped = False
        self.t0 = time.time()
        self.t.start()

    def update(self):
        while not self.stopped:
            self.reconstruct()
            t1 = time.time()
            time.sleep(max(0, self.period - (t1 - self.t0)))
            self.t0 = t1

    def reconstruct(self):

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            self.rgb_image_o3d, self.depth_image_o3d, depth_scale=1.0, depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)

        self.live_point_cloud = o3d.geometry.PointCloud().create_from_rgbd_image(
            rgbd_image, intrinsic=self.depth_intrinsic)  # , extrinsic=np.eye(4), depth_scale=1.0)

        self.live_point_cloud.transform(self.T)
        self.live_point_cloud.scale(100.0, center=np.array([0, 0, 0]))

    def stop(self):
        print("Stopping reconstruct thread...")
        self.stopped = True
        self.t.join()
        print("Reconstruct thread stopped.")

    def get_pcd(self):
        return self.live_point_cloud
