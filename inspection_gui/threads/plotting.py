#!/usr/bin/env python3

from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
import time
import cv2  # OpenCV library
import numpy as np
import threading
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class PlottingThread():
    def __init__(self):
        self.stopped = True
        self.t = threading.Thread(target=self.update_plots)
        self.t.daemon = True

        # Light Map
        self.update_light_map_flag = True
        self.light_map_image_cv2 = np.zeros((100, 100, 3))

        # Depth Image
        self.plot_depth_image_flag = True
        self.depth_image_cmap = 'viridis'
        self.depth_image = np.zeros((100, 100))
        self.depth_image_figure = plt.figure()
        self.depth_image_cv2 = np.zeros((100, 100, 3))

        # Metrics
        self.plot_focus_flag = True
        self.focus_metric_cmap = 'Greys'

        self.focus_metric_time = np.arange(0, 100, 1)
        self.focus_metric_data = np.zeros((100, 1))
        self.focus_metric_data_figure = plt.figure()
        self.focus_metric_data_figure.set_tight_layout(True)
        self.focus_metric_plot_cv2 = np.zeros((100, 100, 3))

        self.focus_metric_image = np.zeros((100, 100))
        self.focus_metric_image_figure = plt.figure()
        self.focus_metric_image_figure.set_tight_layout(True)
        self.focus_metric_image_cv2 = np.zeros((100, 100, 3))

        self.t0 = time.time()

    def start(self):
        self.stopped = False
        self.t.start()    # method passed to thread to read next available frame

    def update_depth_image(self, depth_image):
        self.depth_image = depth_image
        self.plot_depth_image_flag = True

    def update_focus_metric(self, focus_metric_time, focus_metric_data, focus_metric_image):
        self.focus_metric_time = focus_metric_time
        self.focus_metric_data = focus_metric_data
        self.focus_metric_image = focus_metric_image
        self.plot_focus_flag = True

    def get_depth_image(self):
        return self.depth_image_cv2.astype(np.uint8)

    def get_focus_metric_plot(self):
        return self.focus_metric_plot_cv2.astype(np.uint8)

    def get_focus_metric_image(self):
        return self.focus_metric_image_cv2.astype(np.uint8)

    def start_measure(self):
        self.t0 = time.time()

    def stop_measure(self):
        t1 = time.time()
        print(f'Plotting Thread: {t1-self.t0:.2f} seconds')

    def update_plots(self):
        while not self.stopped:
            # Depth Image Figures
            if self.plot_depth_image_flag:

                ax = self.depth_image_figure.add_subplot()
                pos = ax.imshow(self.depth_image, cmap=self.depth_image_cmap,
                                interpolation='none')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(pos, cax=cax)
                ax.axis('off')
                buf = BytesIO()
                self.depth_image_figure.savefig(
                    buf, format='png', bbox_inches='tight')
                buf.seek(0)
                depth_image_plot = np.frombuffer(
                    buf.getvalue(), dtype=np.uint8)
                self.depth_image_figure.clf()
                depth_image_cv2 = cv2.imdecode(
                    depth_image_plot, cv2.IMREAD_UNCHANGED)
                self.depth_image_cv2 = cv2.cvtColor(
                    depth_image_cv2, cv2.COLOR_BGR2RGB)

                self.plot_depth_image_flag = False

            # Focus Figures
            if self.plot_focus_flag:

                ax = self.focus_metric_data_figure.add_subplot()
                pos = ax.plot(self.focus_metric_time, self.focus_metric_data)
                ax.spines[['top', 'right']].set_visible(False)
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # ax.axis('off')
                # plt.savefig(buf, format='png', bbox_inches='tight')

                # self.focus_metric_data_figure.savefig(
                # buf, format='png', bbox_inches='tight')
                self.focus_metric_data_figure.canvas.draw()
                buf = self.focus_metric_data_figure.canvas.tostring_rgb()
                ncols, nrows = self.focus_metric_data_figure.canvas.get_width_height()
                plot_image = np.frombuffer(
                    buf, dtype=np.uint8).reshape(nrows, ncols, 3)

                self.focus_metric_data_figure.clf()
                # buf.seek(0)
                # plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                # plot_data_cv2 = cv2.imdecode(
                # plot_image, cv2.IMREAD_UNCHANGED)
                # self.focus_metric_plot_cv2 = cv2.cvtColor(
                # plot_image, cv2.COLOR_BGR2RGB)
                self.focus_metric_plot_cv2 = plot_image

                # FOCUS # IMAGE
                ax = self.focus_metric_image_figure.add_subplot()
                pos = ax.imshow(self.focus_metric_image, cmap=self.focus_metric_cmap,
                                interpolation='none')
                divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)
                # cbar = plt.colorbar(pos, cax=cax)
                ax.axis('off')

                self.focus_metric_image_figure.canvas.draw()
                buf = self.focus_metric_image_figure.canvas.tostring_rgb()
                ncols, nrows = self.focus_metric_image_figure.canvas.get_width_height()
                plot_image = np.frombuffer(
                    buf, dtype=np.uint8).reshape(nrows, ncols, 3)
                self.focus_metric_image_figure.clf()

                # plot_focus_image_cv2 = cv2.imdecode(
                # plot_image, cv2.IMREAD_UNCHANGED)
                # self.focus_metric_image_cv2 = cv2.cvtColor(
                # plot_focus_image_cv2, cv2.COLOR_BGR2RGB)
                self.focus_metric_image_cv2 = plot_image

                self.plot_focus_flag = True

    def stop(self):
        self.stopped = True
        self.t.join()
