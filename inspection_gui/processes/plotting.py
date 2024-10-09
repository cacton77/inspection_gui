#!/usr/bin/env python3

from mpl_toolkits.axes_grid1 import make_axes_locatable
from io import BytesIO
import time
import cv2  # OpenCV library
import numpy as np
import threading
from multiprocessing import Process, shared_memory
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class PlottingProcess():
    def __init__(self, pipe, shared_data_dict):
        self.stopped = True

        self.p = Process(target=self.update_plots)
        self.pipe = pipe

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
        self.filtered_focus_metric_data = np.zeros((100, 1))
        self.raw_focus_metric_data = np.zeros((100, 1))
        self.focus_metric_data_figure = plt.figure()
        self.focus_metric_data_figure.set_tight_layout(True)
        self.focus_metric_plot_cv2 = np.zeros((100, 100, 3))

        self.focus_metric_image = np.zeros((100, 100))
        self.focus_metric_image_figure = plt.figure()
        self.focus_metric_image_figure.set_tight_layout(True)
        self.focus_metric_image_cv2 = np.zeros((100, 100, 3))

        self.data_in = {'depth_image': self.depth_image,
                        'focus_metric_time': self.focus_metric_time,
                        'filtered_focus_metric_data': self.filtered_focus_metric_data,
                        'raw_focus_metric_data': self.raw_focus_metric_data,
                        'focus_metric_image': self.focus_metric_image}

        self.shared_data_dict = shared_data_dict

        self.data_out = {'depth_image': self.depth_image_cv2.shape,
                         'focus_plot': self.focus_metric_plot_cv2.shape,
                         'focus_image': self.focus_metric_image_cv2.shape}

        self.sh_depth_image = shared_memory.SharedMemory(
            create=True, size=self.depth_image.nbytes)
        self.sh_focus_plot = shared_memory.SharedMemory(
            create=True, size=self.focus_metric_plot_cv2.nbytes)
        self.sh_focus_image = shared_memory.SharedMemory(
            create=True, size=self.focus_metric_image_cv2.nbytes)

        self.t0 = time.time()

    def start(self):
        self.stopped = False
        self.p.start()

    def start_measure(self):
        self.t0 = time.time()

    def stop_measure(self):
        t1 = time.time()
        print(f'Plotting Thread: {t1-self.t0:.2f} seconds')

    def update_plots(self):
        while not self.stopped:
            self.data_in = self.pipe.recv()

            self.start_measure()

            self.depth_image = self.data_in['depth_image']
            self.focus_metric_time = self.data_in['focus_metric_time']
            self.filtered_focus_metric_data = self.data_in['filtered_focus_metric_data']
            self.raw_focus_metric_data = self.data_in['raw_focus_metric_data']
            self.focus_metric_image = self.data_in['focus_metric_image']

            # DEPTH IMAGE FIGURES ######################################

            ax = self.depth_image_figure.add_subplot()
            pos = ax.imshow(self.depth_image, cmap=self.depth_image_cmap,
                            interpolation='none')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(pos, cax=cax)
            ax.axis('off')
            self.depth_image_figure.canvas.draw()
            buf = self.depth_image_figure.canvas.tostring_rgb()
            ncols, nrows = self.depth_image_figure.canvas.get_width_height()
            plot_image = np.frombuffer(
                buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            self.depth_image_figure.clf()
            self.depth_image = plot_image.astype(np.uint8)

            self.shared_data_dict['depth_image']['shape'] = plot_image.shape

            shm = shared_memory.SharedMemory(name='depth_image')
            shared_array = np.ndarray(
                self.depth_image.shape, dtype=plot_image.dtype, buffer=shm.buf)
            np.copyto(shared_array, plot_image)

            # FOCUS FIGURES ############################################

            # FOCUS METRIC PLOT

            ax = self.focus_metric_data_figure.add_subplot()
            ax.plot(self.focus_metric_time,
                    self.raw_focus_metric_data)
            ax.plot(self.focus_metric_time,
                    self.filtered_focus_metric_data, color='green')
            ax.spines[['top', 'right']].set_visible(False)
            self.focus_metric_data_figure.canvas.draw()
            buf = self.focus_metric_data_figure.canvas.tostring_rgb()
            ncols, nrows = self.focus_metric_data_figure.canvas.get_width_height()
            plot_image = np.frombuffer(
                buf, dtype=np.uint8).reshape(nrows, ncols, 3)

            self.focus_metric_data_figure.clf()
            self.focus_metric_plot_cv2 = plot_image.astype(np.uint8)

            self.shared_data_dict['focus_plot']['shape'] = plot_image.shape

            shm = shared_memory.SharedMemory(name='focus_plot')
            shared_array = np.ndarray(
                plot_image.shape, dtype=plot_image.dtype, buffer=shm.buf)
            np.copyto(shared_array, plot_image)

            # FOCUS IMAGE

            ax = self.focus_metric_image_figure.add_subplot()
            pos = ax.imshow(self.focus_metric_image, cmap=self.focus_metric_cmap,
                            interpolation='none')
            divider = make_axes_locatable(ax)
            ax.axis('off')
            self.focus_metric_image_figure.canvas.draw()
            buf = self.focus_metric_image_figure.canvas.tostring_rgb()
            ncols, nrows = self.focus_metric_image_figure.canvas.get_width_height()
            plot_image = np.frombuffer(
                buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            self.focus_metric_image_figure.clf()
            self.focus_metric_image_cv2 = plot_image.astype(np.uint8)

            self.shared_data_dict['focus_image']['shape'] = plot_image.shape

            shm = shared_memory.SharedMemory(name='focus_image')
            shared_array = np.ndarray(
                plot_image.shape, dtype=plot_image.dtype, buffer=shm.buf)
            np.copyto(shared_array, plot_image)

            self.stop_measure()

            # SEND DATA OUT ############################################

            # self.data_out = {'depth_image': self.depth_image.shape,
            #                  'focus_plot': self.focus_metric_plot_cv2.shape,
            #                  'focus_image': self.focus_metric_image_cv2.shape}
            # self.pipe.send(self.data_out)

    def stop(self):
        self.stopped = True
        self.t.join()
