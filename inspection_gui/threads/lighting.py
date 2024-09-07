import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from io import BytesIO


def gaussian(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))


class LightMap():
    def __init__(self, shape_mm, dpmm, led_locations):
        self.stopped = True
        self.t = threading.Thread(target=self.update_map)
        self.t.daemon = True

        self.update_map_flag = True

        self.intensity = 0

        self.width_mm = shape_mm[0]
        self.height_mm = shape_mm[1]

        width_px = int(shape_mm[0]*dpmm)
        height_px = int(shape_mm[1]*dpmm)

        self.led_locations = led_locations
        self.led_locations_px = []
        for i in range(len(led_locations)):
            led_location = (
                int(led_locations[i][0]*dpmm + width_px/2), int(led_locations[i][1]*dpmm + height_px/2))
            self.led_locations_px.append(led_location)

        x = np.linspace(-shape_mm[0]/2, shape_mm[0]/2, width_px)
        y = np.linspace(-shape_mm[1]/2, shape_mm[1]/2, height_px)
        self.x, self.y = np.meshgrid(x, y)

        self.mu_x = 0
        self.mu_y = 0
        self.sigma = 20

        self.plot_light_map_flag = True
        self.map_image_cv2 = np.zeros((100, 100, 3))
        # self.update_map()

        self.figure = plt.figure()
        self.cmap = "gray"

    def start(self):
        self.stopped = False
        self.t.start()

    def update_map(self):
        while not self.stopped:
            if self.update_map_flag:
                self.map = gaussian(self.x, self.y, self.mu_x,
                                    self.mu_y, self.sigma)

                map = self.map * self.intensity
                map = np.clip(map, 0, 255)
                map = map.astype(np.uint8)

                map_image_cv2 = cv2.cvtColor(
                    map, cv2.COLOR_BGR2RGB)
                # draw a samll circle at the location of the LEDs
                for i in range(len(self.led_locations)):
                    center = (self.led_locations_px[i]
                              [0], self.led_locations_px[i][1])
                    cv2.circle(map_image_cv2, center, 20, (255, 255, 255), 3)

                # Flip along the x-axis
                self.map_image_cv2 = cv2.flip(map_image_cv2, 0)

                pixel_values = []
                for i in range(len(self.led_locations)):
                    pixel_values.append(int(self.intensity*self.map[self.led_locations_px[i][1],
                                                                    self.led_locations_px[i][0]]))
                self.pixel_values = pixel_values

                self.update_map_flag = False

    def set_intensity(self, intensity):
        self.intensity = int(intensity)
        self.update_map_flag = True

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.update_map_flag = True

    def set_mu_x(self, x):
        self.mu_x = self.width_mm*x
        self.update_map_flag = True

    def set_mu_y(self, y):
        self.mu_y = self.height_mm*y
        self.update_map_flag = True

    def get_pixel_values(self):
        # Sample the map at the location of the LEDs
        return self.pixel_values

    def get_map_image(self):
        return self.map_image_cv2

    def stop(self):
        self.stopped = True
        self.t.join()
