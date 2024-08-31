import cv2
import numpy as np


class FocusMonitor:

    MONITOR = 0
    RECORD = 1

    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

        self.state = self.MONITOR

    def get_metrics():
        return ['Variance of Sobel', 'Squared Gradient', 'FSWM', 'FFT']

    def sobel(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        sobel_image = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3)
        sobel_value = sobel_image[y0:y1, x0:x1].var()
        image_out = sobel_image[y0:y1, x0:x1]
        # image_out = cv2.convertScaleAbs(sobel_image)
        # image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

        return sobel_value, image_out

    def squared_gradient_focus_measure(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        roi_gray_image = gray_image[y0:y1, x0:x1]

        # Compute the squared differences of adjacent pixels in both directions
        gradient_x = np.diff(roi_gray_image, axis=0)
        gradient_y = np.diff(roi_gray_image, axis=1)
        squared_gradient_x = gradient_x ** 2
        squared_gradient_y = gradient_y ** 2
        focus_value = np.sum(squared_gradient_x) + np.sum(squared_gradient_y)

        combined_gradient = np.sqrt(
            squared_gradient_x[:-1, :] + squared_gradient_y[:, :-1])
        normalized_image = cv2.normalize(
            combined_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out

    def fswm_focus_measure(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        roi_gray_image = gray_image[y0:y1, x0:x1]

        # Applying Frequency Selective Weighted Median Filter
        ksize = 3
        median_filtered_x = cv2.medianBlur(roi_gray_image, ksize)
        median_filtered_y = cv2.medianBlur(roi_gray_image.T, ksize).T
        fswm_x = np.abs(roi_gray_image - median_filtered_x)
        fswm_y = np.abs(roi_gray_image - median_filtered_y)

        focus_value = np.mean(fswm_x + fswm_y)

        combined_fswm = fswm_x + fswm_y
        normalized_image = cv2.normalize(
            combined_fswm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out

    def fft_focus_measure(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        roi_gray_image = gray_image[y0:y1, x0:x1]

        # Apply FFT
        f = np.fft.fft2(roi_gray_image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # Calculate high frequency energy
        rows, cols = roi_gray_image.shape
        crow, ccol = rows // 2, cols // 2
        high_freq_energy = np.sum(
            magnitude_spectrum[crow-10:crow+10, ccol-10:ccol+10])

        # Create a visual representation
        normalized_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_spectrum, cv2.COLOR_GRAY2RGB)
        return high_freq_energy, image_out
