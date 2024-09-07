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

        self.dict = {}

    def get_metrics():
        return ['Variance of Sobel', 'Squared Gradient', 'FSWM', 'FFT']

    def measure_focus(self, image_in):
        if self.state == 'sobel':
            focus_value, focus_image = self.sobel(image_in)
        elif self.state == 'squared_gradient':
            focus_value, focus_image = self.squared_gradient(image_in)
        elif self.state == 'squared_sobel':
            focus_value, focus_image = self.squared_sobel(image_in)
        elif self.state == 'fswm':
            focus_value, focus_image = self.fswm(image_in)
        elif self.state == 'fft':
            focus_value, focus_image = self.fft(image_in)
        return focus_value, focus_image

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

    def squared_gradient(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Compute the squared differences of adjacent pixels in both directions
        gradient_x = np.diff(gray_image, axis=0)
        gradient_y = np.diff(gray_image, axis=1)
        squared_gradient_x = gradient_x**2
        squared_gradient_y = gradient_y**2

        # # Adjust the shapes to be compatible for addition
        min_height = min(
            squared_gradient_x.shape[0], squared_gradient_y.shape[0])
        min_width = min(
            squared_gradient_x.shape[1], squared_gradient_y.shape[1])

        squared_gradient_x = squared_gradient_x[:min_height, :min_width]
        squared_gradient_y = squared_gradient_y[:min_height, :min_width]

        # + (np.var(squared_gradient_x[yl:yh, xl:xh]) + np.var(squared_gradient_y[yl:yh, xl:xh]))/2
        focus_value = ((np.var(
            squared_gradient_x[y0:y1, x0:x1]) + np.mean(squared_gradient_y[y0:y1, x0:x1]))**1.5)/2

        combined_gradient = np.sqrt(
            squared_gradient_x+squared_gradient_y).astype(np.float32)
        normalized_image = cv2.normalize(
            combined_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[
            y0:y1, x0:x1]

        return focus_value, image_out

    def squared_sobel(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Compute the squared differences of adjacent pixels in both directions using Sobel
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # np.var(smoothed_combined_gradient[yl:yh, xl:xh]) #+ np.mean(smoothed_combined_gradient[yl:yh, xl:xh])**1.5
        focus_value = np.var(gradient_magnitude[y0:y1, x0:x1])

        signal = np.mean(gradient_magnitude[y0:y1, x0:x1])
        noise = np.std(gradient_magnitude[y0:y1, x0:x1])
        snr = signal/noise

        # normalized_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.convertScaleAbs(gradient_magnitude)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]

        return focus_value, image_out

    def fswm(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        ksize = 17
        sigma = 1.5

        # Apply median filter in x and y directions
        median_filtered_x = cv2.medianBlur(gray_image, ksize)
        median_filtered_y = cv2.medianBlur(gray_image.T, ksize).T

        fswm_x = np.abs(gray_image - median_filtered_x)
        fswm_y = np.abs(gray_image - median_filtered_y)
        combined_fswm = fswm_x + fswm_y

        # Apply Gaussian blur to denoise
        denoised_combined_fswm = cv2.GaussianBlur(
            combined_fswm, (0, 0), sigmaX=sigma, sigmaY=sigma)

        focus_value = np.var(denoised_combined_fswm[y0:y1, x0:x1])

        normalized_image = cv2.normalize(
            denoised_combined_fswm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[
            y0:y1, x0:x1]

        return focus_value, image_out

    def fft(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        size = 30
        # Apply FFT to the entire grayscale image
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)

        # Logarithmic scaling for better visualization of the magnitude spectrum
        magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

        # Determine the center of the frequency spectrum
        rows, cols = gray_image.shape
        cX, cY = cols // 2, rows // 2

        # Zero out the low-frequency components around the center
        fshift[cY - size:cY + size, cX - size:cX + size] = 0

        # Apply the inverse FFT to focus on high-frequency components
        f_ishift = np.fft.ifftshift(fshift)
        recon = np.fft.ifft2(f_ishift)
        recon = np.abs(recon)

        # Calculate the focus value as the variance of the high-frequency components in the ROI
        focus_value = np.var(recon)

        # Normalize the magnitude spectrum for visualization
        normalized_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_spectrum, cv2.COLOR_GRAY2RGB)

        return focus_value, image_out
