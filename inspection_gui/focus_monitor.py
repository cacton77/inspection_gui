import cv2
import numpy as np
from scipy.stats import entropy
import cupy as cp

class FocusMonitor:

    def __init__(self, cx, cy, w, h, metric='sobel'):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

        self.metric = metric

        self.dict = {}

    def get_metrics():
        return ['Variance of Sobel', 'Squared Gradient', 'Squared Sobel', 'FSWM', 'FFT', 'Mix Sobel', 'Sobel+Laplacian', 'combined_focus_measure', 'combined_focus_measure2']

    def set_metric(self, name):
        if name == 'Variance of Sobel':
            self.metric = 'sobel'
        elif name == 'Squared Gradient':
            self.metric = 'squared_gradient'
        elif name == 'Squared Sobel':
            self.metric = 'squared_sobel'
        elif name == 'FSWM':
            self.metric = 'fswm'
        elif name == 'FFT':
            self.metric = 'fft'
        elif name =='Mix Sobel':
            self.metric = 'mix_sobel'
        elif name == 'Wavelet':
            self.metric = 'wavelet'
        elif name == 'Sobel+Laplacian':
            self.metric = 'sobel_laplacian'
        elif name == 'Sobel+Wavelet':
            self.metric = 'sobel_wavelet'
        elif name == 'combined_focus_measure':
            self.metric = 'combined_focus_measure'
        elif name == 'combined_focus_measure2':
            self.metric = 'combined_focus_measure2'            

    def measure_focus(self, image_in):
        if self.metric == 'sobel':
            focus_value, focus_image = self.sobel(image_in)
        elif self.metric == 'squared_gradient':
            focus_value, focus_image = self.squared_gradient(image_in)
        elif self.metric == 'squared_sobel':
            focus_value, focus_image = self.squared_sobel(image_in)
        elif self.metric == 'fswm':
            focus_value, focus_image = self.fswm(image_in)
        elif self.metric == 'fft':
            focus_value, focus_image = self.fft(image_in)
        elif self.metric == 'mix_sobel':
            focus_value, focus_image = self.mix_sobel(image_in)
        elif self.metric == 'wavelet':
            focus_value, focus_image = self.wavelet(image_in)
        elif self.metric == 'sobel_laplacian':
            focus_value, focus_image = self.sobel_laplacian(image_in)
        elif self.metric == 'combined_focus_measure2':
            focus_value, focus_image = self.combined_focus_measure2(image_in)
        elif self.metric == 'combined_focus_measure':
            focus_value, focus_image = self.combined_focus_measure(image_in)
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
        image_out = cv2.convertScaleAbs(sobel_image)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

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

        # ((np.var(squared_gradient_x[y0:y1, x0:x1]) + np.mean(squared_gradient_y[y0:y1, x0:x1]))**1.5)/2
        focus_value = np.var(squared_gradient_x[y0:y1, x0:x1])

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

        signal = np.mean(gradient_magnitude[y0:y1, x0:x1])
        noise = np.std(gradient_magnitude[y0:y1, x0:x1])
        snr = signal/noise

        # np.var(smoothed_combined_gradient[yl:yh, xl:xh]) #+ np.mean(smoothed_combined_gradient[yl:yh, xl:xh])**1.5
        focus_value = np.var(gradient_magnitude[y0:y1, x0:x1])

        # normalized_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # image_out = cv2.convertScaleAbs(gradient_magnitude)
        # image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]
        
        normalized_image = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[
            y0:y1, x0:x1]
        return focus_value, image_out

    def fswm(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        # ksize = 17
        # sigma = 1.5

        # # Apply median filter in x and y directions
        # median_filtered_x = cv2.medianBlur(gray_image, ksize)
        # median_filtered_y = cv2.medianBlur(gray_image.T, ksize).T

        # fswm_x = np.abs(gray_image - median_filtered_x)
        # fswm_y = np.abs(gray_image - median_filtered_y)
        # combined_fswm = fswm_x + fswm_y

        # # Apply Gaussian blur to denoise
        # denoised_combined_fswm = cv2.GaussianBlur(
        #     combined_fswm, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # focus_value = np.var(denoised_combined_fswm[y0:y1, x0:x1])

        # normalized_image = cv2.normalize(
        #     denoised_combined_fswm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[
        #     y0:y1, x0:x1]

        # Apply a bandpass filter using Difference of Gaussians (DoG)
        sigma_low = 2.5
        sigma_high = 3.0
        blur_low = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
        blur_high = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
        bandpass = blur_low - blur_high

        # Create a weight matrix
        rows, cols = bandpass.shape
        center_y, center_x = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.max(distance)
        weights = 1 - (distance / max_distance)  # Weights decrease with distance from center

        # Compute the weighted mean
        weighted_bandpass = bandpass * weights
        focus_value = np.var(bandpass[y0:y1, x0:x1])

        # For visualization, normalize the weighted bandpass image
        bandpass_normalized = cv2.normalize(weighted_bandpass, None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(bandpass_normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]


        return focus_value, image_out

    def fft(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)

        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        # size = 30
        # # Apply FFT to the entire grayscale image
        # f = np.fft.fft2(gray_image)
        # fshift = np.fft.fftshift(f)

        # # Logarithmic scaling for better visualization of the magnitude spectrum
        # magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

        # # Determine the center of the frequency spectrum
        # rows, cols = gray_image.shape
        # cX, cY = cols // 2, rows // 2

        # # Zero out the low-frequency components around the center
        # fshift[cY - size:cY + size, cX - size:cX + size] = 0

        # # Apply the inverse FFT to focus on high-frequency components
        # f_ishift = np.fft.ifftshift(fshift)
        # recon = np.fft.ifft2(f_ishift)
        # recon = np.abs(recon)

        # # Calculate the focus value as the variance of the high-frequency components in the ROI
        # focus_value = np.var(recon)

        # # Normalize the magnitude spectrum for visualization
        # normalized_spectrum = cv2.normalize(
        #     magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # image_out = cv2.cvtColor(normalized_spectrum, cv2.COLOR_GRAY2RGB)
        
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        # Apply a window function to reduce edge effects
        window = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
        gray_windowed = gray * window
        # Compute the FFT
        f = np.fft.fft2(gray_windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        # Ground zero low frequencies
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        low_freq_size = 10 
        magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                    center_x - low_freq_size:center_x + low_freq_size] = 0
        # Focus measure: sum of magnitude spectrum values
        focus_value = np.var(magnitude_spectrum[y0:y1, x0:x1])
        magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
        image_out = cv2.normalize(magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(image_out.astype(np.uint8), cv2.COLOR_GRAY2BGR)[y0:y1, x0:x1]
        return focus_value, image_out
    
    def mix_sobel(self, image_in): 
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        combined_gradients = gradient_magnitude + np.abs(sobel_xy)
        focus_value = np.var(combined_gradients[y0:y1, x0:x1])
        
        normalized_image = cv2.normalize(
            combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]

        return focus_value, image_out        

    def sobel_laplacian(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx * width - self.w / 2)
        y0 = int(self.cy * height - self.h / 2)
        x1 = int(self.cx * width + self.w / 2)
        y1 = int(self.cy * height + self.h / 2)

        # Convert to grayscale
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        # Apply Sobel filter
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Apply Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Combine Sobel and Laplacian
        combined = sobel_magnitude + np.abs(laplacian)

        # Compute focus value
        focus_value = np.var(combined[y0:y1, x0:x1])

        # Normalize for visualization
        normalized_image = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]
        return focus_value, image_out

    def wavelet(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx * width - self.w / 2)
        y0 = int(self.cy * height - self.h / 2)
        x1 = int(self.cx * width + self.w / 2)
        y1 = int(self.cy * height + self.h / 2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        roi = gray_image[y0:y1, x0:x1]

        rows, cols = roi.shape
        if rows % 2 != 0:
            roi = roi[:-1, :]
            rows -= 1
        if cols % 2 != 0:
            roi = roi[:, :-1]
            cols -= 1

        # Perform single-level Haar wavelet transform manually
        LL = (roi[0::2, 0::2] + roi[0::2, 1::2] + roi[1::2, 0::2] + roi[1::2, 1::2]) / 4
        LH = (roi[0::2, 0::2] - roi[0::2, 1::2] + roi[1::2, 0::2] - roi[1::2, 1::2]) / 4
        HL = (roi[0::2, 0::2] + roi[0::2, 1::2] - roi[1::2, 0::2] - roi[1::2, 1::2]) / 4
        HH = (roi[0::2, 0::2] - roi[0::2, 1::2] - roi[1::2, 0::2] + roi[1::2, 1::2]) / 4

        # Calculate the energy of the high-frequency components
        high_freq = np.sqrt(LH**2 + HL**2 + HH**2 - LL**2)
        focus_value = np.mean(LH**2 + HL**2 + HH**2)
        high_freq_resized = cv2.resize(high_freq, (cols, rows), interpolation=cv2.INTER_LINEAR)

        # Normalize the image for display
        high_freq_normalized = cv2.normalize(high_freq_resized, None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(high_freq_normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]

        return focus_value, image_out
    
    def lpq(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx * width - self.w / 2)
        y0 = int(self.cy * height - self.h / 2)
        x1 = int(self.cx * width + self.w / 2)
        y1 = int(self.cy * height + self.h / 2)

        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        roi = gray_image[y0:y1, x0:x1]

        # Parameters
        win_size = 7
        rho = 0.95 
        STFTalpha = 1.0 / win_size 


        x = np.arange(-(win_size // 2), win_size // 2 + 1)
        wx = np.hamming(win_size)
        [X, Y] = np.meshgrid(x, x)

        w0 = (1 / win_size) * np.ones((win_size, win_size))
        w1 = np.exp(-2j * np.pi * STFTalpha * X)
        w2 = np.exp(-2j * np.pi * STFTalpha * Y)


        filters = [
            w0,
            w1,
            w2,
            w1 * w2 
        ]


        LPQdesc = np.zeros(roi.shape, dtype=np.uint8)


        for i, filt in enumerate(filters[1:]):
            conv_real = cv2.filter2D(roi.astype(np.float32), -1, np.real(filt))
            conv_imag = cv2.filter2D(roi.astype(np.float32), -1, np.imag(filt))

            LPQdesc += ((conv_real >= 0).astype(np.uint8) << (2 * i))
            LPQdesc += ((conv_imag >= 0).astype(np.uint8) << (2 * i + 1))

        hist, _ = np.histogram(LPQdesc.ravel(), bins=256, range=(0, 256))
        focus_value = entropy(hist + np.finfo(float).eps)

        image_out = cv2.normalize(LPQdesc.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        image_out = cv2.cvtColor(image_out.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        return focus_value, image_out
    
    def combined_focus_measure(self, image_in):
        height, width, _ = image_in.shape

        x0 = int(self.cx*width - self.w/2)
        y0 = int(self.cy*height - self.h/2)
        x1 = int(self.cx*width + self.w/2)
        y1 = int(self.cy*height + self.h/2)
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image_in)
        gray_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

        #sobel
        sobel_x = cv2.cuda.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.cuda.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_xy = cv2.cuda.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        
        sobel_x_copy = cp.asarray(sobel_x.download())
        sobel_y_copy = cp.asarray(sobel_y.download())
        sobel_xy_copy = cp.asarray(sobel_xy.download())        
        
        gradient_magnitude = cp.sqrt(sobel_x_copy**2 + sobel_y_copy**2)
        combined_gradients = gradient_magnitude + cp.abs(sobel_xy_copy)
        sobel_var = cp.var(combined_gradients[y0:y1, x0:x1])

        #fswm
        sigma_low = 2.5
        sigma_high = 3.0
        blur_low = cv2.cuda.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
        blur_high = cv2.cuda.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
        blur_low_copy = cp.asarray(blur_low.download())
        blur_high_copy = cp.asarray(blur_high.download())
        bandpass = blur_low_copy - blur_high_copy       
        fswm_var = np.var(bandpass[y0:y1, x0:x1]) 
        
        focus_value = sobel_var + 0.5*(fswm_var**0.75)
        combined_cpu = cp.asnumpy(combined_gradients)
        normalized_image = cv2.normalize(
            combined_cpu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]
        focus_value = float(cp.asnumpy(focus_value))
        
        return focus_value, image_out 
        
    def combined_focus_measure2(self, image_in):
        height, width, _ = image_in.shape
        x0 = int(self.cx * width - self.w / 2)
        y0 = int(self.cy * height - self.h / 2)
        x1 = int(self.cx * width + self.w / 2)
        y1 = int(self.cy * height + self.h / 2)
        gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

        #Sobel-based focus value
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
        combined_gradients = gradient_magnitude + np.abs(sobel_xy)
        sobel_var = np.var(combined_gradients[y0:y1, x0:x1])

        #Compute FFT-based focus value
        window = np.hanning(gray_image.shape[0])[:, None] * np.hanning(gray_image.shape[1])[None, :]
        gray_windowed = gray_image * window

        f = np.fft.fft2(gray_windowed)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        low_freq_size = 10 
        magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                           center_x - low_freq_size:center_x + low_freq_size] = 0

        fft_var = np.var(magnitude_spectrum[y0:y1, x0:x1])

        focus_value = sobel_var + (0.5*fft_var/(1e5))

        # magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
        normalized_image = cv2.normalize(combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[y0:y1, x0:x1]

        return focus_value, image_out 