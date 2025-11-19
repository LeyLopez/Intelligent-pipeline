import numpy as np
import cv2
from typing import Tuple
import logging

logger = logging.getLogger(__name__)    

class ConvolutionFilters:

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:

        logger.info(f"Applying Gaussian Blur with kernel size {kernel_size}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    


    @staticmethod
    def apply_edge_detection(
        image: np.ndarray, 
        threshold1: int = 100, 
        threshold2: int = 200) -> np.ndarray:
        
        logger.info("Applying Edge Detection using Canny")
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        edges = cv2.Canny(gray_image, threshold1, threshold2)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    


    @staticmethod
    def apply_sharpening(image: np.ndarray)-> np.ndarray:
        logger.info("Applying Sharpening Filter")
        
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image
    


    @staticmethod
    def apply_emboss(image: np.ndarray)->np.ndarray:
        logger.info("Applying Emboss Filter")
        
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]])
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        embossed_image = cv2.filter2D(gray, -1, kernel)

        embossed_image = cv2.normalize(embossed_image, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.cvtColor(embossed_image, cv2.COLOR_GRAY2BGR)
    


    @staticmethod
    def apply_all_filters(image: np.ndarray)-> dict:
        logger.info("Applying all convolution filters")
        
        filters = {
            "original": image,
            "gaussian_blur": ConvolutionFilters.apply_gaussian_blur(image),
            "edge_detection": ConvolutionFilters.apply_edge_detection(image),
            "sharpening": ConvolutionFilters.apply_sharpening(image),
            "emboss": ConvolutionFilters.apply_emboss(image)
        }
        
        return filters