import torch
import cv2
import yaml
import os
import traceback
import numpy as np
import time
from ultralytics import YOLO
import math

##### logging Info #####
from loguru import logger
import datetime
import random



class Distance_Estimator:
    """
    A class that encapsulates the model loading, warm-up, and inference operations.
    This approach ensures the model is loaded into memory once and used for multiple inferences.
    """
    def __init__(self, config_path='', model_path='', warmup_runs=3):
        """"
        Initialize the model.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Confidence threshold for detections.
            warmup_runs (int): Number of dummy inference runs to warm up the model.
        """
        
        #### Model Loading ####
        self.model_path               = model_path
        self.model                    = YOLO(self.model_path)
        self.parameters               = self.load_config(config_path)

        # Initialize logger
        log_dir = self.parameters['paths']['logs_dir']
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"distance_estimator_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger.add(log_file, rotation="10 MB", retention="10 days", enqueue=True)
        self.logger = logger
        
        ## Dummy Image for warmup 
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            _ = self.model.predict(source=dummy_image, verbose=False)
            
        logger.info(f"Model loaded from {self.model_path} ")


    def load_config(self, config_path: str) -> dict:
        '''
        Load configuration from a YAML file.
        Inputs:
            - config_path (str): Path to the YAML configuration file.
        Outputs:
            - config (dict): Dictionary containing the configuration parameters.
        '''
    
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"YAML config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
                    
        return config


    def pre_process(self, image: np.ndarray, previous_detections = []):
        """
        Pre-process the input image before running inference.

        Args:
            image (np.ndarray): The input image array.

        Returns:
            image (np.ndarray): The pre-processed image array.
        """
        tic = time.time()   
        toc = time.time()
        logger.info(f"Pre-processing completed in {toc - tic:.2f} seconds")
        return image
    
       
    def infer(self, image: np.ndarray, stream=False):
        """
        Inference function to get inference from YOLO
        
        Args: Image
        Returns: YOLO detection objects
        """

        logger.info("Starting inference...")
        tic = time.time()   
        det_results = self.model.predict(source  = image,
                                         stream  = stream, 
                                         save    = self.parameters['yolo_parameters']['save'],
                                         conf    = self.parameters['yolo_parameters']['confidence'], 
                                         verbose = False,
                                         iou     = self.parameters['yolo_parameters']['iou'],
                                            ) 
        toc = time.time()
        inference_time = toc - tic
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        return det_results
    

    def calculate_distance(self, camera_height, hf, hi, focal_length, vanishing_point_offset, distance_offset=0):
        """
        Calculates the distance based on YOLO's Bounding Boxes.
        
        Args:
            - camera_height: Image on which to draw the bounding boxes
            - hf_y: Vanishing point height in pixels
            - hi_y: center_y of the object in pixels
            - focal_length: Focal length of the camera in pixels
            - scale: Scale factor for distance calculation to accumulate blind spot (default is 0, which means no scaling)

        Returns:
            - (float): Distance in meters

        """
        tic = time.time()
        distance = (camera_height/((hf - hi) + vanishing_point_offset )) * focal_length
        distance = distance - distance_offset  # Adjusting distance with offset
        toc = time.time()
        
        logger.info(f"Distance calculated in {toc - tic:.2f} seconds")
        
        return distance

    def annotate_image(self, image, annotations):
        """
        Draws bounding boxes on the image based on YOLO annotations.

        Args:
            - image (np.ndarray): Image on which to draw the bounding boxes
            - annotations (List[List[int, x, y, w, h, distance]]): Annotations in YOLO format

        Returns:
            - np.ndarray: Image with bounding boxes drawn
        """
        
        tic = time.time()
        
        BOX_COLOR     =  (0, 0, 255)     # Red box
        TEXT_COLOR    =  (255, 255, 255)  # White text
        FONT          =  cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE    =  0.7
        THICKNESS     =  2
        TEXT_PADDING  =  5
            
        
        class_names = self.parameters['yolo_parameters']['class_names']
        
        for ann in annotations:
            cx, cy, w, h, conf, class_id, distance = ann
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            x_max = int(cx + w / 2)
            y_max = int(cy + h / 2)
            
            txt = f'{class_names[int(class_id)]}: {distance:.2f} m'
            
            (text_w, text_h), _  =  cv2.getTextSize(txt, FONT, FONT_SCALE, THICKNESS) ## text width and height
            bg_topleft           =  (x_min, y_min - text_h - TEXT_PADDING * 2)
            bg_bottomright       =  (x_min + text_w + TEXT_PADDING * 2, y_min)
            bg_topleft           =  (max(bg_topleft[0], 0), max(bg_topleft[1], 0)) ## Prevent negative coordinates

            # Draw filled rectangle for text background
            cv2.rectangle(image, bg_topleft, bg_bottomright, BOX_COLOR, -1)

            # Put text over red box
            text_origin = (bg_topleft[0] + TEXT_PADDING, bg_bottomright[1] - TEXT_PADDING)
            cv2.putText(image, txt, text_origin, FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        
        toc = time.time()
        logger.info(f"Image annotated in {toc - tic:.2f} seconds")

        return image
        
    
    
    def main(self, image: np.ndarray):  #TODO 7: You will receive image and the detection results from previous model (if this model is not the first one)
        """
        Main function to run inference and all the post-processing steps.

        Args:
            image (np.ndarray): The input image array.
            previous_detections (list): A list of detection results from the previous model.
        Returns:
            annotated_image (np.ndarray): The image annotated with detection results.
            Results (list): A list of detection results, each containing bounding boxes,
                                confidence scores, classes, and labels.
        """

        try:    
            # Initialize storage for YOLO results based on device
            tic   = time.time()
            
            image = self.pre_process(image)
            
            if self.parameters['flags']['use_gpu']:
                results_storage = torch.empty((0, 7), device='cuda')  # [x, y, w, h, conf, class]
            else:
                results_storage = np.empty((0, 7), dtype=np.float32)
            
             
            results = self.infer(image)
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0: ### Check if there are any boxes detected
                    for box in result.boxes:
                        if self.parameters['flags']['use_gpu']:
                            bbox = box.xywh[0]
                            class_id = int(box.cls[0])
                        else:
                            bbox = box.xywh[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().item())
            
                        distance = self.calculate_distance(
                                                            camera_height           =  self.parameters['distance_parameters']['camera_height'],
                                                            hf                      =  self.parameters['distance_parameters']['hf'],
                                                            hi                      =  image.shape[0] - bbox[1],
                                                            focal_length            =  self.parameters['distance_parameters']['focal_length'],
                                                            vanishing_point_offset  =  self.parameters['distance_parameters']['vanishing_point_offset'],
                                                            distance_offset         =  self.parameters['distance_parameters']['dist_offset']
                                                            )   
                        
                        if self.parameters['flags']['use_gpu']:
                            result_row      = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3], box.conf[0], class_id, distance], device='cuda').unsqueeze(0)
                            results_storage = torch.cat((results_storage, result_row), dim=0)
                        
                        else:
                            result_row      = np.array([[bbox[0], bbox[1], bbox[2], bbox[3], box.conf[0].cpu().item(), class_id, distance]], dtype=np.float32)
                            results_storage = np.vstack((results_storage, result_row))

            
            if self.parameters['flags']['save_results']:    
                annotated_image = self.annotate_image(image, results_storage)
                
                toc = time.time()
                logger.info(f"Total processing time: {toc - tic:.2f} seconds")
                return results_storage, annotated_image
            
            else:
                
                toc = time.time()
                logger.info(f"Total processing time without saving: {toc - tic:.2f} seconds")
                return results_storage, None
        
        except Exception as e:
            self.logger.error(f"Error in main function: {e}")
            self.logger.debug(traceback.format_exc())
            return None, None
                        


# Example usage:
if __name__ == "__main__":

       
    def load_images_from_dir(image_dir: str):
        """
        Get paths of images from a specified directory.

        Args:
            image_dir (str): Directory containing images.

        Returns:
            List[str]: List of image file paths.
        """
        image_paths = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, filename)
                image_paths.append(img_path)
        return image_paths
    

    model = Distance_Estimator(
                               config_path = r"/home/gpuadmin/ALI/X_UP/xup_project/distance_estimator_config.yaml", 
                               model_path  = r'/home/gpuadmin/ALI/X_UP/xup_project/weights/YOLO8s_07_25.pt', 
                               warmup_runs = 3,
                               )
    
    
    images_path = load_images_from_dir(r"/home/gpuadmin/ALI/X_UP/xup_project/distance_est_imgs")
    save_dir = r"/home/gpuadmin/ALI/X_UP/xup_project/distance_est_results/"
    os.makedirs(save_dir, exist_ok=True)
    
    
    for img_path in images_path:
        image = cv2.imread(img_path)
        results, annotated_image = model.main(image)
        if annotated_image is not None:
         cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), annotated_image)


