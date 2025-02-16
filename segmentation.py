import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class image_segmentation:
    def __init__(self, checkpoint, config, device):
        self.device_ = device
        
        self.model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def select_point(self, image):
        point = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("choose point prompts for segmentation", image)
                point.append([x, y]) # col, row

        cv2.imshow("choose point prompts for segmentation", image)
        cv2.setMouseCallback("choose point prompts for segmentation", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        point = np.array(point, np.float32)
        labels = np.array([1]*len(point), np.int32)
        return point, labels
        
    def segment_image(self, image):
        point, labels = self.select_point(image)
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
                                                        point_coords = point,
                                                        point_labels = labels,
                                                        multimask_output = False
                                                       )

        return masks