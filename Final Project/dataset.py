import cv2 
import os 
import glob 

from torch.utils.data import Dataset
from torchvision.transforms import transforms

class FIVE_dataset(Dataset): 
    def __init__(self, root_path): 
        self.root_path = root_path

        self.images_dir = os.path.join(root_path, "Original/")
        self.masks_dir = os.path.join(root_path, "Ground_truth/")
        
        