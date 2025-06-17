import os
import cv2
import numpy as np


class FlowerDataset:
    def __init__(self, folder_dir, transform="histogram", bins=8):  
        self.folder_dir = folder_dir
        self.bins = bins  
        self.labels = []
        self.features = []
        self.label_to_idx = {}  
        self.bins = bins
        self.image_dir = [] 

        for idx, label in enumerate(sorted(os.listdir(folder_dir))):
            label_path = os.path.join(folder_dir, label)

            if os.path.isdir(label_path):
                self.label_to_idx[label] = idx  
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)

                    if transform == "histogram":
                        feature = self.extract_histogram(img_path, self.bins)
                    elif transform == "momentum":
                        feature = self.extract_color_momentum(img_path)
                    elif transform == "dominant":
                        feature = self.extract_dominant_color(img_path)
                    elif transform == "histogram old":
                        feature = self.extract_histogram_old(img_path)

                    self.labels.append(idx)  
                    self.features.append(feature)
                    self.image_dir.append(img_path)

    @staticmethod
    def extract_histogram(img_path, bins=8):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0,180, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        return np.array(histogram, dtype=np.float32)

    @ staticmethod 
    def extract_histogram_old(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        return np.array(histogram, dtype=np.float32)
    

    @staticmethod
    def extract_dominant_color(img_path, k=3):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixels = img.reshape(-1, 3)
        kmeans = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(np.float32(pixels), k, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, kmeans)
        hist = np.bincount(labels.flatten(), minlength=k) / labels.size
        return np.hstack([centers.flatten(), hist])

    @staticmethod
    def extract_color_momentum(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        skew = np.mean(((img - mean) / (std + 1e-6)) ** 3, axis=(0, 1))
        return np.concatenate([mean, std, skew])


    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.image_dir[index]
