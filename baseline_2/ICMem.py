import json
import os
from PIL import Image
from torch.utils.data import Dataset

import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ICMemDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, baseline_file_high=None, baseline_file_low=None):
        """
        Args:
            root_dir (str): Path to the dataset root.
            train (bool): True for training, False for validation.
            transform (callable, optional): Transformations applied.
            Baseline_file (str): JSON file specifying training images.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = [] 
    
        # Ensure label file is provided
        assert baseline_file_high is not None, "Baseline file must be specified for training dataset."
        assert baseline_file_low is not None, "Baseline file must be specified for training dataset."

        self.train_dir = os.path.join(root_dir, "ICMem/train")
        baseline_path_high = os.path.join(self.train_dir, baseline_file_high)
        baseline_path_low = os.path.join(self.train_dir, baseline_file_low)

        # Load images from JSON file
        with open(baseline_path_high, 'r') as f:
            image_list_high = json.load(f)  # List of image paths (e.g., "dog/n02099712_1743.JPEG")

        with open(baseline_path_low, 'r') as f:
            image_list_low = json.load(f)

        if self.train:
            self.mem = [1] * len(image_list_high) + [0] * len(image_list_low)
            image_list = image_list_high + image_list_low
            
            # Process the image paths
            for img in image_list:
                class_name = img.split('/')[0]  # Extract class from path
                img_path = os.path.join(class_name, img.split('/')[1])
                self.data.append(img_path)  # Store full path relative to 'train/'
                self.targets.append(class_name)  # Store class name
        
        else:
            # Load validation set: all images in 'val/' (no filtering, split happens later)
            self.val_dir = os.path.join(root_dir, "ICMem/val")
            for class_name in os.listdir(self.val_dir):  # Sort to maintain order
                class_dir = os.path.join(self.val_dir, class_name)
                if os.path.isdir(class_dir):  # Ensure it's a directory
                    for img_name in os.listdir(class_dir):  # Sort files to keep order
                        img_path = os.path.join(class_name, img_name)  # Relative path
                        self.data.append(img_path)
                        self.targets.append(class_name)
            
            # Add additional training images (not included in json) to validation set
            # For 500 training images per class, approx. 215 images per class are added to maintain 70/30 split
            image_list = image_list_high + image_list_low

            for class_name in os.listdir(self.train_dir):  # Sort to maintain order
                class_dir = os.path.join(self.train_dir, class_name)
                if os.path.isdir(class_dir):  # Ensure it's a directory
                    count = 0
                    for img_name in os.listdir(class_dir):  # Sort files to keep order
                        img_path = os.path.join(class_name, img_name)  # Relative path
                        if img_path not in image_list:
                            self.data.append(img_path)
                            self.targets.append(class_name)
                            count += 1
                        if count == 165:
                            break

            

        # Unique class names and mapping
        self.classes = sorted(set(self.targets))  # Maintain class order
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Convert class names to numerical labels
        self.targets = [self.class_to_idx[c] for c in self.targets]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image.
            label (int): Numerical label corresponding to the class.
        """
        if self.train:
            img_path = os.path.join(self.train_dir, self.data[idx])
        else:
            img_path = os.path.join(self.val_dir if "ILSVRC" in self.data[idx] else self.train_dir, self.data[idx])
            
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __repr__(self):
        """Return dataset summary similar to torchvision.datasets.CIFAR10"""
        split = "Train" if self.train else "Test"
        return f"""Dataset ICMemDataset
            Number of datapoints: {len(self.data)}
            Root location: {self.root_dir}
            Split: {split}
            Transforms: {self.transform}
            Classes: {self.classes}
            Class indices: {self.class_to_idx}
        """



