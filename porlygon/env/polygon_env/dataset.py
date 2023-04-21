import glob
import os

from torch.utils.data import Dataset
from torchvision.io import read_image

class EnvironmentDataset(Dataset):
    """
    A PyTorch Dataset that loads input images from disk.
    
    Args:
        img_dir (str): The path to the directory containing the input images.
        transforms (torchvision.transforms.Compose): A list of transforms to apply to the input images (default: None).
    """
    def __init__(self, img_dir, transforms=None):
        self.dir = img_dir
        self.transforms = transforms
        self.img_files = glob.glob(os.path.join(self.dir, "*.jpg"))

    def __getitem__(self, idx):
        """
        Returns the input image at the specified index.
        
        Args:
            idx (int): The index of the input image to return.
            
        Returns:
            torch.Tensor: The input image as a PyTorch tensor.
        """
        image = read_image(self.img_files[idx])
        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        """
        Returns the number of input images in the dataset.
        
        Returns: 
            int: The number of input images in the dataset.
        """
        return len(self.img_files)
