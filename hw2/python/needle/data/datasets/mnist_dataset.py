from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms


    def __getitem__(self, index) -> object:
        img = self.images[index]
        return self.apply_transforms(img), self.labels[index]

    def __len__(self) -> int:
        return self.images.shape[0]
# copy from hw0
def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as img_f:
        img_f.read(4) #skip magic number
        num_images = int.from_bytes(img_f.read(4), 'big') # stored by high(big) endian
        rows = int.from_bytes(img_f.read(4), 'big')
        cols = int.from_bytes(img_f.read(4), 'big')
        
        image_data = img_f.read(num_images * rows * cols)
        X = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X = X.reshape(num_images, rows, cols, 1) #(sample_numbles, H, W, C)
        # X = X.reshape(num_images, rows*cols)
        X /= 255.0 # normalize to [0,1]
        
    with gzip.open(label_filename, 'rb') as lb_f:
        lb_f.read(4)
        num_labels = int.from_bytes(lb_f.read(4), 'big')
        
        lable_data = lb_f.read(num_labels)
        y = np.frombuffer(lable_data, dtype=np.uint8)
    
    # print(X[42])
    return X, y