import torch, os
import numpy as np
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.preprocessing import get_transforms, load_dicom

logger = logging.getLogger(__name__)

class OsteoDataset(Dataset):
    """
    Dataset for CXR images with accompanying tabular features

    This dataset reads image file paths derived from `uid` in provided
    DataFrame and returns image tensors, tabular features and labels for training

    Note that images flagged as inverted via the ``photometric`` column set to
    ``'MONOCHROME1'`` will be inverted prior to transformation.
    """
    def __init__(self, image_dir, tabular_df, transforms, type="train"):
        self.img_paths = list(image_dir + tabular_df.uid.astype(str) + '.dcm')
        self.tabular_data = tabular_df
        self.transforms = transforms
        self.type = type
    
    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """Fetches the image, tabular features and label for a given index"""
        try:
            row = self.tabular_data.iloc[idx]
            inverted = (row['photometric'] == 'MONOCHROME1')

            tabular_features = torch.tensor([
                row['patient_age'],
                row['gender_MALE']
            ], dtype=torch.float32)

            image = load_dicom(self.img_paths[idx])
            
            if inverted:
                image = transforms.functional.invert(image)
            if self.transforms:
                image = self.transforms(image)
            
            if self.type == "inference":
                return image, tabular_features, os.path.basename(self.img_paths[idx])
            else:
                label = row['target']
                return image, tabular_features, os.path.basename(self.img_paths[idx]), label
        
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            print(f"File path: {self.img_paths[idx]}")
            raise

def create_dataloader(conf, image_dir, tabular_df, type="train", num_workers=4):
    try:
        dataset = OsteoDataset(
            image_dir=image_dir,
            tabular_df=tabular_df,
            transforms=get_transforms(conf.img_size, type),
            type=type
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")

    try:
        if type == "train":
            dataloader = DataLoader(
                dataset,
                batch_size=conf.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if num_workers > 0 else False,  # Keeps workers alive between epochs
                prefetch_factor=2
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=conf.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if num_workers > 0 else False,  # Keeps workers alive between epochs
                prefetch_factor=2, 
            )
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
    return dataloader