
### 2.1 Create Dataset and Dataloaders (script mode)

"""
Contains functionality for creating PyTorch DataLoader's for 
image classification data.
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int = 16,
        num_workers: int = NUM_WORKERS
):
        # Follow this style guide from Google style guide for Python
        """Create training and testing DataLoaders.

        Takes in a training directory and a testing directory path and turns them into
        PyTorch Dataset and then into PyTorch Dataloaders.

        Args:
            train_dir (str): Path to training directory.
            test_dir (str): Path to testing directory.
            transform (callable): Optional transform to be applied on a sample.
            batch_size (int, optional): Number of samples per batch. Defaults to 16.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to os.cpu_count().

        Returns:
            A tuple of (train_dataloader, test_dataloader, class_names)
        
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir=path_to_train_dir,
                test_dir=path_to_test_dir,
                transform=transform,
                batch_size=32,
                num_workers=4
            )
        """

        train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                          transform=transform, # transforms to perform on data (images)
                                          target_transform=None) # transforms to perform on labels (if necessary)

        test_data = datasets.ImageFolder(root=test_dir,
                                        transform=transform)

        print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

        class_names = train_data.classes

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True) # enables fast data transfer to CUDA-enabled GPUs 

        test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)

        return train_dataloader, test_dataloader, class_names
