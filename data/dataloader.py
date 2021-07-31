from torch.utils import data
from torch.utils.data import DataLoader
from .dataset import CTLung_data, COVID19_data
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensor
import torchvision.transforms as transforms


def get_segment_loader(data_dir, img_size=(256, 256), batch_size=8, shuffle=False, num_workers=4, sampler=None):
    """Create dataloader for the segmentation stage
    
    Args:
        data_dir (tuple): contain directories to the dataset
        img_size (tuple, optional): [description]. Defaults to (256, 256).
        batch_size (int, optional): [description]. Defaults to 8.
        shuffle (bool, optional): [description]. Defaults to False.
        num_worker (int, optional): [description]. Defaults to 4.
        sampler ([type], optional): [description]. Defaults to None.
    """
    # define tranformations used to process data before fit to your model
    segment_trans = Compose([
        Resize(img_size[0], img_size[1]),
        ToTensor()
    ])

    # define dataset and dataloader
    lung_dataset = CTLung_data(data_dir, segment_trans)
    lung_loader = DataLoader(lung_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, sampler=sampler)

    return lung_loader


def get_classify_loader(data_dir, label_dir, mode='train', image_size=(192, 288), batch_size=16, num_workers=4, sampler=None):
    """[summary]

    Args:
        data_dir ([type]): [description]
        label_dir ([type]): [description]
        mode (str, optional): [description]. Defaults to 'train'.
        image_size (tuple, optional): [description]. Defaults to (192, 288).
        batch_size (int, optional): [description]. Defaults to 16.
        num_workers (int, optional): [description]. Defaults to 4.
        sampler ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if mode == 'train':
        classify_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, scale=(1.0, 1.1), shear=0),
            #  transforms.RandomRotation(5),
            transforms.Resize(image_size[:2]),
            transforms.ToTensor()
        ])
    else:
        classify_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size[:2]),
            transforms.ToTensor()
        ])

    covid_dataset = COVID19_data(
        label_dir, data_dir, transform=classify_trans)
    
    if mode == 'train' and sampler is None:
        covid_loader = DataLoader(
            covid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    elif mode == 'train':
        covid_loader = DataLoader(
            covid_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        covid_loader = DataLoader(
            covid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return covid_loader
