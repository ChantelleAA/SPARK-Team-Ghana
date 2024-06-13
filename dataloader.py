import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import os
import json
import numpy as np
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

class MultiModalBraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patients = sorted(os.listdir(root_dir))
        self.modalities = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
        self.mask_suffix = '_seg.nii.gz'

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.root_dir, patient_id)
        
        # Load images from the four modalities
        images = []
        for modality in self.modalities:
            img_path = os.path.join(patient_dir, patient_id + modality)
            image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(image)
            images.append(image)
        
        # Stack the images to form a multi-channel tensor
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()

        # Load the segmentation mask
        mask_path = os.path.join(patient_dir, patient_id + self.mask_suffix)
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)
        mask = torch.from_numpy(mask).long()

        sample = {'image': images, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Paths to your data directories
root_dir = "path/to/data/BraTS/BraTS2021_Training_Data"

# Create dataset
dataset = MultiModalBraTSDataset(root_dir)

# Create data loader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through the data loader and print the shapes of the images and masks
for i, batch in enumerate(data_loader):
    images, masks = batch['image'], batch['mask']
    print(f"Batch {i+1}:")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

    # For testing purposes, let's visualize one of the images and masks from the batch
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    # Plot the first image in the batch (first modality)
    plt.subplot(1, 2, 1)
    plt.title('Modality 1')
    plt.imshow(images[0, 0, :, :], cmap='gray')

    # Plot the corresponding mask
    plt.subplot(1, 2, 2)
    plt.title('Mask')
    plt.imshow(masks[0, :, :], cmap='gray')

    plt.show()
    break


#Read the data
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

#Setup dataloader
def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = data.Dataset(data=train_files, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader
