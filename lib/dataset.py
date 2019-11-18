import os
# Ignore warnings
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")


class AFLWDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, val=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        if val:
            self.landmarks_frame = self.landmarks_frame.loc[
                self.landmarks_frame['val'] == 1]
        else:
            self.landmarks_frame = self.landmarks_frame.loc[
                self.landmarks_frame['val'] == 0]
        self.landmarks_frame = self.landmarks_frame.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.landmarks_frame)

    def sync_flipped_landmarks(self, image, landmarks):
        lookup = np.array([
            5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 16, 15, 14, 13, 12, 19, 18,
            17, 20
        ])
        flipped_img = transforms.functional.hflip(image)
        flipped_landmarks = np.copy(landmarks)
        flipped_landmarks[
            0, :] = (image.size[1] - landmarks[0, :]) * (landmarks[0, :] != 0)
        flipped_landmarks = flipped_landmarks[:, lookup]

        return flipped_img, flipped_landmarks

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.loc[idx, 'filepath'])
        image = self.pil_loader(img_name)
        landmarks = np.array([self.landmarks_frame.iloc[idx, 3:66]]).reshape(
            -1, 3).T.astype('float')
        # landmarks[:, 1] = image.size[0] - landmarks[:, 1]
        flip_img, flip_lm = self.sync_flipped_landmarks(image, landmarks)
        if self.transform:
            image = self.transform(image)
            flip_img = self.transform(flip_img)
        images = torch.stack([image, flip_img], 0)
        labels = torch.stack(
            [torch.FloatTensor(landmarks),
             torch.FloatTensor(flip_lm)], 0)

        return images, labels

    @staticmethod
    def collate_method(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]

        data = torch.cat(data, 0)
        target = torch.cat(target, 0)

        return data, target


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        x = tensor.new(*tensor.size())
        x[:,
          0, :, :] = (tensor[:, 0, :, :] * self.std[0] + self.mean[0]) / 255.0
        x[:,
          1, :, :] = (tensor[:, 1, :, :] * self.std[1] + self.mean[1]) / 255.0
        x[:,
          2, :, :] = (tensor[:, 2, :, :] * self.std[2] + self.mean[2]) / 255.0

        return x
