import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from lib.dataset import AFLWDataset


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(
            landmarks_batch[i, 0, landmarks_batch[i, 2] != 0].numpy() +
            i * im_size + (i + 1) * grid_border_size,
            landmarks_batch[i, 1, landmarks_batch[i, 2] != 0].numpy() +
            grid_border_size,
            s=10,
            marker='.',
            c='r')

    plt.title('Batch from dataloader')

    plt.savefig('batch_aflw.png')


composed = transforms.Compose([transforms.ToTensor()])
transforms.Normalize((129.1863, 104.7624, 93.5940), (1.0, 1.0, 1.0))
aflw = AFLWDataset('./data/aflw_cropped_label.csv',
                   './data/aflw_processed_data', composed)

for i in range(4):
    sample = aflw[i]
    print(sample[0].size(), sample[1].size())

aflw_loader = DataLoader(
    aflw, 8 // 2, collate_fn=AFLWDataset.collate_method, shuffle=True)

sample = iter(aflw_loader).next()
print(sample[0].size(), sample[1].size())
show_landmarks_batch(sample)
