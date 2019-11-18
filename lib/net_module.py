import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import AFLWDataset, UnNormalize
from .model import MMSTN, TotalLoss


class Net3DMMSTN(pl.LightningModule):
    def __init__(self, hparams):
        super(Net3DMMSTN, self).__init__()
        self.hparams = hparams
        self.net = MMSTN(
            vgg_faces_weight_path=self.hparams.vgg_faces_path,
            tutte_embedding_path=self.hparams.tutte_emb_path)
        self.criterion = TotalLoss()
        self.last_batch = None
        self.matlab_vgg_mean = (129.1863, 104.7624, 93.5940)
        self.pytorch_vgg_mean = (131.4538, 103.9875, 91.4623)
        self.oxford_vgg_mean = (131.45376586914062, 103.98748016357422,
                                91.46234893798828)
        self.unorm = UnNormalize(self.oxford_vgg_mean, (1.0, 1.0, 1.0))

    def forward(self, input, extra=False):
        return self.net.forward(input, extra)

    def training_step(self, batch, batch_nb):
        input, label = batch
        sel, alpha, predgrid = self.forward(input)
        loss, l1, l2, l3, l4 = self.criterion(sel, label, alpha, predgrid)

        log_dict = {
            'train/loss': loss,
            'train/euclidean_loss': l1,
            'train/sse_loss': l2,
            'train/siamese_loss': l3,
            'train/symmetry_loss': l4
        }

        return {'loss': loss, 'log': log_dict}

    def configure_optimizers(self):
        minus_theta = list(list(
            self.net.children())[0].children())[:-1] + list(
                self.net.children())[1:]
        param_list = [{'params': m.parameters()} for m in minus_theta]
        param_list.append({
            "params": self.net.vgg_localizer.fc8.weight,
            "lr": self.hparams.learning_rate * 4
        })
        param_list.append({
            "params": self.net.vgg_localizer.fc8.bias,
            "lr": self.hparams.learning_rate * 8
        })

        return torch.optim.SGD(param_list, lr=self.hparams.learning_rate)

    def configure_optimizers_not_used(self):
        minus_theta = [list(self.net.children())[0]] + list(
            self.net.children())[2:]
        param_list = [{'params': m.parameters()} for m in minus_theta]
        param_list.append({
            "params": self.net.fc8.weight,
            "lr": self.hparams.learning_rate * 4
        })
        param_list.append({
            "params": self.net.fc8.bias,
            "lr": self.hparams.learning_rate * 8
        })

        return torch.optim.SGD(param_list, lr=self.hparams.learning_rate)

    def validation_step(self, batch, batch_nb):
        input, label = batch
        sel, alpha, predgrid = self.forward(input)
        loss, l1, l2, l3, l4 = self.criterion(sel, label, alpha, predgrid)
        self.last_batch = batch

        log_dict = {
            'valid/loss': loss,
            'valid/euclidean_loss': l1,
            'valid/sse_loss': l2,
            'valid/siamese_loss': l3,
            'valid/symmetry_loss': l4
        }

        return {'val_loss': loss, 'log': log_dict}

    def draw_landmarks(self, images, landmarks):

        batch_size = images.shape[0]
        landmarks_has_visible = landmarks.shape[1] == 3

        fig, ax = plt.subplots(nrows=1, ncols=batch_size, figsize=(10, 2))

        images = images.transpose(0, 2, 3, 1)

        for i, row in enumerate(ax):
            row.imshow(images[i])
            if landmarks_has_visible:
                mask = landmarks[i, 2] != 0
                row.scatter(
                    landmarks[i, 0, mask],
                    landmarks[i, 1, mask],
                    marker='.',
                    c='r')
            else:
                row.scatter(
                    landmarks[i, 0], landmarks[i, 1], marker='.', c='r')

        for r in ax:
            r.set_yticks([])
            r.set_xticks([])
            r.set_xlim(0, images.shape[2])
            r.set_ylim(images.shape[1], 0)

        plt.tight_layout()

        return fig

    def draw_verts(self, images, verts):

        batch_size = images.shape[0]

        fig, ax = plt.subplots(nrows=1, ncols=batch_size, figsize=(10, 2))

        images = images.transpose(0, 2, 3, 1)

        for i, row in enumerate(ax):
            row.imshow(images[i])
            row.scatter(verts[i, 0], verts[i, 1], marker='.', c='r', s=1)

        for r in ax:
            r.set_yticks([])
            r.set_xticks([])
            r.set_xlim(0, images.shape[2])
            r.set_ylim(images.shape[1], 0)

        plt.tight_layout()

        return fig

    def on_epoch_end(self):
        images, labels = self.last_batch
        with torch.no_grad():
            if self.current_epoch % 5 == 0:
                tensor_dict = self.forward(images, extra=True)

                self.logger.experiment.add_images('images', self.unorm(images),
                                                  self.current_epoch)
                image_np = self.unorm(images).cpu().numpy()

                self.logger.experiment.add_histogram(
                    'input_dist', images + (self.current_epoch / 5),
                    self.current_epoch / 5)
                self.logger.experiment.add_images(
                    'premask', self.unorm(tensor_dict['premask']),
                    self.current_epoch)
                self.logger.experiment.add_images('mask', tensor_dict['mask'],
                                                  self.current_epoch)
                self.logger.experiment.add_images(
                    'pred', self.unorm(tensor_dict['predgrid']),
                    self.current_epoch)

                self.logger.experiment.add_figure(
                    'input_landmarks',
                    self.draw_landmarks(image_np,
                                        labels.cpu().numpy()))
                self.logger.experiment.add_figure(
                    'output_landmarks',
                    self.draw_landmarks(image_np,
                                        tensor_dict['sel'].cpu().numpy()))

                self.logger.experiment.add_figure(
                    '3d_verts',
                    self.draw_verts(image_np,
                                    tensor_dict['3d_verts'].cpu().numpy()))

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {
            'valid/loss':
            avg_loss,
            'valid/euclidean_loss':
            torch.stack(
                [x['log']['valid/euclidean_loss'] for x in outputs]).mean(),
            'valid/sse_loss':
            torch.stack([x['log']['valid/sse_loss'] for x in outputs]).mean(),
            'valid/siamese_loss':
            torch.stack(
                [x['log']['valid/siamese_loss'] for x in outputs]).mean(),
            'valid/symmetry_loss':
            torch.stack(
                [x['log']['valid/symmetry_loss'] for x in outputs]).mean()
        }

        return {'avg_val_loss': avg_loss, 'log': log_dict}

    @pl.data_loader
    def train_dataloader(self):
        composed = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 255.0,
            transforms.Normalize(self.oxford_vgg_mean, (1.0, 1.0, 1.0))
        ])

        aflw = AFLWDataset(
            self.hparams.dataset_csv,
            self.hparams.dataset_root,
            composed,
            val=False)

        aflw_loader = DataLoader(
            aflw,
            self.hparams.batch_size // 2,
            collate_fn=AFLWDataset.collate_method,
            shuffle=True)

        return aflw_loader

    @pl.data_loader
    def val_dataloader(self):
        composed = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 255.0,
            transforms.Normalize(self.oxford_vgg_mean, (1.0, 1.0, 1.0))
        ])

        aflw = AFLWDataset(
            self.hparams.dataset_csv,
            self.hparams.dataset_root,
            composed,
            val=True)

        aflw_loader = DataLoader(
            aflw,
            self.hparams.batch_size // 2,
            collate_fn=AFLWDataset.collate_method,
            shuffle=True)

        return aflw_loader
