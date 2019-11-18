import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy import io as sio
from .vgg_dag_face import vgg_dag_face


def get_vgg16_with_vggfaces_wt(weight_path='../models/vgg_face_dag.pth'):
    net = torchvision.models.vgg16()
    weights = torch.load(weight_path)

    converted_weights = {}
    for x, y in list(zip(net.state_dict(), weights.keys()))[:-2]:
        converted_weights[x] = weights[y]

    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
        nn.Linear(4096, 4096), nn.ReLU(True))
    net.load_state_dict(converted_weights)

    return net


class SplitLayer(nn.Module):
    def __init__(self):
        super(SplitLayer, self).__init__()

    def forward(self, input):
        r = input[:, 0:3]
        t = input[:, 3:5]
        logs = input[:, 5]
        alpha = input[:, 6:16]

        return alpha, r, t, logs


class Model3D(nn.Module):
    def __init__(self, model):
        super(Model3D, self).__init__()
        self.shapePC = torch.from_numpy(model['shapePC']).float()
        self.shapeMU = torch.from_numpy(model['shapeMU']).float()
        self.nverts = model['nverts'][0, 0]
        # self.faces = torch.from_numpy(model['faces'])

    def forward(self, input):
        batch_size = input.size(0)
        self.shapeMU = self.shapeMU.to(input.device)
        self.shapePC = self.shapePC.to(input.device)
        y = self.shapeMU + torch.matmul(self.shapePC, input.t())
        y = y.t().reshape(batch_size, self.nverts, 3).transpose(2, 1)

        return y


class r2R(nn.Module):
    # r2R Axis angle to rotation matrix layer
    # Forwards mode:
    # R = vl_nnr2R(r);
    # r is of size 1 x 1 x 3 x nbatch containing axis angle vectors
    # R is of size 1 x 3 x 3 x nbatch containing rotation matrices
    # Backwards mode:
    # Useful reference for formula:
    # [1] Gallego, Guillermo, and Anthony Yezzi. "A compact formula for the
    #     derivative of a 3-D rotation in exponential coordinates." Journal
    #     of Mathematical Imaging and Vision 51.3 (2015): 378-384.

    def __init__(self):
        super(r2R, self).__init__()

    def forward(self, r):

        batch_size = r.size(0)
        mask = (r == 0).all(dim=1)

        R = torch.zeros(batch_size, 3, 3, device=r.device)

        # Fill in the identity cases
        if torch.sum(mask) != 0:
            idx = torch.arange(0, 3)
            R[mask][:, idx, idx] = 1

        if torch.sum(mask) == batch_size:
            return R

        theta = torch.sqrt(torch.sum(r[~mask, :]**2, 1))
        k = r[~mask, :] / theta.unsqueeze(0).t()

        # Fill in the non-identity cases using the following formula (see [1]):
        # K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];
        # R = eye(3) + sin(theta).*K + (1-cos(theta)).*K*K;

        K = torch.zeros(batch_size, 3, 3, device=r.device)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]

        idx = torch.arange(3)
        I = torch.zeros(batch_size, 3, 3, device=r.device)
        I[:, idx, idx] = 1
        cos_t = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
        sin_t = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
        R[~mask] = I + K * sin_t + (1 - cos_t) * torch.matmul(K, K)

        return R


class Rotate3D(nn.Module):
    def __init__(self):
        super(Rotate3D, self).__init__()

    def forward(self, X, R):
        return torch.matmul(R, X)


class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()

    def forward(self, X_hat):
        return X_hat[:, 0:2, :]


class Exponential(nn.Module):
    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, input):
        return (99 / 100.0)**input


class Scale2D(nn.Module):
    def __init__(self):
        super(Scale2D, self).__init__()

    def forward(self, s, Y):
        return s.unsqueeze(-1).unsqueeze(-1) * Y


class Translate2D(nn.Module):
    def __init__(self):
        super(Translate2D, self).__init__()

    def forward(self, X, T):
        return X + T.unsqueeze(-1)


class Selection(nn.Module):
    def __init__(self, idx):
        super(Selection, self).__init__()
        self.idx = idx

    def forward(self, X):
        return X[:, :, self.idx]


class ResampleGrid(nn.Module):
    def __init__(self):
        super(ResampleGrid, self).__init__()

    def forward(self, X):
        batch_size = X.size(0)
        grid_dim = int(np.sqrt(X.size(2)))

        X0 = (X - 112.0) / 112.0
        X0 = X0.reshape(batch_size, 2, grid_dim, grid_dim)

        newgrid = torch.empty_like(X0)
        newgrid[:, 0, :, :] = X0[:, 0, :, :]
        newgrid[:, 1, :, :] = X0[:, 1, :, :]

        return newgrid.permute(0, 2, 3, 1)


class VisbilityMask(nn.Module):
    def __init__(self, faces):
        super(VisbilityMask, self).__init__()
        self.faces = faces

    def visibiltyNotSelf(self, vertices, faces):
        # Approximate visibility testing only for self occlusions

        # Get the triangle vertices
        print(vertices.shape, faces.shape)
        v0 = faces[:, 0]
        v1 = faces[:, 1]
        v2 = faces[:, 2]

        # Compute the edge vectors
        e0s = vertices[v1, :] - vertices[v0, :]
        e1s = vertices[v2, :] - vertices[v0, :]
        e2s = vertices[v1, :] - vertices[v2, :]

        # Normalize the edge vectors
        e0s_norm = e0s / torch.sqrt(torch.sum(e0s**2, 1)).unsqueeze(-1)
        e1s_norm = e1s / torch.sqrt(torch.sum(e1s**2, 1)).unsqueeze(-1)
        e2s_norm = e2s / torch.sqrt(torch.sum(e2s**2, 1)).unsqueeze(-1)

        # Compute the angles
        angles = torch.empty_like(e0s_norm)
        angles[:, 0] = torch.acos(torch.sum(e0s_norm * e1s_norm, 1))
        angles[:, 1] = torch.acos(torch.sum(e2s_norm * e0s_norm, 1))
        angles[:, 2] = math.pi - (angles[:, 1] - angles[:, 0])

        # Compute the triangle weighted normals
        triangle_normals = torch.cross(e0s, e2s, 1)
        w1_triangle_normals = triangle_normals * angles[:, 0].unsqueeze(-1)
        w2_triangle_normals = triangle_normals * angles[:, 1].unsqueeze(-1)
        w3_triangle_normals = triangle_normals * angles[:, 2].unsqueeze(-1)

        # Initialize the vertex normals
        normals = torch.zeros_like(vertices)
        normals[v0, :] = w1_triangle_normals
        normals[v1, :] = w2_triangle_normals
        normals[v2, :] = w3_triangle_normals

        # Self-occlusions
        visibility = normals[:, 2] >= 0
        print(visibility)

    def batchVisibiltyNotSelf(self, vertices, faces):
        # Approximate visibility testing only for self occlusions

        # Get the triangle vertices
        v0 = faces[:, 0]
        v1 = faces[:, 1]
        v2 = faces[:, 2]

        # Compute the edge vectors
        e0s = vertices[:, v1, :] - vertices[:, v0, :]
        e1s = vertices[:, v2, :] - vertices[:, v0, :]
        e2s = vertices[:, v1, :] - vertices[:, v2, :]

        # Normalize the edge vectors
        e0s_norm = e0s / torch.sqrt(torch.sum(e0s**2, 2)).unsqueeze(-1)
        e1s_norm = e1s / torch.sqrt(torch.sum(e1s**2, 2)).unsqueeze(-1)
        e2s_norm = e2s / torch.sqrt(torch.sum(e2s**2, 2)).unsqueeze(-1)

        # Compute the angles
        angles = torch.empty_like(e0s_norm)
        angles[:, :, 0] = torch.acos(torch.sum(e0s_norm * e1s_norm, 2))
        angles[:, :, 1] = torch.acos(torch.sum(e2s_norm * e0s_norm, 2))
        angles[:, :, 2] = math.pi - (angles[:, :, 1] - angles[:, :, 0])

        # Compute the triangle weighted normals
        triangle_normals = torch.cross(e0s, e2s, 2)
        w1_triangle_normals = triangle_normals * angles[:, :, 0].unsqueeze(-1)
        w2_triangle_normals = triangle_normals * angles[:, :, 1].unsqueeze(-1)
        w3_triangle_normals = triangle_normals * angles[:, :, 2].unsqueeze(-1)

        # Initialize the vertex normals
        normals = torch.zeros_like(vertices)
        normals[:, v0, :] = w1_triangle_normals
        normals[:, v1, :] = w2_triangle_normals
        normals[:, v2, :] = w3_triangle_normals

        # Self-occlusions
        visibility = normals[:, :, 2] >= 0

        return visibility

    def forward(self, X):
        batch_size = X.size(0)
        grid_dim = int(np.sqrt(X.size(2)))

        y = torch.zeros(
            batch_size, 3, grid_dim, grid_dim, dtype=X.dtype, device=X.device)

        self.faces = self.faces.to(X.device)
        mask = self.batchVisibiltyNotSelf(X.transpose(2, 1), self.faces)
        mask = mask.reshape(batch_size, grid_dim, grid_dim)
        mask = torch.repeat_interleave(mask, 3, 0)
        mask = mask.reshape(batch_size, 3, grid_dim, grid_dim).type_as(X)

        return 1 - mask


class Visibility(nn.Module):
    def __init__(self):
        super(Visibility, self).__init__()

    def forward(self, X, V):
        return X * V


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, y, label):
        temp = torch.empty_like(label[:, 0:2, :])
        temp[:, 0, :] = label[:, 2, :]
        temp[:, 1, :] = label[:, 2, :]
        delta = (y - label[:, 0:2, :])**2
        delta = delta * temp
        y = torch.sum(delta)

        return y


class StatisticalPriorLoss(nn.Module):
    def __init__(self):
        super(StatisticalPriorLoss, self).__init__()

    def forward(self, input):
        return torch.sum(input**2)


class SiameseLoss(nn.Module):
    def __init__(self):
        super(SiameseLoss, self).__init__()

    def forward(self, input):
        batch_size = input.size(0)
        idx_e = torch.arange(0, batch_size, 2)
        idx_o = torch.arange(1, batch_size, 2)
        mask = (input[idx_e] != 0) & (input[idx_o] != 0)
        delta = (input[idx_o] - input[idx_e]) * mask
        y = torch.sum(delta**2) / torch.sum(input != 0)

        return y


class SymmetryLoss(nn.Module):
    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def forward(self, input):
        flipped_input = input.flip(3)
        mask = (input != 0) & (flipped_input != 0)
        delta = (input - flipped_input) * mask
        y = torch.sum(delta**2) / torch.sum(input != 0)

        return y


class TotalLoss(nn.Module):
    def __init__(self, weights=[0.8998, 0.1, 0.0001, 0.0001]):
        super(TotalLoss, self).__init__()
        self.euclidean_layer = EuclideanLoss()
        self.sse_layer = StatisticalPriorLoss()
        self.siamese_layer = SiameseLoss()
        self.symmetry_layer = SymmetryLoss()
        self.weights = weights

    def forward(self, pred, label, alpha, predgrid):
        l1 = self.euclidean_layer(pred, label)
        l2 = self.sse_layer(alpha)
        l3 = self.siamese_layer(predgrid)
        l4 = self.symmetry_layer(predgrid)

        total_loss = self.weights[0] * l1 + self.weights[1] * l2 + self.weights[2] * l3 + self.weights[3] * l4
        return total_loss, l1, l2, l3, l4


class MMSTN(nn.Module):
    def __init__(self, vgg_faces_weight_path=None, tutte_embedding_path=None):
        super(MMSTN, self).__init__()
        self.vgg_localizer = vgg_dag_face(vgg_faces_weight_path)
        # LR of weight: x4, LR of bias: x8
        self.vgg_localizer.fc8 = nn.Conv2d(
            4096, 16, kernel_size=[1, 1], stride=(1, 1))
        nn.init.xavier_normal_(self.vgg_localizer.fc8.weight)
        with torch.no_grad():
            self.vgg_localizer.fc8.weight = nn.Parameter(
                self.vgg_localizer.fc8.weight * 0.001)
            # self.vgg_localizer.fc8.bias[3:5] = 112
        # 21 Landmark points for Basel Face Model res=112 (AFLW annotation)
        # 1-6: left to right eyebrows
        # 7-12: left to right eyes
        # 13-17: left ear - nose - right ear
        # 18-20: mouth
        # 21: chin
        self.idx = torch.tensor([
            2488, 2278, 2177, 2190, 2313, 2551, 3282, 3176, 3181, 3202, 3207,
            3325, 7176, 4978, 4535, 4989, 7271, 6879, 6663, 6896, 9127
        ])
        self.split_layer = SplitLayer()
        model_vals = sio.loadmat(tutte_embedding_path)
        self.model_3d = Model3D(model_vals)
        faces = torch.from_numpy(model_vals['faces'].astype('int64') - 1)
        self.r2R_layer = r2R()
        self.rotate_layer = Rotate3D()
        self.project_layer = Projection()
        self.log2scale_layer = Exponential()
        self.scale_layer = Scale2D()
        self.translate_layer = Translate2D()
        self.selection_layer = Selection(self.idx)
        self.grid_layer = ResampleGrid()
        self.visibility_mask_layer = VisbilityMask(faces)
        self.visibility_layer = Visibility()

    def forward(self, input):
        x = self.vgg_localizer(input).squeeze()
        alpha, r, t, logs = self.split_layer(x)
        X = self.model_3d(alpha)
        R = self.r2R_layer(r)
        X_hat = self.rotate_layer(X, R)
        Y = self.project_layer(X_hat)
        s = self.log2scale_layer(logs)
        Y_hat = self.scale_layer(s, Y)
        Y_dhat = self.translate_layer(Y_hat, t)
        sel = self.selection_layer(Y_dhat)
        x = self.grid_layer(Y_dhat)
        x = F.grid_sample(input, x, align_corners=True)
        mask = self.visibility_mask_layer(X_hat)
        predgrid = self.visibility_layer(x, mask)

        return sel, mask, alpha, predgrid, x


if __name__ == "__main__":

    n = MMSTN('../models/vgg_dag_face.pth', '../models/model.mat')
    loss_layer = TotalLoss()
    # print(n)
    # # print(n.theta.weight, n.theta.bias)
    inp = torch.randn(6, 3, 224, 224)
    out = torch.rand(6, 3, 21)
    sel, mask, alpha, predgrid = n(inp)
    loss = loss_layer(sel, out, alpha, predgrid)
    print(loss)
    print(list(list(n.children())[0].children())[-1])
    minus_theta = list(list(n.children())[0].children())[:-1] + list(
        n.children())[1:]
    print(minus_theta)
    print([x.parameters() for x in minus_theta])
    loss[0].backward()
    # print([x.shape for x in otp])
