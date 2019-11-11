import torch
import torch.nn as nn
import torchvision.transforms as transforms


class VggDagFace(nn.Module):
    def __init__(self):
        super(VggDagFace, self).__init__()
        self.meta = {
            'mean':
            [131.45376586914062, 103.98748016357422, 91.46234893798828],
            'std': [1, 1, 1],
            'imageSize': [224, 224, 3]
        }
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.conv2 = nn.Conv2d(
            96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(
            256,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=(0, 0),
            dilation=1,
            ceil_mode=True)
        self.conv3 = nn.Conv2d(
            256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(
            512,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(
            512,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(
            512,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=0,
            dilation=1,
            ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(
            4096,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(
            4096,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Conv2d(4096, 2622, kernel_size=[1, 1], stride=(1, 1))

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24 = self.relu7(x23)
        prediction = self.fc8(x24)
        return prediction


def compose_transforms(meta,
                       resize=256,
                       center_crop=True,
                       override_meta_imsize=False):
    """Compose preprocessing transforms for model
    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.
    Args:
        meta (dict): model preprocessing requirements
        resize (int) [256]: resize the input image to this size
        center_crop (bool) [True]: whether to center crop the image
        override_meta_imsize (bool) [False]: if true, use the value of `resize`
           to select the image input size, rather than the properties contained
           in meta (this option only applies when center cropping is not used.
    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    if center_crop:
        transform_list = [
            transforms.Resize(resize),
            transforms.CenterCrop(size=(im_size[0], im_size[1]))
        ]
    else:
        if override_meta_imsize:
            im_size = (resize, resize)
        transform_list = [transforms.Resize(size=(im_size[0], im_size[1]))]
    transform_list += [transforms.ToTensor()]
    if meta['std'] == [1, 1, 1]:  # common amongst mcn models
        transform_list += [lambda x: x * 255.0]
    transform_list.append(normalize)
    return transforms.Compose(transform_list)


def vgg_dag_face(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = VggDagFace()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    net = vgg_dag_face('./vgg_dag_face.pth')
    prepoc = compose_transforms(meta=net.meta, center_crop=False)
    print(prepoc)
    x = torch.randn(6, 3, 224, 224)
    o = net(x)

    print(o.shape)
