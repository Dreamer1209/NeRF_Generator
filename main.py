# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from stylenerf2 import Generator
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch import nn,optim
from shutil import move
import tqdm
import glob
from PIL import Image
from kornia.filters import filter2d
from collections import OrderedDict
from torch.nn.init import kaiming_normal_

data_dir = "./data"

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(17, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 512)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        return x

class LoadData(dataset.Dataset):

    def __init__(self,file_list, istrain = True ):
        super(LoadData, self).__init__()
        self.file_list = file_list
        self.istrain = istrain


    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        poses = np.load(os.path.join(data_dir, "train.npy"))
        if self.istrain:
            pose = poses[idx-1]
        else:
            poses = np.load(os.path.join(data_dir, "val.npy"))
            pose = poses[idx-1]
        img_arr = np.array(Image.open(img_path))
        return torch.from_numpy(pose).float(), torch.from_numpy(img_arr).float()

    def __len__(self):
        return len(self.file_list)

def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
def exists(val):
    return val is not None

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

class RGBBlock(nn.Module):
    def __init__(self, scale, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = scale, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        '''
        x: feature map; latent: 来自w空间的latent code
        '''
        # print(x.shape)
        # print(latent.shape)
        #由w空间得到的512 维 latent code再做一次linear,长度为待融合的feature map channel数的两倍
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        # print(style.shape)
        shape = [-1, 2, x.size(1), 1, 1]
        # style 按通道数一分为二，前半部分表示新的mean(paper中的y_{s,i}),std(paper中的y_{b,i})
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        # rescale & reshift
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class Adain(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(Adain, self).__init__()


        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNormLayer()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm2d(channels)
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, style):

        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        # IN step 1
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        # 将latent code赋予到IN运算后的feature map上，实现风格的”注入“

        if self.style_mod is not None:
            x = self.style_mod(x, style)

        return x

class Synthesis(nn.Module):
    def __init__(self,  latent_size=512, input_channels=512,channels=3):
        super(Synthesis, self).__init__()
        # self.initial_block = nn.Parameter(torch.ones((input_channels, 28, 21)))
        self.input_channels = input_channels
        self.synthesisblock1 = GeneratorBlock(2, latent_size, int(input_channels), int(input_channels / 2),
                                              upsample=False, rgba=False)
        self.synthesisblock2 = GeneratorBlock(7, latent_size, int(input_channels / 2), int(input_channels / 4),
                                              upsample=True, rgba=False)
        self.synthesisblock3 = GeneratorBlock(3, latent_size, int(input_channels / 4),int(input_channels / 8),
                                              upsample=True, rgba=False)
        self.synthesisblock4 = GeneratorBlock(3, latent_size, int(input_channels / 8), int(input_channels / 16),
                                              upsample=True, rgba=False)
        self.synthesisblock5 = GeneratorBlock(2, latent_size, int(input_channels / 16), int(input_channels / 32),
                                              upsample=True, rgba=False)
        self.synthesisblock6 = GeneratorBlock(3, latent_size, int(input_channels / 32), int(input_channels / 64),
                                              upsample=False, rgba=False)
        self.torgb = nn.Conv2d(input_channels, channels, 3, padding=1)

    def forward(self, styles):
        rgb = None
        x = nn.Parameter(torch.ones((styles.shape[0],self.input_channels, 4, 3))).cuda()
        # print(x.shape)
        # styles = styles.transpose(0, 1)
        # print("s0",x.shape)
        style = styles
        x, rgb = self.synthesisblock1(x, rgb, style)
        # print("s1", x.shape)
        style = styles
        x, rgb = self.synthesisblock2(x, rgb, style)
        # print("s2", x.shape)
        style = styles
        x, rgb = self.synthesisblock3(x, rgb, style)
        # print("s3", x.shape)
        style = styles
        x, rgb = self.synthesisblock4(x, rgb, style)
        # print("s4", x.shape)
        style = styles
        x, rgb = self.synthesisblock5(x, rgb, style)
        # print("s5", x.shape)
        x, rgb = self.synthesisblock6(x, rgb, style)
        # print("s6", x.shape)

        return rgb

class SynthesisBlock(nn.Module):
    def __init__(self, scale, latent_dim, input_channels, output_channles,upsample=True):
        super().__init__()
        self.scale = scale
        self.upsample = upsample
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False) if upsample else None
        self.adain1 = Adain(channels=input_channels, dlatent_size=latent_dim, use_wscale=True, use_pixel_norm=True,
                            use_instance_norm=True,use_styles=True)
        self.adain2 = Adain(channels=input_channels, dlatent_size=latent_dim, use_wscale=True, use_pixel_norm=True,
                            use_instance_norm=True, use_styles=True)
        self.conv1 = nn.Conv2d(input_channels, output_channles, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channles, output_channles, 3, padding=1)

        self.activation = leaky_relu()


    def forward(self, x,style):
        x = self.adain1(x,style)
        x = self.conv1(x)
        x = self.activation(x)
        # x = self.conv2(x)
        # x = self.activation(x)
        if self.upsample:
            x = self.upsample(x)

        return x

class GeneratorBlock(nn.Module):
    def __init__(self, scale, latent_dim, input_channels, filters, upsample=True, rgba=False):
        super().__init__()
        self.scale = scale
        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        # self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        # self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(scale, latent_dim, filters, upsample, rgba)


    def forward(self, x, prev_rgb, istyle):

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        if self.upsample:
            x = self.upsample(x)

        return x, rgb

class Net(nn.Module):
    def __init__(self, latent_dim, input_channels, channels):
        super().__init__()
        self.fcnet = MLP().cuda()
        self.synthesis = Synthesis(latent_size=latent_dim, input_channels=input_channels, channels=channels).cuda()

    def forward(self, x):
        x = self.fcnet(x)
        x = self.synthesis(x)
        return x

def train():
    print("Model start!")
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device_ids', type=str, default='0', help='gpu id')
    arg('--root', type=str, default='./runs', help='root path')
    arg('--batch_size', type=int, default=8)
    arg('--image_size', type=int, default=256)
    arg('--n_epochs', type=int, default=2000)
    arg('--lr', type=float, default=1e-4, help='learning rate')
    arg('--num_worker', type=int, default=0)  # 数据载入时间<训练时间时, 无需多线程; 最大线程数=CPU内核数
    arg('--num_classes', type=int, default=0)  # 类别包括背景
    arg('--model_name', type=str, default='SwinUNet')  # 需修改
    args = parser.parse_args()

    num_train = 16
    num_val = 4

    def make_loader(file_names, shuffle=True, train=True, batch_size=None):
        return DataLoader(
            dataset=LoadData(file_names, istrain=train),
            shuffle=shuffle,
            num_workers=args.num_worker,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )
    train_file_names = glob.glob(os.path.join(data_dir, 'train', '*.png'))
    valid_file_names = glob.glob(os.path.join(data_dir, 'val', '*.png'))
    print(len(valid_file_names))
    train_dataset = make_loader(train_file_names, shuffle=True, train=True, batch_size=min(args.batch_size,num_train))
    valid_dataset = make_loader(valid_file_names, shuffle=True, train=False, batch_size=min(args.batch_size,num_val))

    net = Net(latent_dim=512, input_channels=512,channels=3)
    lr = args.lr
    iternum = 0
    mseloss = torch.nn.MSELoss(reduction='mean')
    print(len(valid_dataset))

    for epoch in range(args.n_epochs + 1):
        net.train()
        # tq = tqdm.tqdm(total=(len(train_dataset) * args.batch_size))  # 设置进度条长度
        # tq.set_description('Epoch {}, lr {:.6f}'.format(epoch, lr))  # 设置进度条格式
        td = train_dataset
        vd = valid_dataset
        iternum += 1

        try:
            for i, (inputs, targets) in enumerate(td):
                # print("inputs",inputs.shape)
                # print("tartget",targets.shape)
                inputs = cuda(inputs)
                targets = targets.permute(0,3,2,1)
                with torch.no_grad():
                    targets = cuda(targets)
                outputs = net(inputs)

                loss = mseloss(outputs,targets)
                optimizer = optim.Adam(net.parameters(), lr=0.001)

                optimizer.zero_grad()
                batch_size = inputs.size(0)  # 获取batch_size大小
                loss.backward()
                optimizer.step()
            print('Train Epoch: {} Loss: {:.6f}'.format(epoch , loss))
        except KeyboardInterrupt:
            # tq.close()
            print('Ctrl+C, saving snapshot')
            print('done.')
            return
        if epoch % 100 == 0 and epoch:
            val_loss = 0

            for i, (inputs, targets) in enumerate(vd):

                inputs = cuda(inputs)
                targets = targets.permute(0, 3, 2, 1)
                targets = cuda(targets)
                outputs = net(inputs)
                val_loss +=  mseloss(outputs,targets)
            print("Val loss: ",val_loss)
            torch.save(net.state_dict(), "./save/"+str(epoch)+"_model.pth")

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

