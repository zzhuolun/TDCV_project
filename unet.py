import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # if not middle_channels:
        #     middle_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(p=0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(p=0.2))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, in_channels2=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if not in_channels2:
            in_channels2 = in_channels
        self.double_conv = DoubleConv((in_channels+in_channels2), out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        h_start, h_end = self._crop(x2.shape[2], x1.shape[2])
        w_start, w_end = self._crop(x2.shape[3], x1.shape[3])
        x2 = x2[:, :, h_start:h_end, w_start:w_end]
        return self.double_conv(torch.cat([x2, x1], dim=1))

    def _crop(self, h2, h1):
        assert h2 >= h1
        h_start = int(h2 / 2 + 0.5 - h1 / 2) + 1
        h_end = int(h2 / 2 + 0.5 + h1 / 2)

        h_end -= (h_end - h_start + 1) - h1
        h_start -= 1

        return h_start, h_end


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoder

        # xmask
        self.upx1 = Up(512, 256)
        self.upx2 = Up(256, 256)
        self.upx3 = Up(256, 256, 128)
        self.upx4 = Up(256, 256, 64)
        self.outx = nn.Conv2d(256, 256, kernel_size=1)

        # ymask
        self.upy1 = Up(512, 256)
        self.upy2 = Up(256, 256)
        self.upy3 = Up(256, 256, 128)
        self.upy4 = Up(256, 256, 64)
        self.outy = nn.Conv2d(256, 256, kernel_size=1)

        # zmask
        self.upz1  = Up(512, 256)
        self.upz2  = Up(256, 256)
        self.upz3  = Up(256, 256, 128)
        self.upz4  = Up(256, 256, 64)
        self.outz  = nn.Conv2d(256, 256, kernel_size=1)

        # # idmask
        # self.upid1 = Up(512, 256)
        # self.upid2 = Up(256,128)
        # self.upid3 = Up(128, 64)
        # self.upid4 = Up(64, 32)
        # self.outid = nn.Conv2d(32, id_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.upx1(x4, x3)
        x = self.upx2(x, x2)
        x = self.upx3(x, x1)
        x = self.upx4(x, x0)
        x = self.outx(x)

        y = self.upy1(x4, x3)
        y = self.upy2(y, x2)
        y = self.upy3(y, x1)
        y = self.upy4(y, x0)
        y = self.outy(y)

        z = self.upz1(x4, x3)
        z = self.upz2(z, x2)
        z = self.upz3(z, x1)
        z = self.upz4(z, x0)
        z = self.outz(z)

        # id = self.upid1(x4, x3)
        # id = self.upid2(id, x2)
        # id = self.upid3(id, x1)
        # id = self.upid4(id, x0)
        # id = self.outid(id)

        return x, y, z
