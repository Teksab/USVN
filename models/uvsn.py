import torch
from torch import nn
import torch.nn.functional as F

from models.modules.convlstm import ConvLSTM
from models.modules.coordatt import CoordAtt


class DCBlock(nn.Module):     # Dilation Convolution Block
    def __init__(self, input_channels, output_channels, dilation_rates=[1, 2]):
        super(DCBlock, self).__init__()

        self.branches = nn.ModuleList()

        for dilation_rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.2)     # 增加了dropout
                )
            )

    def forward(self, x):
        out = 0
        for branch in self.branches:
            out += branch(x)
        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UVSN(nn.Module):
    def __init__(self, input_channels=1, nf=64, num_class=3):
        super(UVSN, self).__init__()

        # encoder
        self.conv_block1 = DCBlock(input_channels, nf, dilation_rates=[1, 2])
        self.down_sample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block2 = DCBlock(nf, nf*2, dilation_rates=[1, 2])
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block3 = DCBlock(nf*2, nf*4, dilation_rates=[1, 2])
        self.down_sample3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block4 = DCBlock(nf*4, nf*8, dilation_rates=[1, 2])
        self.down_sample4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottle_temporal_model = ConvLSTM(input_dim=nf*8, hidden_dim=[nf*8, nf*8], kernel_size=(3, 3), num_layers=2)
        self.bottle_neck = DCBlock(nf*8, nf*16, dilation_rates=[1, 2])

        # decoder
        self.up_block = nn.ConvTranspose2d(nf*16, nf*8, kernel_size=2, stride=2)
        self.ag4 = Attention_block(F_g=nf*8, F_l=nf*8, F_int=nf*4)
        self.decode_block4 = DCBlock(nf*16, nf*8, dilation_rates=[1, 2])
        self.up_sample3 = nn.ConvTranspose2d(nf*8, nf*4, kernel_size=2, stride=2)
        self.ag3 = Attention_block(F_g=nf*4, F_l=nf*4, F_int=nf*2)
        self.decode_block3 = DCBlock(nf*8, nf*4, dilation_rates=[1, 2])
        self.up_sample2 = nn.ConvTranspose2d(nf*4, nf*2, kernel_size=2, stride=2)
        self.ag2 = Attention_block(F_g=nf*2, F_l=nf*2, F_int=nf)
        self.decode_block2 = DCBlock(nf*4, nf*2, dilation_rates=[1, 2])
        self.up_sample1 = nn.ConvTranspose2d(nf*2, nf, kernel_size=2, stride=2)
        self.ag1 = Attention_block(F_g=nf, F_l=nf, F_int=nf//2)
        self.decode_block1 = DCBlock(nf*2, nf, dilation_rates=[1, 2])

        self.ca_block1 = CoordAtt(nf, nf)
        self.ds_conv1 = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.ca_block2 = CoordAtt(nf*2, nf*2)
        self.ds_conv2 = nn.Conv2d(nf*2, 1, kernel_size=3, padding=1)
        self.ca_block3 = CoordAtt(nf*4, nf*4)
        self.ds_conv3 = nn.Conv2d(nf*4, 1, kernel_size=3, padding=1)
        self.ca_block4 = CoordAtt(nf*8, nf*8)
        self.ds_conv4 = nn.Conv2d(nf*8, 1, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(4, num_class, kernel_size=3, padding=1)

        # self.ds_conv = nn.Conv2d((nf + nf*2 + nf*4 + nf*8 ), nf, kernel_size=3, padding=1)
        # self.ca_block = CoordAtt(nf, nf) 
        # self.out_conv = nn.Conv2d(nf, num_class, kernel_size=3, padding=1)

    def forward(self, x):
        b, t, c, h, w = x.shape  # [b, 3, 1, 256, 256]

        # encoder
        x1_e = torch.stack([self.conv_block1(x[:, i, :, :, :]) for i in range(t)], dim=1)   # [b, 3, 64, 256, 256]
        x2_e = torch.stack([self.conv_block2(self.down_sample1(x1_e[:, i, :, :, :])) for i in range(t)], dim=1)  # [b, 3, 128, 128, 128]
        x3_e = torch.stack([self.conv_block3(self.down_sample2(x2_e[:, i, :, :, :])) for i in range(t)], dim=1)  # [b, 3, 256, 64, 64]
        x4_e = torch.stack([self.conv_block4(self.down_sample3(x3_e[:, i, :, :, :])) for i in range(t)], dim=1)  # [b, 3, 512, 32, 32]

        # bottleneck
        f_b = torch.stack([self.down_sample3(x4_e[:, i, :, :, :]) for i in range(t)], dim=1)    # [b, 3, 512, 16, 16]
        f_b = self.bottle_temporal_model(f_b)   # [b, 3, 512, 16, 16] -> [b, 3, 512, 16, 16]
        f_b_t = f_b[:, -1, :, :, :]
        f_b_t = self.bottle_neck(f_b_t)     # [b, 1024, 16, 16]

        # decoder
        x4_up = self.up_block(f_b_t)    # [b, 512, 32, 32]
        x4_e_t = x4_e[:, -1, :, :, :]    # [b, 512, 32, 32]
        ag4 = self.ag4(x4_up, x4_e_t)
        x4_d = self.decode_block4(torch.cat([x4_e_t, ag4], dim=1))    # [b, 512, 32, 32]
        x3_up = self.up_sample3(x4_d)
        x3_e_t = x3_e[:, -1, :, :, :]
        ag3 = self.ag3(x3_up, x3_e_t)
        x3_d = self.decode_block3(torch.cat([x3_e_t, ag3], dim=1))    # [b, 256, 64, 64]
        x2_up = self.up_sample2(x3_d)
        x2_e_t = x2_e[:, -1, :, :, :]
        ag2 = self.ag2(x2_up, x2_e_t)
        x2_d = self.decode_block2(torch.cat([x2_e_t, ag2], dim=1))    # [b, 128, 128, 128]
        x1_up = self.up_sample1(x2_d)
        x1_e_t = x1_e[:, -1, :, :, :]
        ag1 = self.ag1(x1_up, x1_e_t)
        x1_d = self.decode_block1(torch.cat([x1_e_t, ag1], dim=1))    # [b, 64, 256, 256]

        # deep supervision
        x1_ds = self.ds_conv1(self.ca_block1(x1_d))  # [b, 1, 256, 256]
        x2_ds = self.ds_conv2(self.ca_block2(x2_d))
        x2_ds = F.interpolate(x2_ds, size=(h, w), mode='bilinear', align_corners=False)   # [b, 1, 256, 256]
        x3_ds = self.ds_conv3(self.ca_block3(x3_d))
        x3_ds = F.interpolate(x3_ds, size=(h, w), mode='bilinear', align_corners=False)
        x4_ds = self.ds_conv4(self.ca_block4(x4_d))
        x4_ds = F.interpolate(x4_ds, size=(h, w), mode='bilinear', align_corners=False)
        f_ds = torch.cat([x1_ds, x2_ds, x3_ds, x4_ds], dim=1)    # [b, 4, 256, 256]
        seg_out = self.out_conv(f_ds)   # [b, 3, 256, 256]

        return seg_out


if __name__ == '__main__':
    input = torch.randn(2, 3, 1, 256, 256).cuda()  # [B, T, C, H, W]
    model = UVSN(nf=32, num_class=3).cuda()
    output = model(input)
    print(f'output size: {output.size()}')  # [B, num_class, H, W]
