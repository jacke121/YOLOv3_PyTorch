import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class CoordXY(nn.Module):
    def __init__(self, out_channels: int, std: float = 0.1):
        super(CoordXY, self).__init__()
        self.out_channels = out_channels
        self.std = std
        self.alpha = Parameter((torch.randn(out_channels) * self.std).view(1,1, self.out_channels,  1))
        self.beta = Parameter((torch.randn(out_channels) * self.std).view(1,1, self.out_channels,  1))

    def forward(self, input):
        assert len(input.shape) == 4, "Tensor should have 4 dimensions"
        dimx = input.shape[-1]
        dimy = input.shape[-2]
        x_special = torch.linspace(-1, 1, steps=dimx).view(1, 1, 1, dimx)
        y_special = torch.linspace(-1, 1, steps=dimy).view(1, 1, dimy, 1)
        if self.alpha.is_cuda:
            device = self.alpha.get_device()
            x_special = x_special.cuda(device=device)
            y_special = y_special.cuda(device=device)

        input.add_(self.alpha * x_special).add_(self.beta * y_special)
        return input

class AddCoordsTh(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.x_dim], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.y_dim], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2,1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        # input_tensor = input_tensor.cuda()
        xx_channel = xx_channel.cuda()
        yy_channel = yy_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self,coord_dim, in_channels, out_channels, kernel_size, stride=1,padding=0, bias=True):
        super(CoordConv, self).__init__()
        # self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        self.addcoords = CoordXY(coord_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=padding,bias=bias)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret