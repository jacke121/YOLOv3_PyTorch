import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, t=6, activation=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride, padding=1, bias=False,
                               groups=inplanes * t)
        self.bn2 = nn.BatchNorm2d(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:
            out += residual

        return out


class MobileNet2(nn.Module):
    """MobileNet2 implementation.
    """

    def __init__(self, scale=1.0, input_size=224, t=6, in_channels=3, num_classes=1000, activation=nn.ReLU6):
        """
        MobileNet2 constructor.
        :param in_channels: (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
        :param input_size:
        :param num_classes: number of classes to predict. Default
                is 1000 for ImageNet.
        :param scale:
        :param t:
        :param activation:
        """

        super(MobileNet2, self).__init__()

        self.scale = scale
        self.t = t
        self.activation_type = activation
        self.activation = activation(inplace=True)
        self.num_classes = num_classes

        self.num_of_channels = [32, 16, 24, 32, 64, 96, 160, 240]
        assert (input_size % 32 == 0)
        self.layers_out_filters = [192, 576, 1280]

        self.num_channel = [_make_divisible(ch * self.scale, 8) for ch in self.num_of_channels]
        # self.n = [1, 1, 2, 3, 4, 3, 3, 1]
        self.n = [1, 1, 1, 2, 3, 2, 2, 1]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.conv1 = nn.Conv2d(in_channels, self.num_channel[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channel[0])
        self.bottlenecks = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_conv_out_ch = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        self.conv_last = nn.Conv2d(self.num_channel[-1], self.last_conv_out_ch, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm2d(self.last_conv_out_ch)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t, stage):
        modules = OrderedDict()
        stage_name = "LinearBottleneck{}".format(stage)

        # First module is the only one utilizing stride
        first_module = LinearBottleneck(inplanes=inplanes, outplanes=outplanes, stride=stride, t=t,
                                        activation=self.activation_type)
        modules[stage_name + "_0"] = first_module

        # add more LinearBottleneck depending on number of repeats
        for i in range(n - 1):
            name = stage_name + "_{}".format(i + 1)
            module = LinearBottleneck(inplanes=outplanes, outplanes=outplanes, stride=1, t=6,
                                      activation=self.activation_type)
            modules[name] = module

        return nn.Sequential(modules)

    def _make_bottlenecks(self):
        modules = OrderedDict()
        stage_name = "Bottlenecks"

        # First module is the only one with t=1
        bottleneck1 = self._make_stage(inplanes=self.num_channel[0], outplanes=self.num_channel[1], n=self.n[1], stride=self.s[1], t=1,
                                       stage=0)
        modules[stage_name + "_0"] = bottleneck1

        # add more LinearBottleneck depending on number of repeats
        for i in range(1, len(self.num_channel) - 1):
            name = stage_name + "_{}".format(i)
            module = self._make_stage(inplanes=self.num_channel[i], outplanes=self.num_channel[i + 1], n=self.n[i + 1],
                                      stride=self.s[i + 1],
                                      t=self.t, stage=i)
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        for i, module in enumerate(self.bottlenecks):
            if i == 3 or i == 5:
                branch = torch.nn.Sequential(
                    list(module[0].named_children())[0][1],
                    list(module[0].named_children())[1][1]
                )
                output.append(branch(x))
            x = module(x)
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.activation(x)
        return output[0], output[1], x


def mobilenetv2(pretrained, **kwargs):
    model = MobileNet2()
    if pretrained:
        if isinstance(pretrained, str):
            print("load pretrained",pretrained)
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("mobilenetv2 request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == "__main__":
    # model = mobilenetv2('../weights/mobilenetv2_weights.pth')
    model = mobilenetv2(pretrained="")
    # print(model)
    x = torch.rand(1, 3, 416, 416)
    out3, out4, out5 = model(x)
    print(out3.size())
    print(out4.size())
    print(out5.size())
