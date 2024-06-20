import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

from .Config import get_CTranS_config
config = get_CTranS_config()

from .CTrans import ChannelTransformer
from .swin_transformer import SwinTransformer

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class CCA(nn.Module):

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Average Pooling: Tensor(20,512,28,28) => Tensor(20,512,1,1)
        # Average Pooling: Tensor(18,128,64,512) => Tensor(18,128,1,1)
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels, F_x=in_channels)
        self.nConvs = _make_nConv(in_channels*2, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        # x, skip_x: Tensor(18,128,64,512)
        up = x
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x) + up

class AttentionConnection(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.param = nn.Parameter(torch.Tensor(1).fill_(factor))

    def forward(self, feature):

        return (self.param) * feature

class ResNet_34(nn.Module):
    def __init__(self, nclasses, aux, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.mtc = ChannelTransformer(config, vis=False, img_size=512,
                                     channel_num=[config.base_channel,config.base_channel,config.base_channel,config.base_channel],
                                     patchSize=config.patch_sizes)

        self.swin = SwinTransformer()
        self.conv_16_128 = BasicConv2d(16, 128, kernel_size=3, padding=1)
        self.alpha = AttentionConnection()


        self.up = UpBlock_attention(128, 128, nb_Conv=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x
        # x: Tensor(18,5,64,512) => Tensor(BatchSize, xyzRI, h, w)
        # Input Module
        x = self.conv1(x) # 3*3 convolution 64 out, Tensor(18,64,64,512)
        x = self.conv2(x) # 3*3 convolution 128 out, Tensor(18,128,64,512)
        x = self.conv3(x) # 3*3 convolution 128 out, Tensor(18,128,64,512)
        origin = x

        # Backbone: generate feature maps with different resolutions
        x_1 = self.layer1(x)    # 1 Tensor(18,128,64,512)
        x_2 = self.layer2(x_1)  # 1/2 Tensor(18,128,32,256)
        x_3 = self.layer3(x_2)  # 1/4 Tensor(18,128,16,128)
        x_4 = self.layer4(x_3)  # 1/8 Tensor(18,128,8,64)

        # Swin Transformer == Decoder
        x_test = [x_1, x_2, x_3, x_4]
        test = self.swin(x_4,x_test) # Tensor(18,16,64,512)

        x = self.conv_16_128(test)
        x = self.up(origin, x)

        # # Channel Transformer
        # x1, x2, x3, x4, att_weights = self.mtc(x_1, x_2, x_3, x_4)
        # # x_1, x_2, x_3, x_4 = x1, x2, x3, x4
        #
        # d1 = self.up(x_1, x1)
        # d2 = self.up(x_2, x2)
        # d3 = self.up(x_3, x3)
        # d4 = self.up(x_4, x4)
        #
        # x_1, x_2, x_3, x_4 = d1, d2, d3, d4

        # FID to x.size
        # Tensor(18,128,64,512)
        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1) # Concatenate 128*5

        # Classification Head
        out = self.conv_1(out) # 3*3 convolution 256 out, Tensor(18,256,64,512)
        out = self.conv_2(out) # 3*3 convolution 128 out, Tensor(18,128,64,512)
        out = self.semantic_output(out) # nn.Conv2d(128, nclasses, 1), nclasses=20
        out = F.softmax(out, dim=1) # 2d image predictions (0 and 1), Tensor(18,20,64,512)

        # Auxilibry Loss (3 feature maps)
        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)

            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)

            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)

        if self.aux:
            return [out, res_2, res_3, res_4]
        else:
            return out





if __name__ == "__main__":
    import time
    model = ResNet_34(20).cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")

    time_train = []
    for i in range(20):
        inputs = torch.randn(1, 5, 64, 2048).cuda()
        model.eval()

        with torch.no_grad():
          start_time = time.time()
          outputs = model(inputs)

        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)

        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))

        time.sleep(0.15)




