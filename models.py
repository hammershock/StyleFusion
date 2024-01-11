import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class VGG(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(VGG, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.content_layers = content_layers.keys()
        self.style_layers = style_layers.keys()

        # 冻结模型的所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        对vgg19网络的包装，前向传播时保留了内容层和风格层的中间输出
        :param x:
        :return: 内容层和风格层的特征图
        """
        content_features = {}
        style_features = {}

        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_features[name] = x
            if name in self.style_layers:
                style_features[name] = x

        return content_features, style_features


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c)
        )

    def forward(self, x):
        return x + self.layer(x)


class TransNet(nn.Module):
    def __init__(self, input_size):
        """
        实时内容生成网络
        """
        super(TransNet, self).__init__()
        self.input_size = input_size
        self.layer = nn.Sequential(
            ###################下采样层################
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            ##################残差层##################
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),

            ################上采样层##################
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),

            ###############输出层#####################
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d(self.input_size)
        )

    def forward(self, x):
        return self.layer(x)
