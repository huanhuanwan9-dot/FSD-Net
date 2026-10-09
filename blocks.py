import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False,
                                  groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False,
                                  groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False,
                                  groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False,
                                      groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
            self,
            features,
            activation,
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output









#
# import torch.nn as nn
#
# def _make_scratch(in_shape, out_shape, groups=1, expand=False):
#     scratch = nn.Module()#  创建一个包含多个卷积层的模块，用于特征提取
#     out_shape1 = out_shape  # 定义用于特征提取的输出形状
#     out_shape2 = out_shape
#     out_shape3 = out_shape
#     out_shape4 = out_shape
#     # 创建四个卷积层作为特征提取器，每个层的输入输出形状可以通过参数设定
#     scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
#     scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
#     scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
#     scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
#     #in_shape[N]：第 N 个卷积层的输入通道数（比如 in_shape[0] 是 layer1_rn 的输入通道）。
# # 这里的 in_shape 通常是一个列表，存储了 backbone 网络不同阶段输出特征的通道数（比如 [64, 128, 256, 512]）。
# # out_shapeN：第 N 个卷积层的输出通道数（比如 out_shape1 是 layer1_rn 的输出通道），用于统一或调整特征维度。
# # kernel_size=3：卷积核大小为 3x3（常用的小卷积核，平衡感受野和计算量）。
# # stride=1：步长为 1（卷积时不压缩特征图的空间尺寸，即输出高宽和输入一致）。
# # padding=1：边缘填充 1 像素（配合 3x3 卷积核和 stride=1，确保输出尺寸 = 输入尺寸，公式：output_size = (input_size - kernel_size + 2*padding) / stride + 1）。
# # bias=False：不使用偏置参数（通常在后续接 BatchNorm 时会省略偏置，因为 BatchNorm 会中心化数据，偏置作用被抵消）。
# # groups=groups：分组卷积参数（若 groups=1 则是普通卷积；若 groups=输入通道数 则是深度可分离卷积），用于控制卷积的计算效率和特征交互方式。
#     return scratch  # 返回创建的模块