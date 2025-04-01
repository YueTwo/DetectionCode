import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys


class SeparableConv2d(nn.Module):  # 可分离卷积(整合了depthwise和pointwise卷积)
    def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.c(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):  # 基本块

    def __init__(self, c_in, c_out, reps, stride=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        """
        c_in:输入层channel数
        c_out:输出层channel数
        reps:一个Block中SeparableConv2d块的数量
        stride:卷积步长
        start_with_relu:是否为网络中的第一个Block
        grow_first：是否为网络中的最后一个Block
    """
        self.skip = None
        self.skip_bn = None
        if c_out != c_in or stride != 1:  # 是否加入残差模块的条件判断
            self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(c_out)

        self.relu = nn.ReLU(inplace=False)  #########################

        rep = []
        c = c_in
        if grow_first:  # 如果是Block中的第一次卷积，就需要调整Channel数
            rep.append(self.relu)
            rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c_out))
            c = c_out

        # 中间过程的卷积Channel数不变
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(c, c, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c))

        # 只有Exit flow中卷积块的最后一次卷积，Channel数扩充
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(c_out))

        # 如果是第一个Block中的第一个卷积块,就需要去掉最开始的relu层
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            y = self.skip(inp)
            y = self.skip_bn(y)
        else:
            y = inp
        x += y
        return x


# Reg. Map
# 采用Direct regression，通过一个可分离卷积和Sigmoid函数可以得到
class RegressionMap(nn.Module):
    def __init__(self, c_in):
        super(RegressionMap, self).__init__()
        self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
        self.s = nn.Sigmoid()

    def forward(self, x):
        # 可分离卷积
        mask = self.c(x)
        # sigmoid
        mask = self.s(mask)
        return mask, None


# MAM Map 全称Manipulation Appearance Model，是作者自己提出的一个结构。
# 该结构包括一个可分离卷积块和一个全连接，以此来估计各个区域的权重α
class TemplateMap(nn.Module):
    def __init__(self, c_in, templates):
        super(TemplateMap, self).__init__()
        self.c = Block(c_in, 364, 2, 2, start_with_relu=True, grow_first=False)
        self.l = nn.Linear(364, 10)
        self.relu = nn.ReLU(inplace=True)

        self.templates = templates

    def forward(self, x):
        # 可分离卷积
        v = self.c(x)
        v = self.relu(v)
        # 适应性平均池化
        v = F.adaptive_avg_pool2d(v, (1, 1))
        v = v.view(v.size(0), -1)
        # 全连接
        v = self.l(v)
        mask = torch.mm(v, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)

        return mask, v


# M a t t = M ‾ + A ⋅ α  其中-M和A AA通过主成分分析(PCA)提取。其数据来源于FaceApp计算的100个真实的manipulation masks
class PCATemplateMap(nn.Module):
    def __init__(self, templates):
        super(PCATemplateMap, self).__init__()
        self.templates = templates

    def forward(self, x):
        fe = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        fe = torch.transpose(fe, 1, 2)
        # 均值
        mu = torch.mean(fe, 2, keepdim=True)
        fea_diff = fe - mu

        cov_fea = torch.bmm(fea_diff, torch.transpose(fea_diff, 1, 2))
        B = self.templates.reshape(1, 10, 361).repeat(x.shape[0], 1, 1)
        D = torch.bmm(torch.bmm(B, cov_fea), torch.transpose(B, 1, 2))
        # 特征值和特征向量
        eigen_value, eigen_vector = D.symeig(eigenvectors=True)
        index = torch.tensor([9]).cuda()
        eigen = torch.index_select(eigen_vector, 2, index)

        v = eigen.squeeze(-1)
        mask = torch.mm(v, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)
        return mask, v


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class OutlookAttention0(nn.Module):

    def __init__(self, dim, num_heads=4, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride)  # 手动卷积
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                   h * w).permute(
            0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size \
                                       , self.kernel_size * self.kernel_size).permute(0, 2, 1, 3,
                                                                                      4)  # B，num_head，H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size,
                                                        h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out)

        return out


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        self.unflod = nn.Unfold(kernel_size, padding, stride)  # 手动卷积
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        # B, H, W, C = x.shape  # torch.Size([24, 728, 19, 19])
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        # v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                   h * w).permute(
            0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size \
                                       , self.kernel_size * self.kernel_size).permute(0, 2, 1, 3,
                                                                                      4)  # B，num_head，H*W,kxk,kxk
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size,
                                                        h * w)  # B,dimxkxk,H*W
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
                     padding=self.padding, stride=self.stride)  # B,C,H,W
        out = self.proj(out.permute(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out).permute(0, 3, 1, 2)

        return out


# 主干网络(注意力层也夹在其中)
class Xception(nn.Module):
    """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """

    def __init__(self, maptype, templates, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # Entry flow 这一模块包括最开始的两层普通卷积和三个可分离卷积块，其中每个可分离卷积块又包括两次可分离卷积，
        # 且第一次可分离卷积需要对Channel数调整，第二次可分离卷积后需要残差块处理。
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)  #######

        # self.attention_block1 = ECALayer()

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        # self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
        self.block3 = Block(256, 512, 2, 2, start_with_relu=True, grow_first=True)
        self.outlook_att0 = OutlookAttention0(512)
        self.outlook_att1 = OutlookAttention0(512)
        ######xf3

        self.block34 = Block(512, 728, 3, 1, start_with_relu=True, grow_first=True)
        # Middle flow 这一模块为一个可分离卷积块重复八次，每个卷积快包括三次可分离卷积
        # 注意力层在Middle flow中第4个模块和第5个模块之间
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow 这一模块包括一个可分离卷积块和后续收尾操作：2*可分离卷积，全局平均池化，全连接等
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.outlook_att2 = OutlookAttention0(1024)
        self.outlook_att3 = OutlookAttention0(1024)
        # xf4

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.outlook_att4 = OutlookAttention0(2048)
        self.outlook_att5 = OutlookAttention0(2048)
        ###xf5

        self.max1 = nn.MaxPool2d(kernel_size=16, stride=16)  ########
        self.max2 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.max3 = nn.MaxPool2d(kernel_size=6, stride=6)
        self.num_ftrs = 2048 * 1 * 1
        self.elu = nn.ELU(inplace=False)  ###########

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 2),
            nn.Linear(1024 * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=False),  #################
            nn.Linear(2048, num_classes),
        )

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(2048, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),  #######
            nn.Linear(self.num_ftrs // 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=False),  ##########
            nn.Linear(2048, num_classes),
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(2048, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=False),  ###########
            nn.Linear(2048, num_classes),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, 2048, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(2048, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=False),  #################
            nn.Linear(2048, num_classes),
        )
        ###
        self.last_linear = nn.Linear(2048, num_classes)

        if maptype == 'none':
            self.map = [1, None]
        elif maptype == 'reg':
            self.map = RegressionMap(728)
        elif maptype == 'tmp':
            self.map = TemplateMap(728, templates)
        elif maptype == 'pca_tmp':
            self.map = PCATemplateMap(728)
        else:
            print('Unknown map type: `{0}`'.format(maptype))
            sys.exit()

    def features(self, input):
        # Entry flow
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block34(x)
        # print(x.shape) torch.Size([24, 728, 19, 19])
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        # Exit flow
        x = self.block12(x)
        x = x.permute(0, 2, 3, 1)
        x = self.outlook_att2(x)  # B,H,W,C
        x = self.outlook_att3(x)  # B,H,W,C
        x = x.permute(0, 3, 1, 2)
        x4 = x

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.outlook_att4(x)  # B,H,W,C
        x = self.outlook_att5(x)  # B,H,W,C
        x = x.permute(0, 3, 1, 2)
        x5 = x  #################
        return x4, x5

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (batch_size, num_channels)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        xf4, xf5 = self.features(input)

        xl2 = self.conv_block2(xf4)
        xl3 = self.conv_block3(xf5)

        xl2 = self.max2(xl2)
        xl2 = xl2.view(xl2.size(0), -1)
        xc2 = self.classifier2(xl2)

        xl3 = self.max3(xl3)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)

        x_concat = torch.cat((xl2, xl3), -1)
        x_concat = self.classifier_concat(x_concat)

        # x = self.logits(x)
        return xc2, xc3, x_concat


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('SeparableConv2d') != -1:
        # Initialization for SeparableConv2d
        m.c.weight.data.normal_(0.0, 0.01)
        if m.c.bias is not None:
            m.c.bias.data.fill_(0)
        m.pointwise.weight.data.normal_(0.0, 0.01)
        if m.pointwise.bias is not None:
            m.pointwise.bias.data.fill_(0)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        # Initialize weights for Conv2d and Linear layers
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        # Initialization for BatchNorm2d
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('LSTM') != -1:
        for i in m._parameters:
            if i.__class__.__name__.find('weight') != -1:
                i.data.normal_(0.0, 0.01)
            elif i.__class__.__name__.find('bias') != -1:
                i.bias.data.fill_(0)


class Model:
    def __init__(self, maptype='None', templates=None, num_classes=2, load_pretrain=True):
        model = Xception(maptype, templates, num_classes=num_classes)
        if load_pretrain:
            state_dict = torch.load('./xception-b5690688.pth')
            for name, weights in state_dict:
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            del state_dict['fc.weight']
            del state_dict['fc.bias']
            model.load_state_dict(state_dict, False)
        else:
            model.apply(init_weights)
        self.model = model

    def save(self, epoch, optim, model_dir):
        state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
        torch.save(state, '{0}/{1:06d}.tar'.format(model_dir, epoch))
        print('Saved models `{0}`'.format(epoch))

    def load(self, epoch, model_dir):
        filename = '{0}{1:06d}.tar'.format(model_dir, epoch)
        #print('模型文件路径为{0}：'.format(filename))
        #print('Loading models from {0}'.format(filename))
        if os.path.exists(filename):
            state = torch.load(filename, map_location=torch.device('cuda:0'))
            self.model.load_state_dict(state['net'])
            #print("LocalFace模型加载完成！")
        else:
            print('Failed to load models from {0}'.format(filename))
