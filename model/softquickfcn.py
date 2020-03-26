"""Quicknat architecture"""
import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from base import BaseModel

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class DenseBlock(nn.Module):
    """Block with dense connections

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DenseBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        # self.norm1 = nn.BatchNorm2d(num_features=params['num_channels'], momentum=0.9)
        # self.norm2 = nn.BatchNorm2d(num_features=conv1_out_size, momentum=0.9)
        # self.norm3 = nn.BatchNorm2d(num_features=conv2_out_size, momentum=0.9)
        # self.norm1 = GroupNorm(num_features=params['num_channels'], num_groups=2)
        # self.norm2 = GroupNorm(num_features=conv1_out_size, num_groups=2)
        # self.norm3 = GroupNorm(num_features=conv2_out_size, num_groups=2)
        self.norm1 = nn.InstanceNorm2d(num_features=params['num_channels'])
        self.norm2 = nn.InstanceNorm2d(num_features=conv1_out_size)
        self.norm3 = nn.InstanceNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.norm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.norm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.norm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
    """Dense encoder block with maxpool and an optional SE block

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(EncoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
        """

        out_block = super(EncoderBlock, self).forward(input)
        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DecoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block

class SoftQuickFCN(BaseModel):
    """
    A PyTorch implementation of QuickNAT

    """
    def __init__(self, params):
        """

        :param params: {'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_classes':28
                        'se_block': False,
                        'drop_out':0.2}
        """
        super(SoftQuickFCN, self).__init__()

        print('num channels: ', params['num_channels'])

        self.encode1_seg = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode1_class = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        cross1s_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        cross1c_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        self.cross1ss = torch.ones(1, params['num_filters'], 1, 1) * cross1s_init
        self.cross1sc = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross1s_init)
        self.cross1cc = torch.ones(1, params['num_filters'], 1, 1) * cross1c_init
        self.cross1cs = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross1c_init)

        params['num_channels'] = params['num_filters']
        self.encode2_seg = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode2_class = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        cross2s_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        cross2c_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        self.cross2ss = torch.ones(1, params['num_filters'], 1, 1) * cross2s_init
        self.cross2sc = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross2s_init)
        self.cross2cc = torch.ones(1, params['num_filters'], 1, 1) * cross2c_init
        self.cross2cs = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross2c_init)

        self.encode3_seg = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode3_class = EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        cross3s_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        cross3c_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        self.cross3ss = torch.ones(1, params['num_filters'], 1, 1) * cross3s_init
        self.cross3sc = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross3s_init)
        self.cross3cc = torch.ones(1, params['num_filters'], 1, 1) * cross3c_init
        self.cross3cs = torch.ones(1, params['num_filters'], 1, 1) * (1 - cross3c_init)

        self.bottleneck_seg = DenseBlock(params, se_block_type=se.SELayer.CSSE)
        self.bottleneck_class = DenseBlock(params, se_block_type=se.SELayer.CSSE)
        crossbs_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        crossbc_init = torch.FloatTensor(1, 1).uniform_(0.05, 0.95)
        self.crossbss = torch.ones(1, params['num_filters'], 1, 1) * crossbs_init
        self.crossbsc = torch.ones(1, params['num_filters'], 1, 1) * (1 - crossbs_init)
        self.crossbcc = torch.ones(1, params['num_filters'], 1, 1) * crossbc_init
        self.crossbcs = torch.ones(1, params['num_filters'], 1, 1) * (1 - crossbc_init)

        params['num_channels'] = 2 * params['num_filters']
        self.decode1_seg = DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode2_seg = DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode3_seg = DecoderBlock(params, se_block_type=se.SELayer.CSSE)

        params['num_channels'] = params['num_filters']
        self.classifier_seg = sm.ClassifierBlock(params)

        ############Classification Task############
        self.classifier_class = nn.Sequential(
            nn.Linear(params['num_channels'] * 50 * 50, 25),
            nn.PReLU(),
            nn.Linear(25,3)
        )


    def forward(self, input):
        """

        :param input: X
        :return: probabiliy map
        """
        e1s, out1s, ind1s = self.encode1_seg.forward(input)
        e1c, out1c, ind1c = self.encode1_class.forward(input)
        e1s_sum = self.cross1ss * e1s + self.cross1sc * e1c
        e1c_sum = self.cross1cs * e1s + self.cross1cc * e1c

        e2s, out2s, ind2s = self.encode2_seg.forward(e1s_sum)
        e2c, out2c, ind2c = self.encode2_class.forward(e1c_sum)
        e2s_sum = self.cross2ss * e2s + self.cross2sc * e2c
        e2c_sum = self.cross2cs * e2s + self.cross2cc * e2c

        e3s, out3s, ind3s = self.encode3_seg.forward(e2s_sum)
        e3c, out3c, ind3c = self.encode3_class.forward(e2c_sum)
        e3s_sum = self.cross3ss * e3s + self.cross3sc * e3c
        e3c_sum = self.cross3cs * e3s + self.cross3cc * e3c

        bns = self.bottleneck_seg.forward(e3s_sum)
        bnc = self.bottleneck_class.forward(e3c_sum)
        bns_sum = self.crossbss * bns + self.crossbsc * bnc
        bnc_sum = self.crossbcs * bns + self.crossbcc * bnc

        ############Segmentation Task############
        d3 = self.decode1_seg.forward(bns_sum, out3s, ind3s)
        d2 = self.decode2_seg.forward(d3, out2s, ind2s)
        d1 = self.decode3_seg.forward(d2, out1s, ind1s)
        prob = self.classifier_seg.forward(d1)

        ############Classification Task############
        bn_flattened = bnc_sum.view(bnc_sum.shape[0],-1) #reshape to (Batch Size, Input Dim Flattened)
        classes = self.classifier_class.forward(bn_flattened)


        return prob, classes

    def enable_test_dropout(self):
        """
        Enables test time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    def disable_test_dropout(self):
        """
        Disables train time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.eval)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.eval)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        
        
        for number, x in enumerate(X['test']):
            X,_ = x['image'].float(), x['labels'].float()
        
        
        
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        del X, out, idx, max_val
        return prediction
