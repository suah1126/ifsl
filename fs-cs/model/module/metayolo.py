import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import functional as F
import numpy as np

from model.module.dynamic_conv import dynamic_conv2d
import pdb

import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from common.detection import *
from common.cfg import *
from numbers import Number
from random import random
import pdb

def maybe_repeat(x1, x2):
    n1 = x1.size(0)
    n2 = x2.size(0)
    if n1 == n2:
        pass
    elif n1 < n2:
        assert n2 % n1 == 0
        shape = x1.shape[1:]
        nc = n2 // n1
        x1 = x1.repeat(nc, *[1]*x1.dim())
        x1 = x1.transpose(0,1).contiguous()
        x1 = x1.view(-1, *shape)
    else:
        assert n1 % n2 == 0
        shape = x2.shape[1:]
        nc = n1 // n2
        x2 = x2.repeat(nc, *[1]*x2.dim())
        x2 = x2.transpose(0,1).contiguous()
        x2 = x2.view(-1, *shape)
    return x1, x2

class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalMaxPool2d(Module):

    def __init__(self, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(GlobalMaxPool2d, self).__init__()
        self.stride = stride or 1
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'global max pooling, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input):
        kernel_size = input.size(-1)
        return F.max_pool2d(input, kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

class GlobalAvgPool2d(Module):

    def __init__(self, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(GlobalAvgPool2d, self).__init__()
        self.stride = stride or 1
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self):
        return 'global avg pooling'

    def forward(self, input):
        # kernel_size = input.size(-1)
        return F.adaptive_avg_pool2d(input, 1)

class Split(Module):

    def __init__(self, splits):
        super(Split, self).__init__()
        self.splits = splits

    def extra_repr(self):
        return 'split layer, splits={splits}'.format(**self.__dict__)

    def forward(self, input):
        splits = np.cumsum([0] + self.splits)
        xs = [input[:,splits[i]:splits[i+1],:,:].contiguous() for i in range(len(splits) - 1)]
        return xs

class Darkne_org(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))

        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
    
        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))
    
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    load_conv_tmp = load_convfromcoco if 'coco2voc' in cfg.backup else load_conv
                    load_conv_tmp = load_conv
                    start = load_conv_tmp(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

# class Darknet(nn.Module):
#     def __init__(self):
#         super(Darknet, self).__init__()
#         self.blocks = parse_cfg('./model/module/metayolo_darknet.cfg')
#         self.models = create_network(self.blocks) # merge conv, bn,leaky
#         self.loss = self.models[len(self.models)-1]

#         self.width = int(self.blocks[0]['width'])
#         self.height = int(self.blocks[0]['height'])

#         if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
#             self.anchors = self.loss.anchors
#             self.num_anchors = self.loss.num_anchors
#             self.anchor_step = self.loss.anchor_step
#             self.num_classes = self.loss.num_classes

#         self.header = torch.IntTensor([0,0,0,0])
#         self.seen = 0
#         self.load_weights(self.blocks, self.models)

#     def forward(self, x, dynamic_weights):
#         # Perform detection
#         ind = -2
#         dynamic_cnt = 0
#         self.loss = None
#         outputs = dict()
#         for block in self.blocks:                           # darknet 
#             ind = ind + 1

#             if block['type'] == 'net':
#                 continue
#             elif block['type'] == 'convolutional' or \
#                  block['type'] == 'maxpool' or \
#                  block['type'] == 'reorg' or \
#                  block['type'] == 'avgpool' or \
#                  block['type'] == 'softmax' or \
#                  block['type'] == 'connected' or \
#                  block['type'] == 'globalavg' or \
#                  block['type'] == 'globalmax':
#                 if is_dynamic(block):                  # feature + reweighting vector
#                     x = self.models[ind]((x, dynamic_weights[dynamic_cnt]))
#                     #print('d: ', dynamic_weights[dynamic_cnt].shape)
#                     dynamic_cnt += 1
#                 else:
#                     x = self.models[ind](x)
#                 outputs[ind] = x
#             elif block['type'] == 'route':
#                 layers = block['layers'].split(',')
#                 layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
#                 if len(layers) == 1:
#                     x = outputs[layers[0]]
#                     outputs[ind] = x
#                 elif len(layers) == 2:
#                     x1 = outputs[layers[0]]
#                     x2 = outputs[layers[1]]
#                     if 'concat' in block and int(block['concat']) == 0:
#                         x = (x1, x2)
#                     else:
#                         x1, x2 = maybe_repeat(x1, x2)
#                         x = torch.cat((x1,x2),1)
#                     outputs[ind] = x
#             elif block['type'] == 'shortcut':
#                 from_layer = int(block['from'])
#                 activation = block['activation']
#                 from_layer = from_layer if from_layer > 0 else from_layer + ind
#                 x1 = outputs[from_layer]
#                 x2 = outputs[ind-1]
#                 x  = x1 + x2
#                 if activation == 'leaky':
#                     x = F.leaky_relu(x, 0.1, inplace=True)
#                 elif activation == 'relu':
#                     x = F.relu(x, inplace=True)
#                 outputs[ind] = x
#             elif block['type'] == 'region':
#                 continue
#                 if self.loss:
#                     self.loss = self.loss + self.models[ind](x)
#                 else:
#                     self.loss = self.models[ind](x)
#                 outputs[ind] = None
#             elif block['type'] == 'cost':
#                 continue
#             else:
#                 print('unknown type %s' % (block['type']))

#         #     if block['type'] == 'darknet':
#         #         continue
#         #     elif block['type'] == 'route':
#         #         layers = block['layers'].split(',')
#         #         layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
#         #         if len(layers) == 1:
#         #             x = outputs[layers[0]]
#         #             outputs[ind] = x
#         #         elif len(layers) == 2:
#         #             x1 = outputs[layers[0]]
#         #             x2 = outputs[layers[1]]
#         #             if 'concat' in block and int(block['concat']) == 0:
#         #                 x = (x1, x2)
#         #             else:
#         #                 x1, x2 = maybe_repeat(x1, x2)
#         #                 x = torch.cat((x1,x2),1)
#         #             outputs[ind] = x
#         #     else:
#         #         if is_dynamic(block):                  # feature + reweighting vector
#         #             x = self.models[ind]((x, dynamic_weights[dynamic_cnt]))
#         #             dynamic_cnt += 1
#         #         else:
#         #             x = self.models[ind](x)
#         #         outputs[ind] = x
#         #     #print(x.shape)
#         # return x

#     def load_weights(self, blocks, models):
#         fp = open('/home/suachoi/ifsl/fs-cs/model/module/darknet19_448.conv.23', 'rb')
#         header = np.fromfile(fp, count=4, dtype=np.int32)
#         self.header = torch.from_numpy(header)
#         self.seen = self.header[3]
#         buf = np.fromfile(fp, dtype = np.float32)
#         fp.close()

#         start = 0
#         ind = -2
#         for block in blocks:
#             if start >= buf.size:
#                 break
#             ind = ind + 1
#             if block['type'] == 'darknet' or block['type'] == 'learnet':
#                 continue
#             elif block['type'] == 'convolutional':
#                 model = models[ind]
#                 if is_dynamic(block) and model[0].weight is None:
#                     continue    
#                 batch_normalize = int(block['batch_normalize'])
#                 if batch_normalize:
#                     start = load_conv_bn(buf, start, model[0], model[1])
#                 else:
                    
#                     start = load_conv(buf, start, model[0])
#             elif block['type'] == 'connected':
#                 model = models[ind]
#                 if block['activation'] != 'linear':
#                     start = load_fc(buf, start, model[0])
#                 else:
#                     start = load_fc(buf, start, model)
#             elif block['type'] == 'maxpool':
#                 pass
#             elif block['type'] == 'reorg':
#                 pass
#             elif block['type'] == 'route':
#                 pass
#             elif block['type'] == 'shortcut':
#                 pass
#             elif block['type'] == 'region':
#                 pass
#             elif block['type'] == 'avgpool':
#                 pass
#             elif block['type'] == 'softmax':
#                 pass
#             elif block['type'] == 'cost':
#                 pass
#             elif block['type'] == 'globalmax':
#                 pass
#             elif block['type'] == 'globalavg':
#                 pass
#             elif block['type'] == 'split':
#                 pass
#             else:
#                 print('unknown type %s' % (block['type']))

class MetaYoloLearner(nn.Module):
    def __init__(self):
        super(MetaYoloLearner, self).__init__()
        self.blocks = parse_cfg('./model/module/metayolo_learner.cfg')
        self.models = create_network(self.blocks) # merge conv, bn,leaky

    def forward(self, spt):
        dynamic_weights = []
        for model in self.models:
            #print('learner: ', spt.shape)
            spt = model(spt)
            if isinstance(spt, list):
                dynamic_weights.append(spt[0])
                spt = spt[-1]
        dynamic_weights.append(spt)

        return dynamic_weights

class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg('./model/module/metayolo_darknet.cfg')
        self.learnet_blocks = parse_cfg('./model/module/metayolo_learner.cfg')
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.learnet_models = self.create_network(self.learnet_blocks)
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        #self.load_weights('./model/module/darknet19_448.conv.23')
        #self.load_weights('./logs/pascal/fold3/resnet101/last.ckpt')

    def meta_forward(self, metax):
        # Get weights from learnet
        done_split = False
        for i in range(int(self.learnet_blocks[0]['feat_layer'])):
            if i == 0 and metax.size(1) == 6:
                done_split = True
                metax = torch.cat(torch.split(metax, 3, dim=1))
            metax = self.models[i](metax)
        if done_split:
            metax = torch.cat(torch.split(metax, int(metax.size(0)/2)), dim=1)
        # if cfg.metain_type in [2, 3]:
        #     metax = torch.cat([metax, mask], dim=1)
        dynamic_weights = []
        for model in self.learnet_models:
            #print('learner: ', metax.shape)
            metax = model(metax)
            if isinstance(metax, list):
                dynamic_weights.append(metax[0])
                metax = metax[-1]
        dynamic_weights.append(metax)
        return dynamic_weights

    def detect_forward(self, x, dynamic_weights):
        # Perform detection
        ind = -2
        dynamic_cnt = 0
        self.loss = None
        outputs = dict()
        for block in self.blocks:                           # darknet 
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'darknet':
                continue
            elif block['type'] == 'convolutional' or \
                 block['type'] == 'maxpool' or \
                 block['type'] == 'reorg' or \
                 block['type'] == 'avgpool' or \
                 block['type'] == 'softmax' or \
                 block['type'] == 'connected' or \
                 block['type'] == 'globalavg' or \
                 block['type'] == 'globalmax':
                if self.is_dynamic(block):                  # feature + reweighting vector
                    x = self.models[ind]((x, dynamic_weights[dynamic_cnt]))
                    #print('d: ', dynamic_weights[dynamic_cnt].shape)
                    dynamic_cnt += 1
                else:
                    x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    if 'concat' in block and int(block['concat']) == 0:
                        x = (x1, x2)
                    else:
                        x1, x2 = maybe_repeat(x1, x2)
                        x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
            #print(x.shape)
        return x
        
    def forward(self, x, metax, ids=None):
        # pdb.set_trace()
        
        dynamic_weights = self.meta_forward(metax)        # reweighting module. input: support, output: reweighting vectors
        x = self.detect_forward(x, dynamic_weights)             # darknet + prediction module
        return x

    def print_network(self):
        print_cfg(self.blocks)
        print('---------------------------------------------------------------------')
        print_cfg(self.learnet_blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters =[]
        conv_id = 0
        dynamic_count = 0
        for block in blocks:
            if block['type'] == 'darknet' or block['type'] == 'learnet':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                groups = 1
                bias = bool(int(block['bias'])) if 'bias' in block else True

                if self.is_dynamic(block):
                    partial = int(block['partial']) if 'partial' in block else None
                    Conv2d = dynamic_conv2d(dynamic_count == 0, partial=partial)
                    dynamic_count += 1
                else:
                    Conv2d = nn.Conv2d
                if 'groups' in block:
                    groups = int(block['groups'])

                model = nn.Sequential()
                if batch_normalize:
                    model.add_module(
                        'conv{0}'.format(conv_id),
                        Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=False))
                    model.add_module(
                        'bn{0}'.format(conv_id),
                        nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module(
                        'conv{0}'.format(conv_id),
                        Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=bias))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model =  F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLossV2()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)//loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            elif block['type'] == 'globalmax':
                model = GlobalMaxPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'globalavg':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'split':
                splits = [int(sz) for sz in block['splits'].split(',')]
                model = Split(splits)
                prev_filters = splits[-1]
                out_filters.append(prev_filters)
                models.append(model)
            else:
                print('unknown type %s' % (block['type']))
    
        # pdb.set_trace()
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        for blocks, models in [(self.blocks, self.models), (self.learnet_blocks, self.learnet_models)]:
            ind = -2
            for block in blocks:
                if start >= buf.size:
                    break
                ind = ind + 1
                if block['type'] == 'darknet' or block['type'] == 'learnet':
                    continue
                elif block['type'] == 'convolutional':
                    model = models[ind]
                    if self.is_dynamic(block) and model[0].weight is None:
                        continue    
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:
                        start = load_conv(buf, start, model[0])
                elif block['type'] == 'connected':
                    model = models[ind]
                    if block['activation'] != 'linear':
                        start = load_fc(buf, start, model[0])
                    else:
                        start = load_fc(buf, start, model)
                elif block['type'] == 'maxpool':
                    pass
                elif block['type'] == 'reorg':
                    pass
                elif block['type'] == 'route':
                    pass
                elif block['type'] == 'shortcut':
                    pass
                elif block['type'] == 'region':
                    pass
                elif block['type'] == 'avgpool':
                    pass
                elif block['type'] == 'softmax':
                    pass
                elif block['type'] == 'cost':
                    pass
                elif block['type'] == 'globalmax':
                    pass
                elif block['type'] == 'globalavg':
                    pass
                elif block['type'] == 'split':
                    pass
                else:
                    print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        # pdb.set_trace()
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1 + len(self.learnet_blocks)

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            # pdb.set_trace()
            if blockId >= len(self.blocks):
                if blockId == len(self.blocks):
                    ind = -2
                blockId = blockId - len(self.blocks)
                blocks = self.learnet_blocks
                models = self.learnet_models
            else:
                blocks = self.blocks
                models = self.models
            ind = ind + 1

            block = blocks[blockId]
            if block['type'] == 'convolutional':
                model = models[ind]
                if self.is_dynamic(block) and model[0].weight is None:
                    continue
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = models[ind]
                if block['activation'] == 'linear':
                    save_fc(fp, model)
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'globalmax':
                pass
            elif block['type'] == 'learnet':
                pass
            elif block['type'] == 'globalavg':
                pass
            elif block['type'] == 'split':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def is_dynamic(self, block):
        return 'dynamic' in block and int(block['dynamic']) == 1

def create_network(blocks):
    models = nn.ModuleList()

    prev_filters = 3
    out_filters =[]
    conv_id = 0
    dynamic_count = 0
    for block in blocks:
        if block['type'] == 'darknet' or block['type'] == 'learnet':
            prev_filters = int(block['channels'])
            continue
        elif block['type'] == 'convolutional':
            conv_id = conv_id + 1
            batch_normalize = int(block['batch_normalize'])
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)//2 if is_pad else 0
            activation = block['activation']
            groups = 1
            bias = bool(int(block['bias'])) if 'bias' in block else True

            if is_dynamic(block):
                partial = int(block['partial']) if 'partial' in block else None
                Conv2d = dynamic_conv2d(dynamic_count == 0, partial=partial)
                dynamic_count += 1
            else:
                Conv2d = nn.Conv2d
            if 'groups' in block:
                groups = int(block['groups'])

            model = nn.Sequential()
            if batch_normalize:
                model.add_module(
                    'conv{0}'.format(conv_id),
                    Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=False))
                model.add_module(
                    'bn{0}'.format(conv_id),
                    nn.BatchNorm2d(filters))
                #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
            else:
                model.add_module(
                    'conv{0}'.format(conv_id),
                    Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=bias))
            if activation == 'leaky':
                model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'relu':
                model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
            prev_filters = filters
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            model = nn.MaxPool2d(pool_size, stride)
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'avgpool':
            model = GlobalAvgPool2d()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'softmax':
            model = nn.Softmax()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'cost':
            if block['_type'] == 'sse':
                model = nn.MSELoss(size_average=True)
            elif block['_type'] == 'L1':
                model = nn.L1Loss(size_average=True)
            elif block['_type'] == 'smooth':
                model = nn.SmoothL1Loss(size_average=True)
            out_filters.append(1)
            models.append(model)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            prev_filters = stride * stride * prev_filters
            out_filters.append(prev_filters)
            models.append(Reorg(stride))
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            ind = len(models)
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                assert(layers[0] == ind - 1)
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_filters.append(prev_filters)
            models.append(EmptyModule())
        elif block['type'] == 'globalmax':
            model = GlobalMaxPool2d()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'globalavg':
            model = GlobalAvgPool2d()
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'split':
            splits = [int(sz) for sz in block['splits'].split(',')]
            model = Split(splits)
            prev_filters = splits[-1]
            out_filters.append(prev_filters)
            models.append(model)
        elif block['type'] == 'region':
            loss = RegionLossV2()
            anchors = block['anchors'].split(',')
            loss.anchors = [float(i) for i in anchors]
            loss.num_classes = int(block['classes'])
            loss.num_anchors = int(block['num'])
            loss.anchor_step = len(loss.anchors)//loss.num_anchors
            loss.object_scale = float(block['object_scale'])
            loss.noobject_scale = float(block['noobject_scale'])
            loss.class_scale = float(block['class_scale'])
            loss.coord_scale = float(block['coord_scale'])
            out_filters.append(prev_filters)
            models.append(loss)
        else:
            print('unknown type %s' % (block['type']))

    # pdb.set_trace()
    return models

    
def neg_filter(pred_boxes, target, withids=False):
    assert pred_boxes.size(0) == target.size(0)
    cfg.neg_ratio = 1
    #print(pred_boxes.shape, target.shape)
    if cfg.neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(cfg.neg_ratio, Number):
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        ratio = cfg.neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            inds = np.argwhere(flags).squeeze()
            pred_boxes, target = pred_boxes[inds], target[inds]
            #print(pred_boxes.shape, target.shape)
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, epoch):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors)//num_anchors
    # #print('anchor_step: ', anchor_step)
    #print('nB', nB)
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW)
    tw         = torch.zeros(nB, nA, nH, nW)
    th         = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()      # clsid, x, y, w, h > transpose
        cur_ious = torch.zeros(nAnchors)
        for t in range(cfg.max_boxes):
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW        #grid scale로 표현한 gt
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        # Find anchors with iou > sil_thresh
        # no loss for that one
        conf_mask[b][torch.reshape(cur_ious, (nA, nH, nW)) > sil_thresh] = 0    # iou > 0.6인 데이터는 loss 계산 안함.
    if epoch < 1:
        if anchor_step == 4:
            tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
            ty = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        # pdb.set_trace()
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            for n in range(nA):     # 가로 세로 비율, 크기가 가장 맞는 anchor box를 고름.
                aw = anchors[anchor_step*n]     #anchor width, height
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:    #anchor_step 2임.
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi].cuda()

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi      # 실제 coordinate -> output format으로 바꿈.
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]

            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLossV2(nn.Module):
    """
    Yolo region loss + Softmax classification across meta-inputs
    """
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLossV2, self).__init__()
        self.num_classes = 1    #way
        self.anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
        self.num_anchors = 5
        self.anchor_step = len(self.anchors)//self.num_anchors
        self.coord_scale = 1.
        self.noobject_scale = 1.
        self.object_scale = 5.
        self.class_scale = 10
        self.thresh = 0.6
        self.seen = 0
        self.log = open('log.txt', 'a')
        print('initialize regionloss')
        #print('class_scale', self.class_scale)

    def forward(self, output, target, epoch):
        #output : BxAs*(4+1+num_classes)*H*W
        # 모든 grid마다 class의 xywh conf 계산함.
        # Get all classification prediction
        # pdb.set_trace()
        #print('target:', target.shape)
        bs = target.size(0)
        cs = target.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls = output.view(output.size(0), nA, (5+nC), nH, nW)
        #print(output.size(0), bs)
        #print(cls.shape)
        cls = cls.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda())).squeeze()
        #print(cls.shape)
        cls = cls.view(bs, cs, nA*nC*nH*nW).transpose(1,2).contiguous().view(bs*nA*nC*nH*nW, cs)

        # Rearrange target and perform filtering operation
        target = target.view(-1, target.size(-1))
        # bef = target.size(0)
        output, target, inds = neg_filter(output, target, withids=True)
        counts, _ = np.histogram(inds, bins=bs, range=(0, bs*cs))

        t0 = time.time()
        nB = output.data.size(0)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
        # cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        # cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data.view(nB*nA*nH*nW) + grid_x               # output format
        pred_boxes[1] = y.data.view(nB*nA*nH*nW) + grid_y
        pred_boxes[2] = torch.exp(w.data).view(nB*nA*nH*nW) * anchor_w
        pred_boxes[3] = torch.exp(h.data).view(nB*nA*nH*nW) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, epoch)
        # Take care of class mask
        cls_num = torch.sum(cls_mask)
        idx_start = 0
        cls_mask_list = []
        tcls_list = []
        for i in range(len(counts)):
            if counts[i] == 0:
                cur_mask = torch.zeros(nA, nH, nW)
                cur_tcls = torch.zeros(nA, nH, nW)
            else:
                cur_mask = torch.sum(cls_mask[idx_start:idx_start+counts[i]], dim=0)
                cur_tcls = torch.sum(tcls[idx_start:idx_start+counts[i]], dim=0)
            cls_mask_list.append(cur_mask)
            tcls_list.append(cur_tcls)
            idx_start += counts[i]
        cls_mask = torch.stack(cls_mask_list)
        tcls = torch.stack(tcls_list)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).float().sum().item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        # cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())
        cls        = cls[Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())].view(-1, cs)  
        tcls = Variable((tcls[cls_mask]).long().cuda())

        ClassificationLoss = nn.CrossEntropyLoss(size_average=False)
       
        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        self.log.write('\n')
        self.log.write(str(cls[0]))
        self.log.write('\n')
        self.log.write(str(tcls[0]))
        self.log.write('\n')
        # # pdb.set_trace()
        # ids = [9,11,12,16]
        # new_cls, new_tcls = select_classes(cls, tcls, ids)
        # new_tcls = Variable(torch.from_numpy(new_tcls).long().cuda())
        # loss_cls_new = self.class_scale * nn.CrossEntropyLoss(size_average=False)(new_cls, new_tcls)
        # loss_cls_new *= 10
        # loss_cls += loss_cls_new

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (epoch, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        self.log.write('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (epoch, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss.item()))
        # print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, cls_new %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), loss_cls_new.item(), loss.item()))
        return loss, (self.num_classes, self.anchors, self.num_anchors)


def select_classes(pred, tgt, ids):
    # convert tgt to numpy
    tgt = tgt.cpu().data.numpy()
    new_tgt = [(tgt == d) * i  for i, d in enumerate(ids)]
    new_tgt = np.max(np.stack(new_tgt), axis=0)
    idxes = np.argwhere(new_tgt > 0).squeeze()
    new_pred = pred[idxes]
    new_pred = new_pred[:, ids]
    new_tgt = new_tgt[idxes]
    return new_pred, new_tgt

def is_dynamic(block):
    return 'dynamic' in block and int(block['dynamic']) == 1

def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks