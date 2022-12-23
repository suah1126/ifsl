from functools import reduce
from operator import add

import math
import torch
from einops import rearrange

from model.ifsl import iFSLModule
from model.module.metayolo import Darknet, MetaYoloLearner, RegionLossV2


class MetaYoloNetwork(iFSLModule):
    def __init__(self, args):
        super(MetaYoloNetwork, self).__init__(args)

        # self.backbone = Darknet()
        # self.learner = MetaYoloLearner()
        self.learner = Darknet()
        self.loss = self.learner.loss
        self.lr = 0.0001 / 3.0
        self.processed_batches = 0
        self.learner.print_network()
        
    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, shot, 3, H, W]
        support_boxes_masks.shape : [bsz, way, shot, 1, H, W]
        support_masks.shape : [bsz, way, shot, H, W]
        '''
        self.processed_batches += 1
        # support_imgs = rearrange(batch['support_imgs'], 'b n s c h w -> (b n s) c h w')
        # support_boxes_masks = rearrange(batch['support_boxes_masks'], 'b n s c h w -> (b n s) c h w')
        # support_ignore_idxs = batch.get('support_ignore_idxs')
        # if support_ignore_idxs is not None:
        #     support_ignore_idxs = rearrange(batch['support_ignore_idxs'], 'b n s h w -> (b n s) h w')
        support_imgs = rearrange(batch['support_imgs'][0, :], 'n s c h w -> (n s) c h w')
        support_boxes_masks = rearrange(batch['support_boxes_masks'][0, :], 'n s c h w -> (n s) c h w')
        # support_imgs = rearrange(batch['support_imgs'][:2, :], 'b n s c h w -> (b n s) c h w')
        # support_boxes_masks = rearrange(batch['support_boxes_masks'][:2, :], 'b n s c h w -> (b n s) c h w')
        support_ignore_idxs = batch.get('support_ignore_idxs')
        if support_ignore_idxs is not None:
            support_ignore_idxs = rearrange(batch['support_ignore_idxs'], 'b n s h w -> (b n s) h w')
        
        query_img = batch['query_img']
        spt = torch.cat([support_imgs, support_boxes_masks], dim=1)

        return self.learner(query_img, spt)
        # dynamic_weights = self.learner(spt)
        # return self.backbone(query_img, dynamic_weights)
    
    def compute_det_objective(self, output, target):
        return self.loss(output, target, self.current_epoch)

    def configure_optimizers(self):
        ''' Taken from authors' official implementation '''
        lr = 0.0001 / 3.0 / 64
        return torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=0.0005)

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # update params
    #     optimizer.step(closure=optimizer_closure)
    #     steps=[-1,6,11,200]
    #     scales=[0.1,10,.1,.1]
    #     lr = 0.1 / 3.0 / 64
    #     for i in range(len(steps)):
    #         scale = scales[i] if i < len(scales) else 1
    #         if epoch >= steps[i]:
    #             lr = lr * scale
    #             if epoch == steps[i]:
    #                 break
    #         else:
    #             break
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr/64
    
    def train_mode(self):
        self.train()