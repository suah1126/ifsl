#import dataset
import torch
import argparse
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from common.detection import *
from common.cfg import cfg
from common.cfg import parse_cfg

from model.metayolo import MetaYoloNetwork
from data.dataset import FSCSDatasetModule
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint, OnlineLogger
from data.pascal import DatasetPASCAL

import os
import pdb
from einops import rearrange



def valid(args, outfile, use_baserw=False):
    m = MetaYoloNetwork.load_from_checkpoint('logs/pascal/fold3/resnet101_recent/best_model-v6.ckpt', args=args)
    m.eval()

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(size=(416, 416)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(img_mean, img_std)])

    dataset = DatasetPASCAL(args.datapath,
                            fold=args.fold,
                            transform=transform,
                            split='trn',
                            way=20,
                            shot=args.shot,
                            task=args.task)
    valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8)

    # metaloader = iter(metaloader)
    n_cls = 15
    enews = [0.0] * n_cls
    cnt = [0.0] * n_cls
    fps = [0]*n_cls

    prefix = './example/output'
    for i, cls_name in enumerate(dataset.CLASSES):
        # if i < 6:
        #     continue
        if i > 15:
            break
        buf = '%s/%s%s.txt' % (prefix, outfile, cls_name)
        fps[i-1] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

    # kkk = 0
    # for batch_idx, batch in enumerate(valid_loader):
    #     clsids = batch['support_classes'][0]
    #     print('generating dynamic weights {}/{}'.format(batch_idx, len(valid_loader)))
    #     support_imgs = rearrange(batch['support_imgs'][0, :], 'n s c h w -> (n s) c h w')
    #     support_boxes_masks = rearrange(batch['support_boxes_masks'][0, :], 'n s c h w -> (n s) c h w')
    #     support_ignore_idxs = batch.get('support_ignore_idxs')
    #     if support_ignore_idxs is not None:
    #         support_ignore_idxs = rearrange(batch['support_ignore_idxs'], 'b n s h w -> (b n s) h w')

    #     spt = torch.cat([support_imgs, support_boxes_masks], dim=1)
    #     dws = m.learner(spt)
    #     dw = dws[0]
        
    #     for ci, c in enumerate(clsids):
    #         c = c-6
    #         enews[c] = enews[c] * cnt[c] / (cnt[c] + 1) + dw[ci] / (cnt[c] + 1)
    #         cnt[c] += 1

    #     dynamic_weights = [torch.stack(enews, dim=0)]


    for batch_idx, batch in enumerate(valid_loader):
        #output = m.backbone(batch['query_img'], dynamic_weights)
        output = m(batch)
        print(f'progress: {batch_idx}/{len(valid_loader)}')
        if isinstance(output, tuple):
            output = (output[0].data, output[1].data)
        else:
            output = output.data

        # import pdb; pdb.set_trace()
        batch_boxes = get_region_boxes_v2(output, n_cls, conf_thresh, 1, anchors, 5, 0, 1)

        if isinstance(output, tuple):
            bs = output[0].size(0)
        else:
            assert output.size(0) % n_cls == 0
            bs = output.size(0) // n_cls

        for b in range(bs):
            width = batch['org_query_imsize'][1][b]
            height = batch['org_query_imsize'][0][b]
            for i in range(n_cls):
                # oi = i * bs + b
                oi = b * n_cls + i
                boxes = batch_boxes[oi]
                boxes = nms(boxes, nms_thresh)
                for box in boxes:
                    x1 = (box[0] - box[2]/2.0) * width
                    y1 = (box[1] - box[3]/2.0) * height
                    x2 = (box[0] + box[2]/2.0) * width
                    y2 = (box[1] + box[3]/2.0) * height

                    det_conf = box[4]
                    for j in range((len(box)-5)//2):
                        cls_conf = box[5+2*j]
                        cls_id = box[6+2*j]
                        prob =det_conf * cls_conf
                        fps[i].write('%s %f %f %f %f %f\n' % (batch['query_name'][b], prob, x1, y1, x2, y2))
                        print(batch['query_name'][b], dataset.CLASSES[i+1], prob, x1, y1, x2, y2)
    for i in range(n_cls):
        fps[i].close()

    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description='Methods for Integrative Few-Shot Classification and Segmentation')
    parser.add_argument('--datapath', type=str, default='/home/suachoi/ifsl/datasets', help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--method', type=str, default='asnet', choices=['panet', 'pfenet', 'hsnet', 'asnet', 'metayolo'], help='FS-CS methods')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco'], help='Experiment benchmark')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--way', type=int, default=15, help='N-way for K-shot evaluation episode')
    parser.add_argument('--shot', type=int, default=1, help='K-shot for N-way K-shot evaluation episode: fixed to 1 for training')
    parser.add_argument('--bsz', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--niter', type=int, default=2000, help='Max iterations')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='4-fold validation fold')
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone CNN network')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--weak', action='store_true', help='Flag to train with cls (weak) labels -- reduce learning rate by 10 times') 
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg', 'det'], help='If classification, reduce learning rate by 10 times')
    parser.add_argument('--resume', action='store_true', help='Flag to resume a finished run')
    parser.add_argument('--vis', action='store_true', help='Flag to visualize. Use with --eval')
    args = parser.parse_args()
    outfile = 'comp4_det_test_'
    valid(args, outfile)

    # if len(sys.argv) in [5,6,7]:
    #     datacfg = sys.argv[1]
    #     darknet = parse_cfg(sys.argv[2])
    #     learnet = parse_cfg(sys.argv[3])
    #     weightfile = sys.argv[4]
    #     if len(sys.argv) >= 6:
    #         gpu = sys.argv[5]
    #     else:
    #         gpu = '0'
    #     if len(sys.argv) == 7:
    #         use_baserw = True
    #     else:
    #         use_baserw = False

    #     data_options  = read_data_cfg(datacfg)
    #     net_options   = darknet[0]
    #     meta_options  = learnet[0]
    #     data_options['gpus'] = gpu
    #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    #     # Configure options
    #     cfg.config_data(data_options)
    #     cfg.config_meta(meta_options)
    #     cfg.config_net(net_options)

    #     outfile = 'comp4_det_test_'
    #     valid(datacfg, darknet, learnet, weightfile, outfile, use_baserw)
    # else:
    #     print('Usage:')
    #     print(' python valid.py datacfg cfgfile weightfile')
