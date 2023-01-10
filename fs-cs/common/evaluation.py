r""" Evaluation helpers """
import torch

from common.detection import *

class Evaluator:
    hyp = {
        'conf_thresh': 0.0005,
        'nms_thresh': 0.45,
        'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
        'num_anchors': 5,
        'only_objectness': 1
    }
    way = 20
    annots = {}
    fps = None

    @classmethod
    def cls_prediction(cls, cls_score, gt):
        cls_pred = cls_score >= 0.5
        pred_correct = cls_pred == gt
        return pred_correct

    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def seg_prediction(cls, pred_mask, batch, ignore_index=255):
        gt_mask = batch.get('query_mask')

        # Apply ignore_index in PASCAL-5i masks (following evaluation scheme in PFE-Net (TPAMI 2020))
        query_ignore_idx = batch.get('query_ignore_idx')
        if query_ignore_idx is not None:
            assert torch.logical_and(query_ignore_idx, gt_mask).sum() == 0
            query_ignore_idx *= ignore_index
            gt_mask = gt_mask + query_ignore_idx
            pred_mask[gt_mask == ignore_index] = ignore_index

        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union

    @classmethod
    def on_test_end(cls):
        if cls.fps:
            for i in range(cls.way):
                cls.fps[i].close()
            eval(cls.annots, cls.filepath, cls.benchmark, cls.way)

    @classmethod
    def init_det_prediction(cls, benchmark, fold):
        cls.filepath = './output/' + benchmark + '/fold' + str(fold)
        cls.way = 20 if benchmark == 'pascal' else 80
        cls.fps = [0] * cls.way

        for i in range(cls.way):
            buf = '%s/pred_%s.txt' % (cls.filepath, str(i+1))
            cls.fps[i-1] = open(buf, 'w')

    @classmethod
    def det_prediction(cls, output, batch):
        bsz = batch['query_img'].size(0)
        shot = batch['support_imgs'].size(2)

        pred_boxes = get_region_boxes(output, cls.way, shot, cls.hyp)

        for b in range(bsz):
            cls.annots[batch['query_name'][b]] = {'bbox': batch['query_box'][b]}
            width = batch['org_query_imsize'][0][b]
            height = batch['org_query_imsize'][1][b]
            for i in range(cls.way):
                oi = b * cls.way + i
                boxes = pred_boxes[oi]
                boxes = nms(boxes, cls.hyp['nms_thresh'])
                for box in boxes:
                    x1 = (box[0] - box[2]/2.0) * width
                    y1 = (box[1] - box[3]/2.0) * height
                    x2 = (box[0] + box[2]/2.0) * width
                    y2 = (box[1] + box[3]/2.0) * height

                    det_conf = box[4]
                    for j in range((len(box)-5)//2):
                        cls_conf = box[5+2*j]
                        cls_id = box[6+2*j]
                        prob = det_conf * cls_conf
                        cls.fps[i].write('%s %f %f %f %f %f\n' % (batch['query_name'][b], prob, x1, y1, x2, y2))


class AverageMeter:
    """
    A class that logs and averages cls and seg metrics
    """
    def __init__(self, dataset, way, ignore_index=255):
        self.benchmark = dataset.benchmark
        self.way = way
        self.class_ids_interest = torch.tensor(dataset.class_ids)
        self.ignore_index = ignore_index

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80

        self.total_area_inter = torch.zeros((self.nclass + 1, ), dtype=torch.float32)
        self.total_area_union = torch.zeros((self.nclass + 1, ), dtype=torch.float32)
        self.ones = torch.ones((len(self.class_ids_interest), ), dtype=torch.float32)

        self.seg_loss_sum = 0.
        self.seg_loss_count = 0.

        self.cls_loss_sum = 0.
        self.cls_er_sum = 0.
        self.cls_loss_count = 0.
        self.cls_er_count = 0.

        self.det_loss_sum = 0.
        self.det_loss_count = 0.

    def update_det(self, output, loss):
        bsz = output.size(0)

        self.det_loss_sum += float(loss) * bsz
        self.det_loss_count += bsz

    def update_seg(self, pred_mask, batch, loss=None):
        ignore_mask = batch.get('query_ignore_idx')
        gt_mask = batch.get('query_mask')
        support_classes = batch.get('support_classes')

        if ignore_mask is not None:
            pred_mask[ignore_mask == self.ignore_index] = self.ignore_index
            gt_mask[ignore_mask == self.ignore_index] = self.ignore_index

        pred_mask, gt_mask, support_classes = pred_mask.cpu(), gt_mask.cpu(), support_classes.cpu()
        class_dicts = self.return_class_mapping_dict(support_classes)

        samplewise_iou = []  # samplewise iou is for visualization purpose only
        for class_dict, pred_mask_i, gt_mask_i in zip(class_dicts, pred_mask, gt_mask):
            area_inter, area_union = self.intersect_and_union(pred_mask_i, gt_mask_i)

            if torch.sum(gt_mask_i.sum()) == 0:  # no foreground
                samplewise_iou.append(torch.tensor([float('nan')]))
            else:
                samplewise_iou.append(self.nanmean(area_inter[1:] / area_union[1:]))

            self.total_area_inter.scatter_(dim=0, index=class_dict, src=area_inter, reduce='add')
            self.total_area_union.scatter_(dim=0, index=class_dict, src=area_union, reduce='add')

            # above is equivalent to the following:
            '''
            self.total_area_inter[0] += area_inter[0].item()
            self.total_area_union[0] += area_union[0].item()
            for i in range(self.way + 1):
                self.total_area_inter[class_dict[i]] += area_inter[i].item()
                self.total_area_union[class_dict[i]] += area_union[i].item()
            '''

        if loss:
            bsz = float(pred_mask.shape[0])
            self.seg_loss_sum += loss * bsz
            self.seg_loss_count += bsz

        return torch.tensor(samplewise_iou) * 100.

    def nanmean(self, v):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum() / (~is_nan).float().sum()

    def return_class_mapping_dict(self, support_classes):
        # [a, b] -> [0, a, b]
        # relative class index -> absolute class id
        bsz = support_classes.shape[0]
        bg_classes = torch.zeros(bsz, 1).to(support_classes.device).type(support_classes.dtype)
        class_dicts = torch.cat((bg_classes, support_classes), dim=1)
        return class_dicts

    def intersect_and_union(self, pred_mask, gt_mask):
        intersect = pred_mask[pred_mask == gt_mask]
        area_inter = torch.histc(intersect.float(), bins=(self.way + 1), min=0, max=self.way)
        area_pred_mask = torch.histc(pred_mask.float(), bins=(self.way + 1), min=0, max=self.way)
        area_gt_mask = torch.histc(gt_mask.float(), bins=(self.way + 1), min=0, max=self.way)
        area_union = area_pred_mask + area_gt_mask - area_inter
        return area_inter, area_union

    def compute_iou(self):
        # miou does not include bg class
        inter_interest = self.total_area_inter[self.class_ids_interest]
        union_interest = self.total_area_union[self.class_ids_interest]
        iou_interest = inter_interest / torch.max(union_interest, self.ones)
        miou = torch.mean(iou_interest)

        '''
        fiou = inter_interest.sum() / union_interest.sum()
        biou = self.total_area_inter[0].sum() / self.total_area_union[0].sum()
        fbiou = (fiou + biou) / 2.
        '''
        return miou * 100.

    def compute_cls_er(self):
        return self.cls_er_sum / self.cls_er_count * 100. if self.cls_er_count else 0

    def avg_seg_loss(self):
        return self.seg_loss_sum / self.seg_loss_count if self.seg_loss_count else 0

    def avg_cls_loss(self):
        return self.cls_loss_sum / self.cls_loss_count if self.cls_loss_count else 0

    def avg_det_loss(self):
        return self.det_loss_sum / self.det_loss_count if self.det_loss_count else 0
        
    def update_cls(self, pred_cls, gt_cls, loss=None):
        pred_cls, gt_cls = pred_cls.cpu(), gt_cls.cpu()
        pred_correct = pred_cls == gt_cls
        bsz = float(pred_correct.shape[0])
        ''' accuracy '''
        # samplewise_acc = pred_correct.float().mean(dim=1)
        ''' exact ratio '''
        samplewise_er = torch.all(pred_correct, dim=1)
        self.cls_er_sum += samplewise_er.sum()
        self.cls_er_count += bsz

        if loss:
            self.cls_loss_sum += loss * bsz
            self.cls_loss_count += bsz

        return samplewise_er * 100.
