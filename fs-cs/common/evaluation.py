r""" Evaluation helpers """
import torch
import os
from torch.autograd import Variable
from common.detection import *

class Evaluator:
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
        self.total_fp = []
        self.total_tp = []
        self.total_bsz = 0

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

    def avg_precision(self):
        fp = np.cumsum(self.total_fp)
        tp = np.cumsum(self.total_tp)
        rec = tp / float(self.total_bsz)
        
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec)
        return ap
        # print(self.correct, self.total)
        # precision = 1.0*self.correct/(self.proposals+1e-5)
        # recall = 1.0*self.correct/(self.total+1e-5)
        # fscore = 2.0*precision*recall/(precision+recall+1e-5)

        # return precision, recall, fscore

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

    def update_det(self, output, target, loss, training, num_classes, anchors, num_anchors, img_size, conf_thresh=0.02, nms_thresh=0.4, iou_thresh=0.5):
        if True:
            #bsz = output.size(0) // self.way  
            if training:
                bsz = output.size(0) // self.way       
            else:
                bsz = 1 
        else:
            bsz = 1
            way = 5
            bsz = output.size(0) // self.way  
            way = 15
            fps = [[] for _ in range(way)]
            all_boxes = get_region_boxes_v2(output, way, conf_thresh, num_classes, anchors, num_anchors, 0, 1)
            for b in range(bsz):
                width = img_size[0][b]
                height = img_size[1][b]
                for i in range(way):
                    # oi = i * bs + b
                    oi = b * way + i
                    boxes = all_boxes[oi]
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
                            prob = det_conf * cls_conf
                            fps[cls_id].append([b, prob, x1, y1, x2, y2])        # b for image id

            for c in range(way):
                image_ids = [x[0] for x in fps[c]]
                confidence = np.array([float(x[1]) for x in fps[c]])
                BB = np.array([[float(z) for z in x[2:]] for x in fps[c]])

                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :] if len(BB) != 0 else BB
                image_ids = [image_ids[x] for x in sorted_ind]

                # go down dets and mark TPs and FPs
                nd = len(image_ids)
                tp = np.zeros(nd)
                fp = np.zeros(nd)
                for d in range(nd):
                    R = target[image_ids[d]][c].view(50, 5)        #  몇번쨰 batch img 들고올지
                    width = img_size[0][image_ids[d]].cpu().detach().numpy()
                    height = img_size[1][image_ids[d]].cpu().detach().numpy()
                    bb = BB[d, :].astype(float)
                    ovmax = -np.inf

                    BBGT = R[R[:, 0]==(c+1)][:, 1:].cpu().detach().numpy()
                    detected = np.zeros(len(BBGT))

                    if len(BBGT) > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(BBGT[:, 0] * width, bb[0])
                        iymin = np.maximum(BBGT[:, 1] * height, bb[1])
                        ixmax = np.minimum(BBGT[:, 2] * width, bb[2])
                        iymax = np.minimum(BBGT[:, 3] * height, bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > iou_thresh:
                        if not detected[jmax]:
                            tp[d] = 1.
                            detected[jmax] = 1
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                # compute precision recall
                self.total_fp.extend(fp)
                self.total_tp.extend(tp)
                self.total_bsz += bsz

            # for i in range(output.size(0)):
            #     boxes = torch.tensor(all_boxes[i]).cuda()
            #     boxes = self.nms(boxes, nms_thresh)
            #     truths = target.view(-1, target.size(2))
            #     num_gts = self.truths_length(truths)
        
            #     self.total = self.total + num_gts
        
            #     for j in range(len(boxes)):
            #         if boxes[j][4] > conf_thresh:
            #             self.proposals = self.proposals+1

            #     for k in range(num_gts):
            #         box_gt = [truths[k][1], truths[k][2], truths[k][3], truths[k][4], 1.0, 1.0, truths[k][0]]
            #         best_iou = 0
            #         best_j = -1
            #         for j in range(len(boxes)):
            #             iou = self.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
            #             print(iou)
            #             if iou > best_iou:
            #                 best_j = j
            #                 best_iou = iou
            #         if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
            #             self.correct = self.correct+1

        self.det_loss_sum += float(loss) * bsz
        self.det_loss_count += bsz

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def get_region_boxes(self, output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
        anchor_step = len(anchors)//num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert(output.size(1) == (5+num_classes)*num_anchors)
        h = output.size(2)
        w = output.size(3)

        all_boxes = []
        output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)


        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)
        
        sz_hw = h*w
        sz_hwa = sz_hw*num_anchors
        det_confs = torch.FloatTensor(det_confs.size()).copy_(det_confs)
        cls_max_confs = torch.FloatTensor(cls_max_confs.size()).copy_(cls_max_confs)
        cls_max_ids = torch.LongTensor(cls_max_ids.size()).copy_(cls_max_ids)
        xs = torch.FloatTensor(xs.size()).copy_(xs)
        ys = torch.FloatTensor(ys.size()).copy_(ys)
        ws = torch.FloatTensor(ws.size()).copy_(ws)
        hs = torch.FloatTensor(hs.size()).copy_(hs)
        if validation:
            cls_confs = torch.FloatTensor(det_confs.size()).copy_(det_confs)(cls_confs.view(-1, num_classes))
        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs[ind]
                        if only_objectness:
                            conf =  det_confs[ind]
                        else:
                            conf = det_confs[ind] * cls_max_confs[ind]
        
                        if conf > conf_thresh:
                            bcx = xs[ind]
                            bcy = ys[ind]
                            bw = ws[ind]
                            bh = hs[ind]
                            cls_max_conf = cls_max_confs[ind]
                            cls_max_id = cls_max_ids[ind]
                            box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                        box.append(tmp_conf)
                                        box.append(c)
                            boxes.append(box)
            all_boxes.append(boxes)

        return all_boxes

    def nms(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        det_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1-boxes[i][4]                

        _,sortIds = torch.sort(det_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i+1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if self.bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[4] = 0
        return out_boxes

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = min(box1[0], box2[0])
            Mx = max(box1[2], box2[2])
            my = min(box1[1], box2[1])
            My = max(box1[3], box2[3])
            w1 = box1[2] - box1[0]
            h1 = box1[3] - box1[1]
            w2 = box2[2] - box2[0]
            h2 = box2[3] - box2[1]
        else:
            mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
            Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
            my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
            My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
            w1 = box1[2]
            h1 = box1[3]
            w2 = box2[2]
            h2 = box2[3]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        carea = 0
        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea/uarea

    def truths_length(self, truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i