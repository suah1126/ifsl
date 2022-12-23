r""" Pascal-5i few-shot classification and segmentation dataset """
import os
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import torchvision
import numpy as np


class DatasetPASCAL(Dataset):
    """
    FS-CS Pascal-5i dataset of which split follows the standard FS-S dataset
    """
    def __init__(self, datapath, fold, transform, split, way, shot, task):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        #self.split = 'trn'
        self.nfolds = 4
        self.nclass = 20
        self.nbox = 50  # max box number per image
        self.benchmark = 'pascal'
        self.CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor')

        self.PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

        self.fold = fold
        self.way = way
        self.shot = shot
        self.task = task

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.bbox_path = os.path.join(datapath, 'VOC2012/Annotations/')
        
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000
        #return 1280

    def __getitem__(self, idx):
        query_name, support_names, _support_classes = self.sample_episode(idx)
        query_img, query_cmask, query_cbox, support_imgs, support_cmasks, support_cboxes, org_qry_imsize = self.load_frame(query_name, support_names)               

        query_class_presence = [s_c in torch.unique(query_cmask) for s_c in _support_classes]  # needed - 1
        rename_class = lambda x: _support_classes.index(x) +1 if x in _support_classes else 0
        
        query_img = self.transform(query_img)
        query_box = self.get_query_box(org_qry_imsize, query_img.shape, query_cbox, rename_class)
        query_mask, query_ignore_idx = self.get_query_mask(query_img, query_cmask, rename_class)
        support_imgs = torch.stack([torch.stack([self.transform(support_img) for support_img in support_imgs_c]) for support_imgs_c in support_imgs])
        support_boxes, support_boxes_masks, support_masks, support_ignore_idxs = self.get_support_masks(support_imgs, _support_classes, support_cboxes, support_cmasks, rename_class)
        
        _support_classes = torch.tensor(_support_classes)
        query_class_presence = torch.tensor(query_class_presence)

        assert query_class_presence.int().sum() == (len(torch.unique(query_mask)) - 1)

#        print(query_name) : 000054352
        # img_mean = [0.485, 0.456, 0.406]
        # img_std = [0.229, 0.224, 0.225]
        # topilimage = torchvision.transforms.ToPILImage()
        # img_pil = query_img * torch.tensor(img_std).view(3, 1, 1)
        # img_pil = img_pil + torch.tensor(img_mean).view(3, 1, 1)
        # img_pil = topilimage(img_pil).convert("RGB")
        # box_query_img = ImageDraw.Draw(img_pil)
        # for i in range(len(query_box)):
        #     object_box = [query_box[i][1]-query_box[i][3]/2,
        #                 query_box[i][2]-query_box[i][4]/2,
        #                 query_box[i][1]+query_box[i][3]/2,
        #                 query_box[i][2]+query_box[i][4]/2]
        #     rescaled_object_box = [object_box[0] * query_img.shape[1], 
        #                             object_box[1] * query_img.shape[2],
        #                             object_box[2] * query_img.shape[1],
        #                             object_box[3] * query_img.shape[2]]
        #     box_query_img.rectangle(rescaled_object_box, outline=(0, 255, 0), width=3)
        #     box_query_img.text((rescaled_object_box[0], rescaled_object_box[1]), str(query_box[i][0]))
        # img_pil.save('./example/'+query_name+'.jpg', 'JPEG')
        
        # for clsid in range(15):
        #     for (s_img, object_box) in zip(support_imgs[clsid], support_boxes[clsid]):
        #         img_pil = s_img * torch.tensor(img_std).view(3, 1, 1)
        #         img_pil = img_pil + torch.tensor(img_mean).view(3, 1, 1)
        #         s_img_pil = topilimage(img_pil)
        #         box_support_img = ImageDraw.Draw(s_img_pil)
        #         box_support_img.rectangle(list(object_box[1:]), outline=(0, 255, 0), width=3)
        #         box_support_img.text((object_box[1], object_box[2]), str(object_box[0]))
        #         s_img_pil.save('./example/s_'+query_name+'_'+ str(clsid)+'.jpg', 'JPEG')      

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_box': query_box, 
                 'query_name': query_name,
                 'query_ignore_idx': query_ignore_idx,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_boxes': support_boxes,
                 'support_boxes_masks': support_boxes_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'support_classes': _support_classes,
                 'query_class_presence': query_class_presence}

        return batch
    
    # def collate_fn(self, batch):
    #     collate_batch = {'query_img': torch.stack([b['query_img'] for b in batch], dim=0),
    #          'query_mask': torch.stack([b['query_mask'] for b in batch], dim=0),
    #          'query_box': [b['query_box'] for b in batch], 
    #          'query_name': [b['query_name'] for b in batch],
    #          'query_ignore_idx': [b['query_ignore_idx'] for b in batch],

    #          'org_query_imsize': [b['org_query_imsize'] for b in batch],

    #          'support_imgs': torch.stack([b['support_imgs'] for b in batch], dim=0),
    #          'support_masks': torch.stack([b['support_masks'] for b in batch], dim=0),
    #          'support_boxes': [b['support_boxes'] for b in batch],
    #          'support_boxes_masks': torch.stack([b['support_boxes_masks'] for b in batch], dim=0),
    #          'support_names': [b['support_names'] for b in batch],
    #          'support_ignore_idxs': torch.stack([b['support_ignore_idxs'] for b in batch], dim=0),

    #          'support_classes': torch.stack([b['support_classes'] for b in batch], dim=0),
    #          'query_class_presence': torch.stack([b['query_class_presence'] for b in batch], dim=0)}
        
    #     return collate_batch
    
    def get_query_box(self, org_imsize, query_imsize, query_cbox, rename_class):
        # x_ratio = query_imsize[1] / org_imsize[0]
        # y_ratio = query_imsize[2] / org_imsize[1]
        x_ratio = 1./org_imsize[0]
        y_ratio = 1./org_imsize[1]
        query_box = np.zeros((self.way, self.nbox, 5))
        cls_idx = [0] * self.way

        for box in query_cbox:
            clsid = rename_class(box[0])
            if clsid==0:
                continue
            x1 = min(0.999, box[1] * x_ratio) 
            y1 = min(0.999, box[2] * y_ratio) 
            x2 = min(0.999, box[3] * x_ratio)
            y2 = min(0.999, box[4] * y_ratio)

            box[1] = (x1 + x2)/2.
            box[2] = (y1 + y2)/2
            box[3] = x2 - x1
            box[4] = y2 - y1

            # if self.split == 'trn':
            #     query_box[clsid-1][cls_idx[clsid-1]] = [clsid, int(box[1]*x_ratio), int(box[2]*y_ratio), int(box[3]*x_ratio), int(box[4]*y_ratio)]
            # else:
            #     query_box[clsid-1][cls_idx[clsid-1]] = [clsid, box[1], box[2], box[3], box[4]]
            query_box[clsid-1][cls_idx[clsid-1]] = box
            query_box[clsid-1][cls_idx[clsid-1]][0] = clsid -1
            cls_idx[clsid-1] += 1
            if sum(cls_idx) >= 50:
                break
        query_box = np.reshape(query_box, (self.way, -1))

        return query_box

    def get_query_mask(self, query_img, query_cmask, rename_class):
        # if self.split == 'trn':  # resize during training and retain orignal sizes during validation
        #     query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask, query_ignore_idx = self.generate_query_episodic_mask(query_cmask.float(), rename_class)
        return query_mask, query_ignore_idx

    def get_support_masks(self, support_imgs, _support_classes, support_cboxes, support_cmasks, rename_class):
        support_masks = []
        support_ignore_idxs = []
        support_boxes = []
        support_boxes_masks = []

        for class_id, scbox_c, scmask_c in zip(_support_classes, support_cboxes, support_cmasks):  # ways
            support_boxes_c = []
            support_boxes_masks_c = []
            support_masks_c = []
            support_ignore_idxs_c = []
            for scbox, scmask in zip(scbox_c, scmask_c):  # shots
                scbox, scbox_mask = self.get_support_boxes(scbox, scmask.size()[-2:], support_imgs.size()[-2:], class_id, rename_class)
                scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
                support_mask, support_ignore_idx = self.generate_support_episodic_mask(scmask, class_id, rename_class)
                assert len(torch.unique(support_mask)) <= 2, f'{len(torch.unique(support_mask))} labels in support'
                support_boxes_c.append(scbox)
                support_boxes_masks_c.append(scbox_mask)
                support_masks_c.append(support_mask)
                support_ignore_idxs_c.append(support_ignore_idx)
            support_boxes.append(support_boxes_c)
            support_boxes_masks.append(torch.stack(support_boxes_masks_c))
            support_masks.append(torch.stack(support_masks_c))
            support_ignore_idxs.append(torch.stack(support_ignore_idxs_c))
        support_boxes_masks = torch.stack(support_boxes_masks)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)
        return support_boxes, support_boxes_masks, support_masks, support_ignore_idxs

    # class_id 외의 labeldms 0으로 만들고 episode label로 rename함.
    # metayolo는 box mask를 사용하기 때문에 이것도 만들어줌.
    def get_support_boxes(self, support_box, org_imsize, support_imsize, class_id, rename_class):
        y_ratio = support_imsize[0] / org_imsize[0]
        x_ratio = support_imsize[1] / org_imsize[1]

        for box in support_box:
            if box[0] == class_id:
                box_renamed = torch.tensor([rename_class(box[0]), int(box[1]*x_ratio), int(box[2]*y_ratio), int(box[3]*x_ratio), int(box[4]*y_ratio)])
                break

        box_mask = torch.zeros((1, support_imsize[0], support_imsize[1]))
        box_mask[:, box_renamed[2]:box_renamed[4], box_renamed[1]:box_renamed[3]] = 1   # [1, ymin:ymax, xmin:xmax]

        return box_renamed, box_mask

    # class_id 외의 label은 0으로 만들고 episode label로 rename함.
    def generate_query_episodic_mask(self, mask, rename_class):
        # mask = mask.clone()
        mask_renamed = torch.zeros_like(mask).to(mask.device).type(mask.dtype)
        boundary = (mask / 255).floor()

        classes = torch.unique(mask)
        for c in classes:
            mask_renamed[mask == c] = 0 if c in [0, 255] else rename_class(c)

        return mask_renamed, boundary

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        mask = mask.clone()
        boundary = (mask / 255).floor()
        mask[mask != class_id] = 0
        mask[mask == class_id] = rename_class(class_id)

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img  = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        query_box = self.read_box(query_name)
        support_imgs  = [[self.read_img(name)  for name in support_names_c] for support_names_c in support_names]
        support_masks = [[self.read_mask(name) for name in support_names_c] for support_names_c in support_names]
        support_boxes = [[self.read_box(name) for name in support_names_c] for support_names_c in support_names]
    
        org_qry_imsize = query_img.size

        return query_img, query_mask, query_box, support_imgs, support_masks, support_boxes, org_qry_imsize

    def read_box(self, img_name):
        r"""Return bounding box in PIL Image"""
        boxes = []
        for i, obj in enumerate(ET.parse(os.path.join(self.bbox_path, img_name) + '.xml').getroot().findall('object')):
            if i > 49:
                break
            xmlbox = obj.find('bndbox')
            boxes.append([self.CLASSES.index(obj.find('name').text), 
                int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)
                ])

        return boxes

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, query_class = self.img_metadata[idx]

        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []

        # p encourage the support classes sampled as the query_class by the prob of 0.5
        if self.task != 'det':
            p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
            p[self.class_ids.index(query_class)] = 1 / 2.
            support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()
        else: #모든 support class에서 1개씩 가져옴. shot=1
            support_classes = self.class_ids

        for sc in support_classes:
            support_names_c = []
            while True:  # keep sampling support set if query == support
                support_name = np.random.choice(self.img_metadata_classwise[sc], 1, replace=False)[0]
                if query_name != support_name and support_name not in support_names_c:
                    support_names_c.append(support_name)
                if len(support_names_c) == self.shot:
                    break
            support_names.append(support_names_c)

        return query_name, support_names, support_classes

    def build_class_ids(self):
        # fs-cs class_ids
        nclass_val = self.nclass // self.nfolds
        # e.g. fold0 val: 1, 2, 3, 4, 5
        class_ids_val = [self.fold * nclass_val + i for i in range(1, nclass_val + 1)]
        # e.g. fold0 trn: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        # fs-d class_ids
        # class_ids_val = [self.CLASSES.index(novel_class) + 1 for novel_class in self.NOVEL_CLASSES[self.novelset]]
        # class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        assert len(set(class_ids_trn + class_ids_val)) == self.nclass
        assert 0 not in class_ids_val
        assert 0 not in class_ids_trn

        #return class_ids_trn
        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):           
        # val: 0 / train: 1,2,3
        # test, finetune: novel / train: base

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(f'data/splits/pascal/{split}/fold{fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1])] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print(f'Total {self.split} images are : {len(img_metadata):,}')

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1, self.nclass + 1):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]

        # img_metadata_classwise.keys(): [1, 2, ..., 20]
        assert 0 not in img_metadata_classwise.keys()
        assert self.nclass in img_metadata_classwise.keys()

        return img_metadata_classwise

'''
base-novel disjoint
query마다 support set
class_id 고정되어있는데 0~14로 사용하긴 함. 어차피 ifsl처럼 batch마다 달라지는 set이 아니라서 그대로 써도 될듯.
'''