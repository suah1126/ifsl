r""" COCO-20i few-shot classification and segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from pycocotools.coco import COCO
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import torchvision
import numpy as np


class DatasetCOCO(Dataset):
    """
    FS-CS COCO-20i dataset of which split follows the standard FS-S dataset
    """
    def __init__(self, datapath, fold, transform, split, way, shot):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'

        self.fold = fold
        self.way = way
        self.shot = shot
        self.split_coco = 'train2014' if split == 'trn' else 'val2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.coco = COCO((os.path.join(self.base_path, 'annotations', 'instances_' + self.split_coco + '.json')))
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()  # keys: 1, 2, ..., 79, 80
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_name, support_names, _support_classes = self.sample_episode(idx)
        query_img, query_cmask, query_cbox, support_imgs, support_cmasks, support_cboxes, org_qry_imsize = self.load_frame(query_name, support_names)

        query_class_presence = [s_c in torch.unique(query_cmask) for s_c in _support_classes]  # needed - 1
        rename_class = lambda x: _support_classes.index(x) + 1 if x in _support_classes else 0

        query_img = self.transform(query_img)
        query_box = self.get_query_box(org_qry_imsize, query_img.shape, query_cbox, rename_class)
        query_mask = self.get_query_mask(query_img, query_cmask, rename_class)
        support_imgs = torch.stack([torch.stack([self.transform(support_img) for support_img in support_imgs_c]) for support_imgs_c in support_imgs])
        support_boxes, support_masks = self.get_support_masks(support_imgs, _support_classes, support_cboxes, support_cmasks, rename_class)

        _support_classes = torch.tensor(_support_classes)
        query_class_presence = torch.tensor(query_class_presence)

        #print(query_name) # 000053462.jpg
        # name = query_name[-16:].strip("0")[:-4]
        # topilimage = torchvision.transforms.ToPILImage()
        # img_pil = topilimage(query_img)
        # box_query_img = ImageDraw.Draw(img_pil)
        # for object_box in query_box:
        #     box_query_img.rectangle([object_box['bbox'][0], object_box['bbox'][1], object_box['bbox'][0]+object_box['bbox'][2], object_box['bbox'][1]+object_box['bbox'][3]], outline=(0, 255, 0), width=3)
        # img_pil.save('./example_coco/'+name+'.jpg', 'JPEG')
        
        # for (s_img, object_box) in zip(support_imgs[0], support_boxes[0]):
        #     s_img_pil = topilimage(s_img)
        #     box_support_img = ImageDraw.Draw(s_img_pil)
        #     for bbox in object_box:
        #         box_support_img.rectangle(bbox['bbox'], outline=(0, 255, 0), width=3)
        # img_pil.save('./example_coco/s_'+query_name+'.jpg', 'JPEG')
        
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_box': query_box, 
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_boxes': support_boxes,
                 'support_names': support_names,

                 'support_classes': _support_classes,
                 'query_class_presence': query_class_presence}

        return batch

    def collate_fn(self, batch):
        collate_batch = {'query_img': torch.stack([b['query_img'] for b in batch], dim=0),
             'query_mask': torch.stack([b['query_mask'] for b in batch], dim=0),
             'query_box': [b['query_box'] for b in batch], 
             'query_name': [b['query_name'] for b in batch],

             'org_query_imsize': [b['org_query_imsize'] for b in batch],

             'support_imgs': torch.stack([b['support_imgs'] for b in batch], dim=0),
             'support_masks': torch.stack([b['support_masks'] for b in batch], dim=0),
             'support_boxes': [b['support_boxes'] for b in batch],
             'support_names': [b['support_names'] for b in batch],

             'support_classes': torch.stack([b['support_classes'] for b in batch], dim=0),
             'query_class_presence': torch.stack([b['query_class_presence'] for b in batch], dim=0)}
        
        return collate_batch

    def build_class_ids(self):
        nclass_val = self.nclass // self.nfolds
        # e.g. fold0 val: 1, 5, 9, ..., 77
        class_ids_val = [self.fold + self.nfolds * v + 1 for v in range(nclass_val)]
        # e.g. fold0 trn: 2, 3, 4, 6, 7, 8, 10, 11, 12, ..., 78, 79, 80
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        assert len(set(class_ids_trn + class_ids_val)) == self.nclass
        assert 0 not in class_ids_val
        assert 0 not in class_ids_trn

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata_classwise(self):
        img_metadata_classwise = dict()

        with open(f'./data/splits/coco/{self.split}/fold{self.fold}.pkl', 'rb') as f:
            # class ids: 0, 1, 2, ..., 79
            img_metadata_classwise_temp = pickle.load(f)

            # class ids: 1, 2, 3, ..., 80
            for k in img_metadata_classwise_temp.keys():
                img_metadata_classwise[k + 1] = img_metadata_classwise_temp[k]

        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]

        print(f'Total {self.split} images are : {len(img_metadata):,}')

        return sorted(list(set(img_metadata)))

    def get_query_box(self, org_qry_imsize, query_imsize, query_cbox, rename_class):
        x_ratio = query_imsize[1] / org_qry_imsize[0]
        y_ratio = query_imsize[2] / org_qry_imsize[1]

        for box in query_cbox:
            if self.split == 'trn':
                box['bbox'] = (int(box['bbox'][0]*x_ratio), int(box['bbox'][1]*y_ratio), int(box['bbox'][2]*x_ratio), int(box['bbox'][3]*y_ratio))
            box['category_id'] = rename_class(box['category_id'])

        return query_cbox

    def get_query_mask(self, query_img, query_cmask, rename_class):
        if self.split == 'trn':  # resize during training and retain orignal sizes during validation
            query_cmask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        query_mask = self.generate_query_episodic_mask(query_cmask.float(), rename_class)
        return query_mask

    def get_support_boxes(self, support_box, img_size, support_img_size, rename_class):
        x_ratio = support_img_size[0] / img_size[0]
        y_ratio = support_img_size[1] / img_size[1]
        
        for box in support_box:
            box['bbox'] = (int(box['bbox'][0]*x_ratio), int(box['bbox'][1]*y_ratio), int(box['bbox'][2]*x_ratio), int(box['bbox'][3]*y_ratio))
            box['category_id'] = rename_class(box['category_id'])

        return support_box

    def get_support_masks(self, support_imgs, _support_classes, support_cboxes, support_cmasks, rename_class):
        support_masks = []
        support_boxes = []

        for class_id, scbox_c, scmask_c in zip(_support_classes, support_cboxes, support_cmasks):  # ways
            support_boxes_c = []
            support_masks_c = []
            for scbox, scmask in zip(scbox_c, scmask_c):  # shots
                scbox = self.get_support_boxes(scbox, scmask.size()[-2:], support_imgs.size()[-2:], rename_class)
                scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
                support_mask = self.generate_support_episodic_mask(scmask, class_id, rename_class)
                assert len(torch.unique(support_mask)) <= 2, f'{len(torch.unique(support_mask))} labels in support'
                support_boxes_c.append(scbox)
                support_masks_c.append(support_mask)
            support_boxes.append(support_boxes_c)
            support_masks.append(torch.stack(support_masks_c))
        support_masks = torch.stack(support_masks)
        return support_boxes, support_masks

    def generate_query_episodic_mask(self, mask, rename_class):
        # mask = mask.clone()
        mask_renamed = torch.zeros_like(mask).to(mask.device).type(mask.dtype)

        classes = torch.unique(mask)
        for c in classes:
            mask_renamed[mask == c] = 0 if c == 0 else rename_class(c)

        return mask_renamed

    def generate_support_episodic_mask(self, mask, class_id, rename_class):
        mask = mask.clone()
        mask[mask != class_id] = 0
        mask[mask == class_id] = rename_class(class_id)

        return mask

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
        boxes = []
        img_id = int(img_name[-16:].strip("0")[:-4])
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        for box in self.coco.loadAnns(ann_ids):
            boxes.append(box)
        return boxes

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def read_img(self, img_name):
        return Image.open(os.path.join(self.base_path, img_name)).convert('RGB')

    def sample_episode(self, idx):
        # Fix (q, s) pair for all queries across different batch sizes for reproducibility
        if self.split == 'val':
            np.random.seed(idx)

        query_class = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[query_class], 1, replace=False)[0]

        # 3-way 2-shot support_names: [[c1_1, c1_2], [c2_1, c2_2], [c3_1, c3_2]]
        support_names = []

        # p encourage the support classes sampled as the query_class by the prob of 0.5
        p = np.ones([len(self.class_ids)]) / 2. / float(len(self.class_ids) - 1)
        p[self.class_ids.index(query_class)] = 1 / 2.
        support_classes = np.random.choice(self.class_ids, self.way, p=p, replace=False).tolist()
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
