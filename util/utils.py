# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random
import itertools
import json
from math import sqrt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from util.box_ops import calc_iou_tensor, box_cxcywh_to_ltrb

'''
custom
'''

def load_chinese(dataPath, vocab=None):
    data_list = [] 
    if vocab is None:
        char2idx = {'background':0,
                    'unknown':1}
        idx2char = {0:'background',
                    1:'unknown'}
    else:
        char2idx = vocab['char2idx']
        idx2char = vocab['idx2char']

    for d in Path(dataPath).glob('*'):
        if not d.is_dir():
            continue
        for jsonFile in d.glob('*.json'):
            with jsonFile.open('r',encoding='utf-8') as f:
                data = json.load(f)
            bbox_list = []
            for coords in data['Image_Text_Coord']:
                for coord, char in coords:
                    lx, ly, width, height, _, _ = coord
                    try:
                        uid = char2idx[char]
                        idx2char[uid] = char
                    except:
                        if vocab is None:
                            char2idx[char] = len(char2idx)
                            uid = char2idx[char]
                            idx2char[uid] = char 
                        else:
                            uid = char2idx['unknown']
                    bbox_list.append([uid, lx, ly, width, height])
            img_path = Path(dataPath) / d.name / (jsonFile.stem + '.jpg')
            data_list.append([str(img_path), bbox_list])

    vocab = { 'char2idx' : char2idx, 'idx2char' : idx2char }
    return data_list, vocab

def load_yethangul(dataPath):
    random.seed(42)
    data_list = []
    char2idx = {'background':0}
    idx2char = {0:'background'}
    for d in Path(dataPath).glob('*'):
        for jsonFile in d.glob('*.json'):
            with jsonFile.open('r',encoding='utf-8-sig') as f:
                data = json.load(f)
            bbox_list = []
            for coord in data['Text_Coord']:
                lx, ly, width, height, _, _ = coord['bbox']
                char = coord['annotate']
                try:
                    uid = char2idx[char]
                    idx2char[uid] = char
                except:
                    char2idx[char] = len(char2idx)
                    uid = char2idx[char]
                    idx2char[uid] = char 
                bbox_list.append([uid, lx, ly, width, height])
            img_path = Path(dataPath).parent / '이미지데이터' / d.name / (data['Image_filename'] + '.png')
            data_list.append([str(img_path),bbox_list])
    random.shuffle(data_list)
    vocab = { 'char2idx' : char2idx, 'idx2char' : idx2char }
    return data_list, vocab

def save_model(model, vocab, encoder):
    weights = dict()
    weights['model'] = model.state_dict()
    weights['vocab'] = vocab
    weights['encoder'] = encoder

    return weights

def del_outbound(image, bbox_tensor):
    '''
    DELETE OUTBOUND of image
    bbox format : lx,ly,w,h
    '''
    image_tensor = TF.to_tensor(image) # CxHxW
    alpha = image_tensor[-1] # HxW
    indices = (alpha == 1).nonzero()
    st_height, st_width = indices[0][0],indices[0][1]
    end_height, end_width = indices[-1][0], indices[-1][1]

    del_img = image_tensor[:3,st_height:end_height,st_width:end_width] # RGB(3)
    del_img = TF.to_pil_image(del_img)

    if bbox_tensor is not None:
        bbox_tensor = bbox_tensor - torch.as_tensor([st_width, st_height, 0, 0])
        return del_img, bbox_tensor
    else:
        return del_img, None

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(list(batch[0]))
    return tuple(batch)

def _nms_bbox(ploc, plabel, nms_score=0.1, iou_threshold=0.1): 
    # Split class 0, not 0, if class is 0, bring second class
    scores, classes = torch.topk(F.softmax(plabel,dim=0),2,dim=0)
    non_background_mask = classes[0] != 0
    scores_in = torch.hstack((scores[0][non_background_mask], scores[1][~non_background_mask]))
    classes_in = torch.hstack((classes[0][non_background_mask], classes[1][~non_background_mask]))
    ploc_in = torch.vstack((ploc[non_background_mask],ploc[~non_background_mask]))

    scores_mask = scores_in > nms_score
    ploc_in, scores_in, classes_in = ploc_in[scores_mask], scores_in[scores_mask], classes_in[scores_mask]

    # categorical nms
    _nms_batched_index = torchvision.ops.batched_nms(ploc_in, scores_in, classes_in, iou_threshold)
    _nms_batched_bboxes = ploc_in[_nms_batched_index]
    _nms_batched_labels = classes_in[_nms_batched_index]
    _nms_batched_scores = scores_in[_nms_batched_index]

    # nms
    _nms_index = torchvision.ops.nms(_nms_batched_bboxes, _nms_batched_scores, iou_threshold)
    _nms_bboxes = _nms_batched_bboxes[_nms_index]
    _nms_label = _nms_batched_labels[_nms_index]
    _nms_scores = _nms_batched_scores[_nms_index]

    return _nms_bboxes, _nms_label, _nms_scores

def _nms_eval_iou(ploc, plabel, gloc, nms_score=0.1, iou_threshold=0.1):
    if gloc.shape[0] == 4:
        gloc = gloc.transpose(1,0).contiguous()
    gloc = box_cxcywh_to_ltrb(gloc)

    # nms 
    _nms_bboxes, _, _= _nms_bbox(ploc, plabel, nms_score, iou_threshold)
    if len(_nms_bboxes) == 0: # no_object 
        return 0

    ious = calc_iou_tensor(_nms_bboxes, gloc)
    best_iou, best_idx = ious.max(dim=1)
    mean_iou = best_iou.mean()

    return mean_iou

def _nms_match_ap(_nms_bboxes, _nms_labels, _nms_scores, gloc, glabel, criterion=0.5):
    gloc = box_cxcywh_to_ltrb(gloc)
    # calc iou with ground-truth
    if len(_nms_bboxes) == 0: # no_object / set label unknown conf 0
        return torch.ones(glabel.size()),torch.zeros(glabel.size()), glabel
    ious = calc_iou_tensor(_nms_bboxes, gloc)
    best_iou, best_idx = ious.max(dim=1)
    criterion_mask = best_iou >= criterion
    confidence_scores = _nms_scores[criterion_mask] * best_iou[criterion_mask]
    confidence_labels = _nms_labels[criterion_mask]
    match_labels = glabel[best_idx[criterion_mask]]

    return confidence_labels, confidence_scores, match_labels

def calc_ap(pred_labels, pred_confs, gt_labels, total_labels):
    pred_labels, pred_confs, gt_labels = torch.as_tensor(pred_labels), torch.as_tensor(pred_confs), torch.as_tensor(gt_labels)
    recall_thresholds = torch.linspace(0.1,1.0,11)
    average_precisions = []

    for label_idx in range(1,len(total_labels)): # 0: background
        precisions, recalls = [], []
        mask = gt_labels == label_idx
        if total_labels[label_idx].item() == 0:
            continue
        conf_indices = torch.argsort(pred_confs[mask],descending=True)
        preds = pred_labels[mask]
        labels = gt_labels[mask]
        for i in range(len(conf_indices)):
            TP = torch.sum(preds[conf_indices[:i+1]] == labels[conf_indices[:i+1]])
            FP = torch.sum(preds[conf_indices[:i+1]] != labels[conf_indices[:i+1]])
            precision = TP / (TP+FP+1e-6)
            recall = TP / (total_labels[label_idx] + 1e-6)
            precisions.append(precision)
            recalls.append(recall)
        precisions, recalls = torch.as_tensor(precisions), torch.as_tensor(recalls) 
        average_precision = 0.0
        for t in recall_thresholds:
            p = torch.max(precisions[recalls >= t]) if torch.sum(recalls >= t) != 0 else 0
            average_precision = average_precision + p / len(recall_thresholds)
        average_precisions.append(average_precision)
    return sum(average_precisions) / len(average_precisions)


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.dbxoes_default = dboxes
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, device, criteria = 0.5):
        #bboxes_cxcywh = bboxes_in.clone().detach()
        bboxes_in = box_cxcywh_to_ltrb(bboxes_in)
        ious = calc_iou_tensor(bboxes_in.to(device), self.dboxes.to(device)).cpu()
        if ious.shape[0] == 0:
            return None, None
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to cxcywh format
        cx, cy, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]
        bboxes_out[:, 0] = cx
        bboxes_out[:, 1] = cy
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in, device):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.to(device)
            self.dboxes_xywh = self.dboxes_xywh.to(device)

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        self.dboxes = self.dboxes.cpu()
        self.dboxes_xywh = self.dboxes_xywh.cpu()

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, device, criteria = 0.45, max_output=500):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in, device)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        return output

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []
        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0: continue
            # print(i)

            score = score.squeeze(1)
            mask = score > 0.05
            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
            #maxdata, maxloc = scores_in.sort()

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i]*len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
               torch.tensor(labels_out, dtype=torch.long), \
               torch.cat(scores_out, dim=0)


        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes

def dboxes512():
    figsize = 512
    feat_size = [8, 16, 32, 64]
    steps = [64, 32, 16, 8]
    scales = [153, 99, 45, 15, 5, 261, 315]
    aspect_ratios = [[2], [2], [2], [2], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def draw_result(outPath, imgPath, loc, label, prob, idx2char):
    outPath = Path(outPath)
    outPath.mkdir(parents=True, exist_ok=True)
    img = Image.open(imgPath)
    img_name = Path(imgPath).name
    if img.mode == 'RGBA':
        img, _ = del_outbound(img, None)
    elif img.mode == 'L':
        img = img.convert('RGB')
    width, height = img.size
    fontPath = 'util/YetHangul.ttf'
    draw = ImageDraw.Draw(img)
    for loc_, label_, prob_ in zip(loc, label, prob):
        l,t,r,b = map(int,loc_)
        if label_ == 0:
            continue
        fontSize = (r-l) // 3
        font = ImageFont.truetype(fontPath, fontSize)
        label_ = idx2char[label_]
        draw.rectangle([(l,t),(r,b)], outline=(255,0,0),width=3)
        anno = f'{label_},{prob_:.2f}'
        draw.text((l+5,t-5), anno, font=font, fill=(0,0,0))
    img.save(outPath / img_name)
    print(f'{img_name} Done!')
