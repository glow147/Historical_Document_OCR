import torch
import argparse
import time, datetime
from pathlib import Path
import torch.optim as optim
from torch.autograd import Variable
from models.model import SSD, Loss, ResNet
from util.datasets import dataset_generator
from util.box_ops import box_cxcywh_to_ltrb
from collections import defaultdict
from util.utils import *

def main(args):
    dataloader = dict()
    vocab = None
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.weight_fn:
        weights = torch.load(args.weight_fn, map_location=device)
        encoder = weights['encoder']
        backbone = weights['backbone']
        print(backbone)
        vocab = weights['vocab']
        dboxes = encoder.dbxoes_default
        model = SSD(backbone=ResNet(backbone), n_classes=len(vocab['char2idx']))
        model.load_state_dict(weights['model'])
        criterion = Loss(dboxes, args.loss_method)
        print('Load weights file Done')
        if args.type == 'chinese':
            train_list, _ = load_chinese(args.data_dir + '/train', vocab)
            valid_list, _ = load_chinese(args.data_dir + '/valid', vocab)
            test_list, _ = load_chinese(args.data_dir + '/test', vocab)
        else:
            data_list, vocab = load_yethangul(args.data_dir)
            split_point = int(len(data_list)*0.9)
            train_list, valid_list = data_list[:split_point], data_list[split_point:]
            test_list = valid_list
    else:
        dboxes = dboxes512()
        encoder = Encoder(dboxes)
        criterion = Loss(dboxes, args.loss_method)
        if args.type == 'chinese':
            train_list, vocab = load_chinese(args.data_dir + '/train')
            valid_list, _ = load_chinese(args.data_dir + '/valid', vocab)
            test_list, _ = load_chinese(args.data_dir + '/test', vocab)
        else:
            data_list, vocab = load_yethangul(args.data_dir)
            split_point = int(len(data_list)*0.9)
            train_list, valid_list = data_list[:split_point], data_list[split_point:]
            test_list = valid_list

        model = SSD(backbone=ResNet(args.backbone), n_classes=len(vocab['char2idx']))

    train_set = dataset_generator(args, train_list, 'train')
    valid_set = dataset_generator(args, valid_list, 'valid')
    test_set = dataset_generator(args, test_list, 'test')
    dataloader['train'] = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    dataloader['valid'] = torch.utils.data.DataLoader(valid_set, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    dataloader['test'] = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

    model, criterion = model.to(device), criterion.to(device)
    if not args.test:
        train(args, model, criterion, dataloader, encoder, vocab, device)
    else: 
        test(args, model, dataloader['test'], encoder, vocab, device)

def train(args, model, criterion, dataloader, encoder, vocab, device):
    scaler = torch.cuda.amp.GradScaler()
    weight_path = Path(args.weights)
    weight_path.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(),lr = args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 0.95)
    print("Start Training")
    prev_time = time.time()
    best_loss = 9999 
    best_iou = -1
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        start = time.time()
        for batch_idx,(imgs, targets) in enumerate(dataloader['train']):
            batch_mask = []
            gloc_list, glabel_list = [],[]
            targets = [{k: v for k, v in t.items()} for t in targets]
            for i,target in enumerate(targets):
                gloc, glabel = encoder.encode(target['boxes'], target['labels'], device)
                if gloc is None:
                    batch_mask.append(False)
                    continue
                gloc_list.append(gloc)
                glabel_list.append(glabel)
                batch_mask.append(True)

            with torch.cuda.amp.autocast():
                imgs = imgs[batch_mask].to(device)
                gloc, glabel  = torch.stack(gloc_list), torch.stack(glabel_list)
                gloc = gloc.transpose(1,2).contiguous()
                gloc = Variable(gloc, requires_grad=False).to(device)
                glabel = Variable(glabel, requires_grad=False).to(device)

                ploc, plabel = model(imgs)
                loss, box_loss, class_loss = criterion(ploc, plabel, gloc, glabel)
                avg_loss += loss.item()

            #calc ETA
            batches_len = len(dataloader['train'])
            batches_done = epoch * batches_len + batch_idx
            batches_left = args.epochs * batches_len - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            #train model
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f'\r epoch : {epoch}/{args.epochs} batch_idx : {batch_idx} / {batches_len} loss : {loss.item():.2f} \
                    box_loss : {box_loss:.2f}, class_loss : {class_loss:.2f} ETA : {time_left}',end='')
        print(f'avg_loss : {avg_loss/batches_len:.2f}')
        val_loss, mean_iou = evaluate(model, criterion, dataloader['valid'], encoder, device)
        model_weights = save_model(model, vocab, encoder, args.backbone)
        if val_loss < best_loss:
            print(f'Loss improve from {best_loss:.2f} to {val_loss:.2f} Iou change from {best_iou:.2f} to {mean_iou:.2f}')
            best_loss = val_loss
            best_iou = mean_iou
            torch.save(model_weights, str(weight_path / f'best.pt'))
        torch.save(model_weights, str(weight_path / f'{epoch}.pt'))

@torch.no_grad()
def evaluate(model, criterion, dataloader, encoder, device):
    model.eval()
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        targets = [{k: v for k, v in t.items()} for t in targets]
        gloc_list, glabel_list = [], []
        for i,target in enumerate(targets):
            gloc, glabel = encoder.encode(target['boxes'], target['labels'], device)
            gloc_list.append(gloc.transpose(1,0).contiguous())
            glabel_list.append(glabel)

        gloc, glabel = torch.stack(gloc_list).to(device), torch.stack(glabel_list).to(device)
        gloc = Variable(gloc, requires_grad=False)
        glabel = Variable(glabel, requires_grad=False)

        ploc, plabel = model(imgs)
        loss, box_loss, class_loss = criterion(ploc, plabel, gloc, glabel)
        ious = 0.0
        ploc_rescaled, _ = encoder.scale_back_batch(ploc, plabel, device)
        for idx in range(ploc_rescaled.shape[0]):
            mask = glabel[idx] > 0
            iou = nms_eval_iou(ploc_rescaled[idx][mask], plabel[idx][:,mask], gloc[idx][:,mask])
            if iou is None: # Detect No Object
                continue 
            ious +=  iou
        mean_iou = ious / ploc.shape[0]
        break

    return loss.item(), mean_iou

@torch.no_grad()
def test(args, model, dataloader, encoder, vocab, device):
    model.eval()
    '''
    AP_result = {0.25:defaultdict(lambda:[]), 
                 0.5 :defaultdict(lambda:[]), 
                 0.75:defaultdict(lambda:[])}
    '''
    AP_result = {0.8 : defaultdict(lambda:[])}
    #AP_result['labels'] = torch.zeros(len(vocab['idx2char']))
    result_dict = defaultdict(lambda: defaultdict(lambda:[])) 
    start = time.time()
    for batch_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = [{k: v for k, v in t.items()} for t in targets]
        ploc, plabel = model(imgs)
        ploc_rescaled, _ = encoder.scale_back_batch(ploc, plabel, device)
        for idx in range(ploc_rescaled.shape[0]):
            gloc_i = targets[idx]['boxes'].to(device)
            glabel_i = targets[idx]['labels'].to(device)
            style = targets[idx]['style']
            name = targets[idx]['imgPath'].stem
#            AP_result['labels'][glabel_i] += 1
            _nms_bboxes, _nms_labels, _nms_scores = nms_bbox(ploc_rescaled[idx], plabel[idx], nms_score=0.1, iou_threshold=0.1)
            for criterion in AP_result:
                if criterion == 'labels':
                    continue
                confidence_bbox, confidence_labels, confidence_scores, gt_labels = nms_match_ap(_nms_bboxes, _nms_labels, _nms_scores, gloc_i, glabel_i, criterion)
                if len(confidence_labels) == 0:
                    TP = torch.tensor(0.0)
                    FP = torch.tensor(len(glabel_i),dtype=float)
                else:
                    TP = torch.sum(confidence_labels == gt_labels)
                    FP = torch.sum(confidence_labels != gt_labels)
                precision = TP / (TP+FP+1e-6)
                recall = TP / (len(glabel_i)+1e-6)
                f1 = 2*(precision*recall) / (precision+recall+1e-6)
                result_dict[style]['precision'].append(precision.item())
                result_dict[style]['recall'].append(recall.item())
                result_dict[style]['f1_score'].append(f1.item())
                if args.output_dir:
                    loc, label, prob = confidence_bbox.cpu(), confidence_labels.tolist(), confidence_scores.tolist()
                    htot, wtot = targets[idx]['orig_size'][0], targets[idx]['orig_size'][1]
                    loc[:,0::2] = loc[:,0::2] * wtot.item()
                    loc[:,1::2] = loc[:,1::2] * htot.item()
                    draw_result(args.output_dir, targets[idx]['imgPath'], loc, label, prob, vocab['idx2char'])
                '''
                AP_result[criterion]['preds'].extend(confidence_labels.tolist())
                AP_result[criterion]['scores'].extend(confidence_scores.tolist())
                AP_result[criterion]['gt_labels'].extend(gt_labels.tolist())
                '''
                print(f'name : {name}, style : {style}, precision : {precision:.2f}, recall : {recall:.2f}, f1_score : {f1:.2f}')
        print(f'{batch_idx} / {len(dataloader)}')
    print('-'*20,'total','-'*20)
    total_pre = [] 
    total_re = [] 
    total_f1 = [] 
    for style in result_dict:
        print(style)
        precision = sum(result_dict[style]['precision']) / len(result_dict[style]['precision'])
        recall = sum(result_dict[style]['recall']) / len(result_dict[style]['recall'])
        f1_score = sum(result_dict[style]['f1_score']) / len(result_dict[style]['f1_score'])
        print('precision : ', precision)
        print('recall : ', recall)
        print('f1_score : ', f1_score)
        total_pre += result_dict[style]['precision']
        total_re += result_dict[style]['recall']
        total_f1 += result_dict[style]['f1_score']
    print('-'*45)
    print('total precision : ', sum(total_pre) / len(total_pre))
    print('total recall : ', sum(total_re) / len(total_re))
    print('total f1_score : ', sum(total_f1) / len(total_f1))
    print('total time : ', time.time() - start)
    '''
    for criterion in AP_result:
        if criterion == 'labels':
            continue
        TP = torch.sum(torch.tensor(AP_result[criterion]['preds']) == torch.tensor(AP_result[criterion]['gt_labels']))
        FP = torch.sum(torch.tensor(AP_result[criterion]['preds']) != torch.tensor(AP_result[criterion]['gt_labels']))
        precision = TP / (TP+FP+1e-6)
        recall = TP / torch.sum(AP_result['labels'])
        mAP = 2*(precision*recall) / (precision+recall)
        #mAP = calc_ap(AP_result[criterion]['preds'], AP_result[criterion]['scores'], AP_result[criterion]['gt_labels'], AP_result['labels']) 
        print(f'{criterion}, {mAP}')
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Historical Document OCR")
    
    parser.add_argument("--data_dir", required=True, default="/data/train",
                        help="Path to the training folder")
    parser.add_argument("--weights", default="weights",
                        help="weight files path to save")
    parser.add_argument("--type", default="chinese", type=str, choices=('chinese','yethangul'))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default='0',
                        help="set gpu number")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="set batch size")
    parser.add_argument("--epochs", default=25,
                        help="set epoch")
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="set learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float)

    # Backbone
    parser.add_argument("--backbone", default='resnet50', type=str)

    # Loss
    parser.add_argument("--loss_method", default='L2', type=str, choices=('L2','SL1'))

    # Load
    parser.add_argument("--weight_fn", default=None, type=str)
    parser.add_argument("--test", action='store_true')

    main(parser.parse_args())
