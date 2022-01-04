import torch
import argparse
from pathlib import Path
from models.model import SSD
from torchvision.transforms as T
from util.utils import draw_result, del_outbound

tx = T.Compose([
        T.Resize((512,512)),
        T.ToTensor()])

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, args, transforms = None):
        exts = ('*.jpg','*.jpeg','*.png','*.gif')
        self.imgPathList = list(Path(args.data_dir).glob(exts))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.imgPathList)

    def __getitem__(self, idx):
        imgPath = self.imgPathList[idx]
        img = Image.open(imgPath)
        sample = dict()

        if img.mode == 'RGBA':
            img, None = del_outbound(img,None)
        elif img.mode == 'L':
            img = img.convert('RGB')

        sample['origin_img'] = img
        sample['imgPath'] = imgPath

        if self.transforms:
            tensor_img = self.transforms(img)
            sample['tensor_img'] = tensor_img

        return sample

@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    #data load
    imageSet = ImageDataset(args, tx)
    dataloader = torch.utils.data.DataLoader(imageSet, batch_size = args.batch_size)

    #setting load
    weights = torch.load(weight_fn,map_location=device)
    encoder = weights['encoder']
    vocab = weights['vocab']
    model = SSD(n_classes=len(vocab['idx2char'])).to(device)
    model.load_state_dict(weights['model'])

    model.eval()

    #make result
    for sample in dataloader:
        imgs = sample['tensor_img'].to(device)
        ploc, plabel = model(imgs)
        for idx in range(ploc.shape[0]):
            ploc_i = encoder.scale_back_batch(ploc[idx],plabel[idx], device)
            plabel_i = plabel[idx]
            _nms_bboxes, _nms_labels, _nms_scores = _nms_bbox(ploc_i, plabel_i, nms_score=0.1, iou_threshold=0.1)
            if _nms_bboxes.shape[0] == 0:
                print('No obejct')
                continue
            htot, wtot = sample['origin_img'].size
            loc, label, prob = _nms_bboxes.cpu(), _nms_labels.tolist(), _nms_scores.tolist()
            loc[:,0::2] = loc[:,0::2] * wtot
            loc[:,1::2] = loc[:,1::2] * htot
            draw_result(args.output_dir, sample, loc, label, prob, vocab['idx2char'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traditional Hangul OCR with DETR")
    
    parser.add_argument("--data_dir", required=True, default="/data/train",
                        help="Path to the training folder")
    parser.add_argument("--weight_fn", default="weights/best.pt",
                        help="Path to model weight path")
    parser.add_argument("--output_dir", default="images")
    parser.add_argument("--device", default='0',
                        help="set gpu number")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="set batch size")

    main(parser.parse_args())
