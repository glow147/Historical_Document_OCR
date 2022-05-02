import torch, random
import numpy as np
import util.transforms as T
from PIL import Image
from util.utils import del_outbound
from util.box_ops import get_rotated_box

output_size = 512 

def dataset_generator(args, data_list, mode):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if mode == 'train':
        tx = T.Compose([
#             T.RandomRotate(),
             T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600)
                    ]),
            ),
            T.Resize([output_size, output_size]),
            normalize
        ])

    else:
        tx = T.Compose([
                T.Resize([output_size, output_size]),
                normalize
                ])
    data_set = Dataset(args, data_list, tx)

    return data_set

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.angles = [0, 90, 180, 270]
        self.type = args.type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        imgPath = self.data_list[idx][0]
        img = Image.open(imgPath)

        boxes = torch.as_tensor([[lx,ly,w,h] for _,lx,ly,w,h in self.data_list[idx][1]], dtype=torch.float32).reshape(-1,4)
        classes = torch.as_tensor([v[0] for v in self.data_list[idx][1]], dtype=torch.int64)

        if img.mode == 'RGBA':
            img, boxes = del_outbound(img, boxes)
        elif img.mode == 'L':
            img = img.convert('RGB')
        pic_width, pic_height = img.size

        boxes[:, 2:] += boxes[:, :2] # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=pic_width)
        boxes[:, 1::2].clamp_(min=0, max=pic_height)
        keep = (boxes[:,3] > boxes[:,1]) & (boxes[:,2] > boxes[:,0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['orig_size'] = torch.as_tensor([int(pic_height),int(pic_width)])
        target['size'] = torch.as_tensor([int(pic_height),int(pic_width)])
        target['imgPath'] = imgPath
        if self.type == 'chinese':
            target['style'] = imgPath.parent.stem
        else: # yethangul
            target['style'] = imgPath.stem.split('_')[0]

        if self.transform:
            img, target  = self.transform(img, target)

        return img, target
