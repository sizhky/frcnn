from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms

device = 'cuda'
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def get_items(root, phase):
    # assert phase in {'train', 'val'}
    with open(root/f'ImageSets/Main/{phase}.txt', 'r') as f:
        _items = f.read().split('\n')[:-1]
        if len(_items[0].split()) == 2:
        	_items = [i.split() for i in _items]
        	_items = [i for i,j in _items if j=='1']
    items = []
    for item in _items:
        im, annot = root/f'JPEGImages/{item}.jpg', root/f'Annotations/{item}.xml'
        items.append((im, annot))
    return items

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox

aug_trn = iaa.Sequential([
    iaa.geometric.Affine(rotate=(-20,20),
                         translate_px=(-20,20),
                         shear=(-5,5),
                         mode='edge'),
    iaa.Fliplr(0.5),
    iaa.size.CropToSquare(),
    iaa.size.Resize(300)
])

aug_val = aug_trn = iaa.Sequential([
    iaa.size.Resize(300)
])

def augment_image_with_bbs(image, bbs, aug_func):
    bbs = [BoundingBox(*bb) for bb in bbs]
    im, bbs = aug_func(image=image, bounding_boxes=bbs)
    h, w = im.shape[:2]
    bbs = [(bb.x1,bb.y1,bb.x2,bb.y2) for bb in bbs]
    bbs = [[int(round(i)) for i in bb] for bb in bbs]
    bbs = [(np.clip(x,0,w), np.clip(y,0,h), np.clip(X,0,w), np.clip(Y,0,h)) for x,y,X,Y in bbs]
    return im, bbs

class VOCDataset(Dataset):
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    def __init__(self, items, tfms=aug_val, objects=None):
        super(VOCDataset).__init__()
        self.items = items
        self.tfms = tfms
        self.objects = objects
        self._objects = [voc_labels.index(o)+1 for o in objects] if objects else None
    def __len__(self): return len(self.items)
    def __getitem__(self, ix):
        image_path, annot_path = self.items[ix]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        annot = parse_annotation(annot_path)
        bbs = annot['boxes']
        difficulties = annot['difficulties']
        clss = [l for l in annot['labels']]
        if self.objects:
        	keep_ixs = [ix for ix,cls in enumerate(clss) if cls in self._objects]
        	bbs = [bbs[ix] for ix in keep_ixs]
        	clss = [clss[ix] for ix in keep_ixs]
        if self.tfms is not None:
            image, bbs = augment_image_with_bbs(image, bbs, self.tfms)
        return Image.fromarray(image), bbs, clss, difficulties
    def sample(self): return choose(self)

if __name__ == '__main__':
    from pathlib import Path
    _2007_root = Path("/home/yyr/data/VOCdevkit/VOC2007")
    _2012_root = Path("/home/yyr/data/VOCdevkit/VOC2012")
    train_items = get_items(_2007_root, 'train') + get_items(_2012_root, 'train')
    val_items   = get_items(_2007_root, 'val') + get_items(_2012_root, 'val')
    logger.info(f'\n{len(train_items)} training images\n{len(val_items)} validation images')
    x = VOCDataset(train_items, tfms=aug_trn)
    np.random.seed(12)
    im, bbs, clss = x.sample()
    show(im, bbs=bbs, texts=map(lambda label:voc_labels[label-1], clss), sz=5, text_sz=10)
