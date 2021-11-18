import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import os
from sampling import DelaunayTriangulationBlur
from PIL import Image
import random
import torch
import torchvision.transforms as transforms

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def show_img(img):
    cv2.namedWindow("img", cv2.WINDOW_NORMAL) 
    cv2.imshow("img", img)
    cv2.waitKey(0)

def save_img(name, img):
    cv2.imwrite(name, img)


def make_model(args):
    model_name = args.alg
    from models import SRCNN
    if model_name=='SRCNN':
        return SRCNN(args)
    else:
        raise NotImplemented('Model %s is not implemented.'%model_name)

def save_model(model, epoch, args):
    save_path = os.path.join(args.log_dir, args.alg+'_'+str(epoch)+'.pth')
    torch.save(model.state_dict(), save_path)

def collect_function(batch):
    img_pair = dict()
    for idx, pair in enumerate(batch):
        if idx==0:
            img_pair['hr'] = pair['hr'].unsqueeze(0)
            img_pair['lr'] = pair['lr'].unsqueeze(0)
            # print(img_pair['lr'].shape) # 1, 256, 256, 3

        else:
            img_pair['hr'] = torch.vstack([img_pair['hr'], pair['hr'].unsqueeze(0)])
            img_pair['lr'] = torch.vstack([img_pair['lr'],pair['lr'].unsqueeze(0)])
    img_pair['hr'] = img_pair['hr'].permute(0, 3,1,2)
    img_pair['lr'] = img_pair['lr'].permute(0, 3,1,2)

    return img_pair

class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_path = os.path.join(args.data_root, 'train')
        self.img_list = os.listdir(self.root_path)
        try:
            self.img_list.remove('.DS_Store')
        except:
            pass
        self.method = args.method
        self.transform_on_hr = self.get_transform('hr')
        self.transform_on_lr = self.get_transform('lr')

    
    def get_transform(self, target):
        if target=='lr':
            trans = transforms.Compose( 
                [transforms.Resize((self.args.img_width,self.args.img_height))]
                )
        elif target=='hr':
            trans = transforms.Compose( 
                [transforms.Resize((self.args.img_width,self.args.img_height))]
                )
        else:
            raise Exception('Transform not supported.')
        return trans

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.img_list[idx])
        img_pair = dict()

        img_hr = Image.open(img_path)
        img_hr= np.array(self.transform_on_hr(img_hr))
        img_pair['hr'] = torch.tensor(img_hr, dtype=torch.float32)
        img_lr = DelaunayTriangulationBlur(img_hr, \
            self.args.point_num, self.args.method)
        img_lr = self.transform_on_lr(Image.fromarray(img_lr))
        img_pair['lr'] = torch.tensor(np.array(img_lr), dtype=torch.float32)
        
        img_pair['hr'] /= 255
        img_pair['lr'] /= 255

        return img_pair
    
    def __len__(self):
        return len(self.img_list)


class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_path = os.path.join(args.data_root, 'test')
        self.img_list = os.listdir(os.path.join(args.data_root, 'test'))
        try:
            self.img_list.remove('.DS_Store')
        except:
            pass
        
        self.method = args.method
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.img_list[idx])
        img_pair = dict()

        img_hr = np.array(Image.open(img_path))
        img_pair['hr'] = torch.tensor(img_hr, dtype=torch.float32)
        img_pair['lr'] = DelaunayTriangulationBlur(img_hr, \
            self.args.point_num, self.args.method)
        img_pair['lr'] = torch.tensor(img_pair['lr'], dtype=torch.float32)

        img_pair['hr'] /= 255
        img_pair['lr'] /= 255
        return img_pair
    
    def __len__(self):
        return len(self.img_list)
    
