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
import torch.nn.functional as F
from torch.autograd import Variable
import numba as nb
from torchvision.transforms import ToTensor, ToPILImage

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
    
    if model_name=='SRCNN':
        from models import SRCNN
        return SRCNN(args)
    elif model_name=='SRCNN2':
        from models import SRCNN2
        return SRCNN2(args)
    elif model_name == 'CNF':
        from models import condNF
        return condNF.FlowModel(
            (3, args.img_height, args.img_width),
            args.cnf_filter_size,
            args.cnf_L,
            args.cnf_K,
            args.batch_size,
            1,
            args.cnf_nb,
            args.cnf_condch,
            args.cnf_nbits,
            args.cnf_noscale,
            args.cnf_noscaletest,
        )
    elif model_name == "VDSR":
        from models import vdsr
        return vdsr.Net()
    else:
        raise NotImplemented('Model %s is not implemented.'%model_name)

def save_model(model, epoch, args):
    save_path = os.path.join(args.log_dir, args.alg+'_'+args.description+'_'+str(epoch)+'.pth')
    torch.save(model.state_dict(), save_path)

def load_model(model, epoch, args):
    save_path = os.path.join(args.log_dir, args.alg+'_'+str(epoch)+'.pth')
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(save_path))
    
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

def show_gt_and_pred(img_hr, img_lr, pred_hr, figsize=(30, 30)):
    plt.figure(1, figsize=figsize)
    plt.subplot(1, 3, 1) #图一包含1行2列子图，当前画在第一行第一列图上
    plt.imshow(img_hr)
    plt.title('ground truth hr')

    plt.figure(1, figsize=figsize)
    plt.subplot(1, 3, 2) #图一包含1行2列子图，当前画在第一行第一列图上
    plt.imshow(img_lr)
    plt.title('ground truth lr')

    plt.figure(1, figsize=figsize)
    plt.subplot(1, 3, 3)#当前画在第一行第2列图上
    plt.imshow(pred_hr)
    plt.title('predict hr')

    plt.show()

def visualize_all(img_hr, img_lr, img_pred, figsize=(25, 25)):
    error_color = np.expand_dims(np.array((1.0, 0, 0)), axis=0)
    
    error = img_pred - img_hr
    squared_error = error * error
    mse_vis = np.sqrt(squared_error).sum(axis=2, keepdims=True)
    mse_vis = mse_vis @ error_color
    
    plt.figure(1, figsize=figsize)
    plt.subplot(1, 4, 1)
    plt.imshow(img_lr)
    plt.title('Delaunay Reconstructed Image')

    plt.figure(1, figsize=figsize)
    plt.subplot(1, 4, 2)
    plt.imshow(img_pred)
    plt.title('MAE Reconstructed Image')

    plt.figure(1, figsize=figsize)
    plt.subplot(1, 4, 3)
    plt.imshow(img_hr)
    plt.title('Groundtruth Image')
    
    plt.figure(1, figsize=figsize)
    plt.subplot(1, 4, 4)
    plt.imshow(mse_vis)
    plt.title('Error Visualization')

    plt.show()

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
                # [transforms.CenterCrop((self.args.img_width,self.args.img_height))]
                )
        elif target=='hr':
            trans = transforms.Compose( 
                [transforms.Resize((self.args.img_width,self.args.img_height))]
                # [transforms.CenterCrop((self.args.img_width,self.args.img_height))]
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
                # [transforms.Resize((self.args.img_width,self.args.img_height))]
                [transforms.CenterCrop((self.args.img_width,self.args.img_height))]
                )
        elif target=='hr':
            trans = transforms.Compose( 
                # [transforms.Resize((self.args.img_width,self.args.img_height))]
                [transforms.CenterCrop((self.args.img_width,self.args.img_height))]
                )
        else:
            raise Exception('Transform not supported.')
        return trans

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.img_list[idx])
        # print(img_path)
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

  

def psnr(gt, pred, size_average = True):
    # img1 and img2 have range [0, 255]
    mse = F.mse_loss(gt, pred, size_average=size_average)
    if mse == 0:
        return float('inf')
    if size_average:
        return (20 * torch.log10(255.0 / torch.sqrt(mse))).mean()
    else:
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

# https://github.com/Po-Hsun-Su/pytorch-ssim

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
