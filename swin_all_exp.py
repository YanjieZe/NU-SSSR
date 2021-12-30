import torch
import utils
import torch.nn as nn
from arguments import parse_args
import torch.optim as optim
import torch.utils.data as data
import tqdm
import wandb
import torch.nn.functional as F
import copy

from models.vit import ViT
from models.mae import MAE

def train_and_test_swinir(args):
    print(args)
    run = wandb.init(project='cg-swinir-experiments', reinit=True)
    run.name = f"{args.alg}-{args.sample_method}-{args.point_num}-{args.method}"
    wandb.config = {
        "args" : args,
        "seed" : seed
    }
    utils.set_seed_everywhere(seed)
    loss_function = nn.MSELoss()
    model = utils.make_model(args).to(device)
    model.to(device)
    # model.load_state_dict(torch.load('/home/purewhite/workspace/cg-proj/NUG-DLSS/logs/MAE_pretrained.pth'))
    
    # optimizer = optim.RAdam(params=model.parameters(),lr=args.lr)
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr)
    train_dataset = utils.TrainDataset(args)
    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=utils.collect_function)
    
    optimizer.zero_grad()
    wandb.watch(model)
    model.train()
    for e in range(args.epoch):
        tot_mse = 0
        tot_psnr = 0
        tot_ssim = 0

        tot_lr_mse = 0
        tot_lr_psnr = 0
        tot_lr_ssim = 0
        
        model.train()
        loop = tqdm.tqdm(train_loader)
        for idx, img_pair in enumerate(loop):
            
            img_hr = img_pair['hr'].to(device)
            img_lr = img_pair['lr'].to(device)
            img_pred = model(img_lr)
            
            mse = loss_function(img_hr, img_pred)
            psnr = utils.psnr(img_hr, img_pred)
            ssim = utils.ssim(img_hr, img_pred)
            loss = -psnr/300 - ssim + mse
            
            tot_mse += mse
            tot_psnr += psnr
            tot_ssim += ssim
            
            lr_psnr = utils.psnr(img_hr, img_lr)
            lr_ssim = utils.ssim(img_hr, img_lr)
            lr_mse = loss_function(img_hr, img_pred)
                
            tot_lr_mse += lr_mse
            tot_lr_psnr += lr_psnr
            tot_lr_ssim += lr_ssim
            
            loss = -psnr/300 - ssim + mse
            wandb.log({"psnr improvement": (psnr-lr_psnr)*100/lr_psnr})
            wandb.log({"ssim improvement": (ssim-lr_ssim)*100/lr_ssim})
            wandb.log({"mse improvement": (lr_mse-mse)*100/lr_mse})
            wandb.log({"psnr": psnr})
            wandb.log({"ssim": ssim})
            wandb.log({"mse": mse})
            wandb.log({"loss": loss})
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"epoch: {e} | iter: {idx}/{len(train_dataset)} | loss: {loss.item()}")
        
        e += 1
        utils.save_model(model, e, args)
    
    # Reset seeds for reproducible test
    utils.set_seed_everywhere(seed)
    test_dataset = utils.TestDataset(args)
    test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True,
                                    collate_fn=utils.collect_function)
    run.finish()
    model.eval()
    tot_mse = 0
    tot_psnr = 0
    tot_ssim = 0

    tot_lr_mse = 0
    tot_lr_psnr = 0
    tot_lr_ssim = 0

    loop = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for idx, img_pair in enumerate(loop):
            
            img_hr = img_pair['hr'].to(device)
            img_lr = img_pair['lr'].to(device)
            
            img_pred = model(img_lr)

            
            mse = loss_function(img_hr, img_pred)
            psnr = utils.psnr(img_hr, img_pred)
            ssim = utils.ssim(img_hr, img_pred)
            loss = -psnr/300 - ssim + mse
            
            tot_mse += mse
            tot_psnr += psnr
            tot_ssim += ssim
            
            lr_psnr = utils.psnr(img_hr, img_lr)
            lr_ssim = utils.ssim(img_hr, img_lr)
            lr_mse = loss_function(img_hr, img_pred)
                
            tot_lr_mse += lr_mse
            tot_lr_psnr += lr_psnr
            tot_lr_ssim += lr_ssim
            
            loop.set_description(f"iter: {idx}/{len(test_loader)} | loss: {loss.item()} | psnr improvement: {(tot_psnr-tot_lr_psnr)*100/tot_lr_psnr} | ssim improvement: {(tot_ssim-tot_lr_ssim)*100/tot_lr_ssim} | mse_decrease: {(tot_lr_mse-tot_mse)*100/tot_lr_mse}")
    print(f"Avg.LR_PSNR: {tot_lr_psnr / len(test_loader)} | Avg.LR_SSIM: {tot_lr_ssim / len(test_loader)}")
    print(f"Avg.PSNR: {tot_psnr / len(test_loader)} | Avg.SSIM: {tot_ssim / len(test_loader)}")
    

# args = [  "--lr", "1e-3"
#         , "--epoch", "50"
#         , '--data_root', 'data/celeba'
#         , '--batch_size', '1'
#         , '--img_width', '256'
#         , '--img_height', '256'
#         , '--img_size', '256'
# ]

if __name__ == "__main__":
    args = parse_args()
    args.lr = 1e-3
    args.epoch = 50
    args.data_root='data/celeba'
    args.batch_size=4
    args.img_height = 64
    args.img_width = 64
    args.img_size = 64
    args.scale = 1
   
    seed = 31415926
    # img_size, patch_size = (256, 256), (16, 16)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    for alg in ['SwinIR']:
        for sample_method, point_num, method in [
            # ('random', '1000', 'center'),
            # ('fourier', '1000', 'center'),
            # ('sobel', '1000', 'center'),
            # ('random', '1000', 'vertex'),
            ('random', '1000', 'barycentric'),
            
            ('random', '10000', 'center'),
            ('fourier', '10000', 'center'),
            ('sobel', '10000', 'center'),
            ('random', '10000', 'vertex'),
            ('random', '10000', 'barycentric'),
            ]:
            nargs = copy.deepcopy(args)
            nargs.alg = alg
            nargs.point_num = int(point_num)
            nargs.method = method
            nargs.sample_method = sample_method

           
            train_and_test_swinir(nargs)
    