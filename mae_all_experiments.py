import torch
import utils
from arguments import parse_args
import torch.optim as optim
import torch.utils.data as data
import tqdm
import wandb
import torch.nn.functional as F

from models.vit import ViT
from models.mae import MAE

def train_and_test_mae(args):
    args = parse_args(args)
    print(args)
    run = wandb.init(project='cg-mae-experiments', entity="purewhite2019", reinit=True)
    run.name = args.description
    wandb.config = {
        "args" : args,
        "seed" : seed
    }
    utils.set_seed_everywhere(seed)
    
    encoder = ViT(img_size, patch_size, depth=6, dim=512, mlp_dim=768, num_heads=8, channels=4) # Simple
    model = MAE(encoder, decoder_depth=6, decoder_dim=512, mask_ratio=0.75)
    model.to(device)
    model.load_state_dict(torch.load('/home/purewhite/workspace/cg-proj/NUG-DLSS/logs/MAE_pretrained.pth'))
    
    optimizer = optim.RAdam(params=model.parameters(),lr=args.lr)
    train_dataset = utils.TrainDataset_4Channel(args)
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
            
            x = img_pair['lr'].to(device)
            output = model.forward_nomask(x)
            
            img_pred = output.permute(0, 2, 3, 1)[:, :, :, :-1].permute(0, 3, 1, 2).to(device)
            img_lr = img_pair['lr'].permute(0, 2, 3, 1)[:, :, :, :-1].permute(0, 3, 1, 2).to(device)
            img_hr = img_pair['hr'].to(device)
            
            mse = F.mse_loss(img_hr.reshape((img_hr.shape[0], -1)), img_pred.reshape((img_hr.shape[0], -1)))
            psnr = utils.psnr(img_hr, img_pred)
            ssim = utils.ssim(img_hr, img_pred)
            loss = -psnr/300 - ssim + mse
            
            tot_mse += mse
            tot_psnr += psnr
            tot_ssim += ssim
            
            lr_psnr = utils.psnr(img_hr, img_lr)
            lr_ssim = utils.ssim(img_hr, img_lr)
            lr_mse = F.mse_loss(img_hr.reshape((img_hr.shape[0], -1)), img_lr.reshape((img_hr.shape[0], -1)))
                
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
    test_dataset = utils.TestDataset_4Channel(args)
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
            
            x = img_pair['lr'].to(device)
            output = model.forward_nomask(x)
            
            img_pred = output.permute(0, 2, 3, 1)[:, :, :, :-1].permute(0, 3, 1, 2).to(device)
            img_lr = img_pair['lr'].permute(0, 2, 3, 1)[:, :, :, :-1].permute(0, 3, 1, 2).to(device)
            img_hr = img_pair['hr'].to(device)
            
            mse = F.mse_loss(img_hr.reshape((img_hr.shape[0], -1)), img_pred.reshape((img_hr.shape[0], -1)))
            psnr = utils.psnr(img_hr, img_pred)
            ssim = utils.ssim(img_hr, img_pred)
            loss = -psnr/300 - ssim + mse
            
            tot_mse += mse
            tot_psnr += psnr
            tot_ssim += ssim
            
            lr_psnr = utils.psnr(img_hr, img_lr)
            lr_ssim = utils.ssim(img_hr, img_lr)
            lr_mse = F.mse_loss(img_hr.reshape((img_hr.shape[0], -1)), img_lr.reshape((img_hr.shape[0], -1)))
                
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
    args = [  "--lr", "1e-3"
            , "--epoch", "50"
            , '--data_root', 'data/celeba'
            , '--batch_size', '1'
            , '--img_width', '256'
            , '--img_height', '256'
            , '--img_size', '256'
    ]
    seed = 31415926
    img_size, patch_size = (256, 256), (16, 16)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    for alg in ['MAE']:
        for sample_method, point_num, method in [
            ('random', '1000', 'center'),
            ('fourier', '1000', 'center'),
            ('sobel', '1000', 'center'),
            ('random', '1000', 'vertex'),
            ('random', '1000', 'barycentric'),
            
            ('random', '10000', 'center'),
            ('fourier', '10000', 'center'),
            ('sobel', '10000', 'center'),
            ('random', '10000', 'vertex'),
            ('random', '10000', 'barycentric'),
            ]:
            nargs = args.copy()
            nargs += ['--alg', alg, '--sample_method', sample_method, '--point_num', point_num, '--method', method,
                          '--description', f"{alg}-{sample_method}-{point_num}-{method}"]
            train_and_test_mae(nargs)
    