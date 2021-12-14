import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.utils.data import dataloader
import utils
from arguments import parse_args
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
import tqdm
try:
	import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
	pass

def evaluation(args):

    utils.set_seed_everywhere(args.seed)

    if args.wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_name, \
		    group=args.wandb_group, job_type=args.wandb_job)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    test_dataset = utils.TestDataset(args)
    # test_dataset = utils.TrainDataset(args)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=utils.collect_function)

    loss_function = nn.MSELoss()

    model = utils.make_model(args).to(device)
    model = model.float()

    # load weight
    epoch = args.predict_epoch
    utils.load_model(model, epoch, args)


    model.eval()
    figsize = (args.img_width, args.img_width)

    psnr_pred_total = 0.
    ssim_pred_total = 0.
    psnr_gt_total = 0.
    ssim_gt_total = 0.
    count = 0
    for idx, img_pair in tqdm.tqdm(enumerate(test_loader)):
        
        with torch.no_grad():
            img_hr = img_pair['hr'].to(device)
            img_lr = img_pair['lr'].to(device)
            
            img_predict = model(img_lr)

            img_predict = (img_predict*255).int()
            img_hr = (img_hr*255).int()
            img_lr = (img_lr*255).int()
            for i in range(img_hr.shape[0]): # batch size
                psnr_pred = calculate_psnr(img_predict[i].numpy(), img_hr[i].numpy(), 0, input_order='CHW')
                ssim_pred = calculate_ssim(img_predict[i].numpy(), img_hr[i].numpy(), 0, input_order='CHW')
                psnr_gt = calculate_psnr(img_lr[i].numpy(), img_hr[i].numpy(), 0, input_order='CHW')
                ssim_gt = calculate_ssim(img_lr[i].numpy(), img_hr[i].numpy(), 0, input_order='CHW')
                # print("psnr_pred: %f | ssim_pred: %f | psnr_gt: %f | ssim_gt: %f |"%(psnr_pred, ssim_pred, psnr_gt, ssim_gt))
                
                psnr_pred_total += psnr_pred
                ssim_pred_total += ssim_pred
                psnr_gt_total += psnr_gt
                ssim_gt_total += ssim_gt
                count += 1

    psnr_gt_total /= count
    psnr_pred_total /= count
    ssim_gt_total /= count
    ssim_pred_total /= count      

    print("totally: psnr_pred: %f | ssim_pred: %f | psnr_gt: %f | ssim_gt: %f |"%(psnr_pred_total, ssim_pred_total, psnr_gt_total, ssim_gt_total))
    print("improvement: psnr: %f | ssim: %f |"%(psnr_pred_total-psnr_gt_total, ssim_pred_total-ssim_gt_total) )

if __name__=='__main__':
    args = parse_args()
    print(args)
    evaluation(args)