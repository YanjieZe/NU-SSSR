import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.utils.data import dataloader
import utils
from arguments import parse_args
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
try:
	import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
	pass

def predict(args):
    # utils.set_seed_everywhere(args.seed)

    if args.wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_name, \
		    group=args.wandb_group, job_type=args.wandb_job)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_dataset = utils.TestDataset(args)
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
    for idx, img_pair in enumerate(test_loader):
        with torch.no_grad():
            img_hr = img_pair['hr'].to(device)
            img_lr = img_pair['lr'].to(device)
            
            img_predict = model(img_lr)

            for i in range(args.batch_size):
                hr = img_hr[i].permute(1,2,0)
                lr = img_lr[i].permute(1,2,0)
                pred_hr = img_predict[i].permute(1,2,0)
                utils.show_gt_and_pred(img_hr=hr, img_lr=lr, pred_hr=pred_hr )

            loss = loss_function(img_hr, img_predict)
        
            print('idx: %u | loss:%f |'%(idx, loss.item()) )
        break

if __name__=='__main__':
    args = parse_args()
    print(args)
    predict(args)