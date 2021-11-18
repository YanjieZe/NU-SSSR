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

def train(args):
    utils.set_seed_everywhere(args.seed)

    if args.wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_name, \
		    group=args.wandb_group, job_type=args.wandb_job)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = utils.TrainDataset(args)
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=utils.collect_function)

    loss_function = nn.MSELoss()

    model = utils.make_model(args).to(device)
    model = model.float()

    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    for e in range(args.epoch):
        model.train()
        for idx, img_pair in enumerate(train_loader):
            break
            img_hr = img_pair['hr'].to(device)
            img_lr = img_pair['lr'].to(device)
            
            img_predict = model(img_lr)

            loss = loss_function(img_hr, img_predict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({'loss':loss.item()})
            else:
                print('epoch: %u | idx: %u | loss:%f |'%(e, idx, loss.item()) )

        utils.save_model(model, e, args)
        break

if __name__=='__main__':
    args = parse_args()
    print(args)
    train(args)