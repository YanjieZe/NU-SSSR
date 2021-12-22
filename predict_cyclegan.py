"""
for prediction in CG project using CycleGAN
"""

import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.utils.data import dataloader
import utils
from arguments import parse_args
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torch_data
from models import Generator, Discriminator
import dataset_gan
import matplotlib.pyplot as plt

try:
	import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
	pass


def show_real_and_fake(realA, fakeA, realB, fakeB, epoch, id):
    plt.figure(1)
    plt.subplot(2, 2, 1) 
    plt.imshow(realA)
    plt.title('real A')

    plt.figure(1)
    plt.subplot(2, 2, 2)
    plt.imshow(fakeA)
    plt.title('fake A')

    plt.figure(1)
    plt.subplot(2, 2, 3)
    plt.imshow(realB)
    plt.title('real B')

    plt.figure(1)
    plt.subplot(2, 2, 4)
    plt.imshow(fakeB)
    plt.title('fake B')
    plt.savefig("imgs/fifa/pred_cycleGAN_epoch%u_%u.png"%( epoch, id ))
    # plt.show()


def predict(args):
    utils.set_seed_everywhere(args.seed)

    if args.wandb:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_name, \
		    group=args.wandb_group, job_type=args.wandb_job)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # test_dataset = dataset_gan.TestDataset(args)
    # test_loader = torch_data.DataLoader(dataset=test_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.num_workers,
    #                               pin_memory=True,
    #                               drop_last=True,
    #                               collate_fn=utils.collect_function)

    # use train set

    train_dataset = dataset_gan.TrainDataset(args)
    test_loader = torch_data.DataLoader(dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=utils.collect_function)


    # create model
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    # load weight
    epoch = args.predict_epoch
    epoch = 130
    utils.load_model_with_name(netG_A2B, 'A2B', epoch, args)
    utils.load_model_with_name(netG_B2A, 'B2A', epoch, args)
    utils.load_model_with_name(netD_A, 'DA', epoch, args)
    utils.load_model_with_name(netD_B, 'DB', epoch, args)



    for idx, data in enumerate(test_loader):
        with torch.no_grad():
            
            real_image_A = data["hr"].to(device)
            real_image_B = data["lr"].to(device)
            batch_size = real_image_A.size(0)


            fake_image_B = netG_A2B(real_image_A)
            fake_image_A = netG_B2A(real_image_B)

            for i in range(batch_size):
                hr = real_image_A[i].permute(1,2,0).cpu()
                lr = real_image_B[i].permute(1,2,0).cpu()
                hr_pred = fake_image_A[i].permute(1,2,0).cpu()
                lr_pred = fake_image_B[i].permute(1,2,0).cpu()
                show_real_and_fake(realA=hr, fakeA=hr_pred, realB=lr, fakeB=lr_pred, epoch=epoch,id=i)
                
        break

if __name__=='__main__':
    args = parse_args()
    print(args)
    predict(args)