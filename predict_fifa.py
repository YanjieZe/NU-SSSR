"""
for prediction in DIP project using CycleGAN
"""
import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.utils.data import dataloader
import torchvision.transforms as transforms
import utils
from arguments import parse_args
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torch_data
from models import Generator, Discriminator
import dataset_gan
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
import cv2
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

    plt.subplots_adjust(wspace =0.3, hspace =0.3)

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

    pathA = 'data/fifa2real/test/A/2701.jpg'
    pathB = 'data/fifa2real/test/B/602.jpg'
    real_image_A = Image.open(pathA) # RGB
    real_image_B = Image.open(pathB)

    image_size = (256,256)

    pre_process = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    ])
    real_image_A = pre_process(real_image_A).unsqueeze(0)
    real_image_A = real_image_A.to(device)
    real_image_B = pre_process(real_image_B).unsqueeze(0)
    real_image_B = real_image_B.to(device)


    # create model
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    # load weight
    epoch = args.predict_epoch
    epoch = 130
    args.log_dir = 'logs/cycleGAN_fifa'
    utils.load_model_with_name(netG_A2B, 'A2B', epoch, args)
    utils.load_model_with_name(netG_B2A, 'B2A', epoch, args)
    utils.load_model_with_name(netD_A, 'DA', epoch, args)
    utils.load_model_with_name(netD_B, 'DB', epoch, args)



    id = 1
    with torch.no_grad():
        fake_image_B = netG_A2B(real_image_A)
        fake_image_A = netG_B2A(real_image_B)

        # vutils.save_image(fake_image_A, 'test.jpg')
        vutils.save_image(fake_image_A, 'imgs/fifa/test1.jpg', normalize=True)
        vutils.save_image(fake_image_A, 'imgs/fifa/test2.jpg', normalize=True)

        fake_image_A = Image.open('imgs/fifa/test1.jpg')
        fake_image_B = Image.open('imgs/fifa/test2.jpg')

        real_image_A = real_image_A[0].permute(1,2,0).cpu().numpy()
        real_image_B = real_image_B[0].permute(1,2,0).cpu().numpy()


        # real_image_A = real_image_A[:, :, ::-1].copy() 
        

        show_real_and_fake(realA=real_image_A, fakeA=fake_image_A, realB=real_image_B, fakeB=fake_image_B, epoch=epoch, id=id)
        

if __name__=='__main__':
    args = parse_args()
    print(args)
    predict(args)