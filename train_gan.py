import torch
import numpy as np
from torch.nn.modules.loss import MSELoss
from torch.utils.data import dataloader
import utils
from arguments import parse_args
import itertools
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torch_data
from models import Generator, Discriminator
import dataset_gan
import tqdm

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

    # create model
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)
    
    netG_A2B.apply(utils.weights_init)
    netG_B2A.apply(utils.weights_init)
    netD_A.apply(utils.weights_init)
    netD_B.apply(utils.weights_init)


    # define loss function (adversarial_loss) and optimizer
    cycle_loss = torch.nn.L1Loss().to(device)
    identity_loss = torch.nn.L1Loss().to(device)
    adversarial_loss = torch.nn.MSELoss().to(device)


    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # dataset
    train_dataset = dataset_gan.TrainDataset(args)
    train_loader = torch_data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=utils.collect_function)
        
    g_losses = []
    d_losses = []

    identity_losses = []
    gan_losses = []
    cycle_losses = []

    fake_A_buffer = utils.ReplayBuffer()
    fake_B_buffer = utils.ReplayBuffer()



    print("Start training...")
    for epoch in range(args.epoch):
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))

        for idx, data in progress_bar:
            
            # fetch data
            real_image_A = data["hr"].to(device)
            real_image_B = data["lr"].to(device)
            batch_size = real_image_A.size(0)

            # real data label is 1, fake data label is 0.
            real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
            fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)
            
            # (1) Update G network: Generators A2B and B2A
            optimizer_G.zero_grad()
            # Identity loss
            # G_B2A(A) should equal A if real A is fed
            identity_image_A = netG_B2A(real_image_A)
            loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
            # G_A2B(B) should equal B if real B is fed
            identity_image_B = netG_A2B(real_image_B)
            loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

            # GAN loss
            # GAN loss D_A(G_A(A))
            fake_image_A = netG_B2A(real_image_B)
            fake_output_A = netD_A(fake_image_A)
            loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
            # GAN loss D_B(G_B(B))
            fake_image_B = netG_A2B(real_image_A)
            fake_output_B = netD_B(fake_image_B)
            loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

            # Cycle loss
            recovered_image_A = netG_B2A(fake_image_B)
            loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

            recovered_image_B = netG_A2B(fake_image_A)
            loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

            # Combined loss and calculate gradients
            errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Calculate gradients for G_A and G_B
            errG.backward()
            # Update G_A and G_B's weights
            optimizer_G.step()



            # (2) Update D network: Discriminator A
            # Set D_A gradients to zero
            optimizer_D_A.zero_grad()

            # Real A image loss
            real_output_A = netD_A(real_image_A)
            errD_real_A = adversarial_loss(real_output_A, real_label)

            # Fake A image loss
            fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
            fake_output_A = netD_A(fake_image_A.detach())
            errD_fake_A = adversarial_loss(fake_output_A, fake_label)

            # Combined loss and calculate gradients
            errD_A = (errD_real_A + errD_fake_A) / 2

            # Calculate gradients for D_A
            errD_A.backward()
            # Update D_A weights
            optimizer_D_A.step()

            
            # (3) Update D network: Discriminator B
            # Set D_B gradients to zero
            optimizer_D_B.zero_grad()

            # Real B image loss
            real_output_B = netD_B(real_image_B)
            errD_real_B = adversarial_loss(real_output_B, real_label)

            # Fake B image loss
            fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
            fake_output_B = netD_B(fake_image_B.detach())
            errD_fake_B = adversarial_loss(fake_output_B, fake_label)

            # Combined loss and calculate gradients
            errD_B = (errD_real_B + errD_fake_B) / 2

            # Calculate gradients for D_B
            errD_B.backward()
            # Update D_B weights
            optimizer_D_B.step()

            progress_bar.set_description(
                f"[{epoch}/{args.epoch - 1}][{idx}/{len(train_loader) - 1}] "
                f"Loss_D: {(errD_A + errD_B).item():.4f} "
                f"Loss_G: {errG.item():.4f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")
                
            if args.wandb:
                wandb.log({'Loss_D':(errD_A + errD_B).item(), 
                            'Loss_G':errG.item() ,
                            'Loss_G_identity':(loss_identity_A + loss_identity_B).item(),
                            'loss_G_GAN':(loss_GAN_A2B + loss_GAN_B2A).item(),
                            'loss_G_cycle':(loss_cycle_ABA + loss_cycle_BAB).item() })
            else:
                pass

        if epoch%10==0:
            utils.save_model_with_name(netG_A2B, 'A2B', epoch, args)
            utils.save_model_with_name(netG_B2A, 'B2A', epoch, args)
            utils.save_model_with_name(netD_A, 'DA', epoch, args)
            utils.save_model_with_name(netD_B, 'DB', epoch, args)

    

if __name__=='__main__':
    args = parse_args()
    print(args)
    train(args)