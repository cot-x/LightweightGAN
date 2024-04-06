#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch import einsum
from einops import rearrange
from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import math
import random
import cv2
import time
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class Mish(nn.Module):
    @staticmethod
    @torch.jit.script
    def mish(x):
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        return Mish.mish(x)


# In[ ]:


class GLU(nn.Module):
    @staticmethod
    @torch.jit.script
    def glu(x):
        channel = x.size(1)
        assert channel % 2 == 0, 'must divide by 2.'
        return x[:, :channel//2] * torch.sigmoid(x[:, channel//2:])
        
    def forward(self, x):
        return GLU.glu(x)


# In[ ]:


class PixelwiseNormalization(nn.Module):
    @staticmethod
    @torch.jit.script
    def pixel_norm(x):
        eps = 1e-8
        return x * torch.rsqrt(torch.mean(x * x, 1, keepdim=True) + eps)
    
    def forward(self, x):
        return PixelwiseNormalization.pixel_norm(x)


# In[ ]:


class GeneratorBlock(nn.Module):
    def __init__(self, input_nc, output_nc, n_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_nc, output_nc * 2, kernel_size=3, stride=1, padding=1)
        self.normalize = PixelwiseNormalization()
        self.activate = GLU()
        
    def forward(self, image):
        image = self.upsample(image)
        image = self.conv(image)
        image = self.normalize(image)
        image = self.activate(image)
        
        return image


# In[ ]:


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1),  # downsample
            PixelwiseNormalization(),
            Mish(),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
        )
           
        self.activation = nn.Sequential(
            PixelwiseNormalization(),
            Mish()
        )

    def forward(self, x):
        out = self.model(x)
        skip = self.conv(x)
        out = out + skip
        out = self.activation(out)
        
        return out


# In[ ]:


class SkipLayerExcitation(nn.Module):
    def __init__(self, dim_input, dim_excitation):
        super().__init__()
        intermediate_nc = max(3, dim_input // 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv1 = nn.Conv2d(dim_excitation, intermediate_nc, kernel_size=4, stride=1, padding=0)
        self.activation = Mish()
        self.conv2 = nn.Conv2d(intermediate_nc, dim_input, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, excitation):
        excitation = self.adaptive_pool(excitation)
        excitation = self.conv1(excitation)
        excitation = self.activation(excitation)
        excitation = self.conv2(excitation)
        excitation = self.sigmoid(excitation)
        x *= excitation
        return x


# In[ ]:


class Generator(nn.Module):
    def __init__(self, num_depth, num_fmap, n_channel=3):
        super().__init__()
        
        self.num_depth = num_depth
        self.blocks = nn.ModuleList([GeneratorBlock(num_fmap(i), num_fmap(i + 1), n_channel) for i in range(num_depth)])
        self.excitations = nn.ModuleList([SkipLayerExcitation(num_fmap(i + 1), num_fmap(i - num_depth//2 + 2)) if math.ceil(num_depth / 2) < i + 1 else None for i in range(num_depth)])
        self.toRGB_128 = nn.Sequential(
            nn.Conv2d(num_fmap(np.log2(128)-1), n_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.toRGB = nn.Sequential(
            nn.Conv2d(num_fmap(num_depth), n_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_list = []
        for i, (block, excitation) in enumerate(zip(self.blocks, self.excitations)):
            x = block(x)
            
            x_list += [x]
            if math.ceil(self.num_depth / 2) < i + 1:
                x = excitation(x, x_list[-(self.num_depth // 2)])
            if x.size(-1) == 128:
                x_128 = x
        
        x_128 = self.toRGB_128(x_128)
        x = self.toRGB(x)
        
        return x, x_128


# In[ ]:


class SimpleDecoder(nn.Module):
    class BasicBlock(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim_in, dim_out * 2, kernel_size=3, stride=1, padding=1),
                PixelwiseNormalization(),
                GLU()
            )
        def forward(self, x):
            return self.block(x)
    
    def __init__(self, input_nc, n_channel=3):
        super().__init__()
        self.block1 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block2 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block3 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        self.block4 = SimpleDecoder.BasicBlock(input_nc, input_nc)
        
        self.toRGB = nn.Sequential(
            nn.Conv2d(input_nc, n_channel, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.toRGB(x)
        return x


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self, num_depth, num_fmap, n_channel=3):
        super().__init__()
        
        self.fromRGB = nn.Conv2d(n_channel, num_fmap(num_depth), kernel_size=3, stride=1, padding=1)
        self.fromRGB_128 = nn.Conv2d(n_channel, num_fmap(np.log2(128)), kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList([DiscriminatorBlock(num_fmap(i+1), num_fmap(i)) for i in range(num_depth)][::-1])
        
        self.decoder1 = SimpleDecoder(num_fmap(2))
        self.decoder2 = SimpleDecoder(num_fmap(1))
        
        self.mse_loss = nn.MSELoss()
        
        self.conv_patch = nn.Conv2d(num_fmap(0), 1, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Sigmoid() # for use to WGAN-gp.
    
    def forward(self, x, is_real=False):
        if is_real:
            real_128x128 = F.interpolate(x, size=(128, 128))
            real_256x256 = F.interpolate(x, size=(256, 256))
        else:
            x_128 = x[1]
            x_128 = self.fromRGB_128(x_128)
            x = x[0]
            
        x = self.fromRGB(x)
            
        for block in self.blocks:
            x = block(x)
            
            if is_real:
                if x.size(-1) == 16:
                    fake_crop_8x8, pos = Util.randomCrop(x, 8)
                    fake_crop_128x128 = self.decoder1(fake_crop_8x8)
                if x.size(-1) == 8:
                    fake_128x128 = self.decoder2(x)
            elif x.size(-1) <= 128:
                x_128 = block(x_128)
        
        out = self.conv_patch(x)
        
        ## Not use PatchGAN
        #out = F.adaptive_avg_pool2d(out, 1).view(out.shape[0], -1) # Global Average Pooling
        
        out = self.activation(x) # for use to WGAN-gp.
        
        if is_real:
            real_crop_128x128 = Util.crop(real_256x256, 128, pos * 16)
            loss_recon = self.mse_loss(real_128x128, fake_128x128) + self.mse_loss(real_crop_128x128, fake_crop_128x128)
            return out, loss_recon
        else:
            out_128 = self.conv_patch(x_128)
            ## Not use PatchGAN
            #out_128 = F.adaptive_avg_pool2d(out_128, 1).view(out_128.shape[0], -1) # Global Average Pooling
            out_128 = self.activation(x_128) # for use to WGAN-gp.
            return out, out_128


# In[ ]:


class RandomErasing:
    def __init__(self, p=0.5, erase_low=0.02, erase_high=0.33, aspect_rl=0.3, aspect_rh=3.3):
        self.p = p
        self.erase_low = erase_low
        self.erase_high = erase_high
        self.aspect_rl = aspect_rl
        self.aspect_rh = aspect_rh
    
    def __call__(self, image):
        if np.random.rand() <= self.p:
            c, h, w = image.shape

            mask_area = np.random.uniform(self.erase_low, self.erase_high) * (h * w)
            mask_aspect_ratio = np.random.uniform(self.aspect_rl, self.aspect_rh)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))

            mask = torch.Tensor(np.random.rand(c, mask_h, mask_w) * 255)

            left = np.random.randint(0, w)
            top = np.random.randint(0, h)
            right = left + mask_w
            bottom = top + mask_h

            if right <= w and bottom <= h:
                image[:, top:bottom, left:right] = mask
        
        return image


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    
    @staticmethod
    def augment(images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            RandomErasing()
        ])
        device = images.device
        return torch.cat([transform(img).unsqueeze(0) for img in images.cpu()], 0).to(device)

    @staticmethod
    def augment_p(images, p):
        device = images.device
        size = images.size(-1)
        images = [img for img in images.cpu()]
        images = [transforms.ToPILImage()(img) for img in images]
        if random.uniform(0, 1) <= p:
            images = [transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5)(img) for img in images]
        if random.uniform(0, 1) <= p:
            images = [transforms.RandomRotation(degrees=30)(img) for img in images]
        images = [transforms.ToTensor()(img) for img in images]
        images = [RandomErasing(p=p)(img) for img in images]
        images = torch.cat([img.unsqueeze(0) for img in images], 0).to(device)
        return images
    
    @staticmethod
    def randomCrop(image, size):
        h = size // (random.randrange(2) + 1) - size // 2
        w = size // (random.randrange(2) + 1) - size // 2
        image = image[:, :, h:h+size, w:w+size]
        return image, (h, w)
    
    @staticmethod
    def crop(image, size, pos):
        image = image[:, :, pos[0]:pos[0]+size, pos[1]:pos[1]+size]
        return image
    
    @staticmethod
    def showImages(dataloader):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()

        for images in dataloader:
            for image in images[0]:
                img = PIL(image)
                fig = plt.figure(dpi=200)
                ax = fig.add_subplot(1, 1, 1) # (row, col, num)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(img)
                #plt.gray()
                plt.show()


# In[ ]:


class Solver:
    def __init__(self, args):
        has_cuda = torch.cuda.is_available() if not args.cpu else False
        self.device = torch.device("cuda" if has_cuda else "cpu")
        
        def num_fmap(stage):
            base_size = self.args.image_size
            fmap_base = base_size * 4
            fmap_max = base_size // 2
            fmap_decay = 1.0
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        
        self.args = args
        self.feed_dim = num_fmap(0)
        self.max_depth = int(np.log2(self.args.image_size)) - 1
        #self.pseudo_aug = 0.0
        self.epoch = 0
        
        self.netG = Generator(self.max_depth, num_fmap).to(self.device)
        self.netD = Discriminator(self.max_depth, num_fmap).to(self.device)

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=2 * self.args.lr, betas=(0, 0.9))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=2 * self.args.lr * self.args.mul_lr_dis, betas=(0, 0.9))
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=4, eta_min=self.args.lr/2)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=4, eta_min=(self.args.lr * self.args.mul_lr_dis)/2)
        
        self.load_dataset()
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
            
    def load_dataset(self):
        self.dataloader = Util.loadImages(self.args.batch_size, self.args.image_dir, self.args.image_size)
        self.max_iters = len(iter(self.dataloader))
            
    def save_state(self, epoch):
        self.netG.cpu(), self.netD.cpu()
        torch.save(self.netG.state_dict(), os.path.join(self.args.weight_dir, f'weight_G.{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.args.weight_dir, f'weight_D.{epoch}.pth'))
        self.netG.to(self.device), self.netD.to(self.device)
        
    def load_state(self):
        if (os.path.exists('weight_G.pth') and os.path.exists('weight_D.pth')):
            self.netG.load_state_dict(torch.load('weight_G.pth', map_location=self.device))
            self.netD.load_state_dict(torch.load('weight_D.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    @staticmethod
    def load(args, resume=True):
        if resume and os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                solver = load(f)
                solver.args = args
                return solver
        else:
            return Solver(args)
        
    def trainGAN_LSGAN(self, epoch, iters, max_iters, real_img, a=0, b=1, c=1, lambda_ms=1):
        ### Train with LSGAN.
        ### for example, (a, b, c) = 0, 1, 1 or (a, b, c) = -1, 1, 0
        
        mse_loss = nn.MSELoss()
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        #noise = torch.Tensor(np.random.normal(0, self.args.lambda_zcr_noise, (real_img.size(0), self.feed_dim, 2, 2))).to(self.device)
        #z = random_data + noise
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        #real_img_aug = Util.augment(real_img) # bCR
        #real_img_aug = Util.augment_p(real_img, self.pseudo_aug) # ADA
        real_src_score, d_loss_recon = self.netD(real_img, is_real=True)
        real_src_loss = torch.sum((real_src_score - b) ** 2)
        
        # Compute loss with fake images.
        fake_img, fake_128 = self.netG(random_data)
        #fake_img_aug = Util.augment_p(fake_img, self.pseudo_aug) # ADA
        #fake_128_aug = Util.augment_p(fake_128, self.pseudo_aug) # ADA
        fake_src_score, fake_128_score = self.netD((fake_img, fake_128))

        # Not APA
        fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        fake_128_loss = torch.sum((fake_128_score - a) ** 2)
        ## APA
        #p = random.uniform(0, 1)
        #if 1 - self.pseudo_aug < p:
        #    fake_src_loss = torch.sum((fake_src_score - b) ** 2) # Pseudo: fake is real.
        #    fake_128_loss = torch.sum((fake_128_score - b) ** 2) # Pseudo: fake is real.
        #else:
        #    fake_src_loss = torch.sum((fake_src_score - a) ** 2)
        #    fake_128_loss = torch.sum((fake_128_score - a) ** 2)
        #
        ## Update Probability Augmentation.
        #lz = (torch.sign(torch.logit(real_src_score)).mean()
        #      - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        #if lz > self.args.aug_threshold:
        #    self.pseudo_aug += self.args.aug_increment
        #else:
        #    self.pseudo_aug -= self.args.aug_increment
        #self.pseudo_aug = min(1, max(0, self.pseudo_aug))

        ## bCR
        #bcr_real = mse_loss(self.netD(real_img, is_real=True)[0], real_src_score)
        #fake_img_aug = Util.augment(fake_img)
        #fake_128_aug = Util.augment(fake_128)
        #fake_src_score_aug, fake_128_score_aug = self.netD((fake_img_aug, fake_128_aug))
        #bcr_fake = mse_loss(fake_src_score_aug, fake_src_score) + mse_loss(fake_128_score_aug, fake_128_score)

        ## zCR
        #z_img, z_128 = self.netG(z)
        #z_score, z_128_score = self.netD((z_img, z_128))
        #zcr_loss = mse_loss(fake_src_score, z_score) + mse_loss(fake_128_score, z_128_score)

        ## for Mode-Seeking
        #_fake_img = Variable(fake_img.data)
        #_random_data = Variable(random_data.data)
        
        # Backward and optimize.
        d_loss = (0.5 * (real_src_loss + fake_src_loss + fake_128_loss) / self.args.batch_size + d_loss_recon
                  #+ self.args.lambda_bcr_real * bcr_real + self.args.lambda_bcr_fake * bcr_fake + self.args.lambda_zcr_dis * zcr_loss
                 )
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/fake_128_loss'] = fake_128_loss.item()
        loss['D/loss_recon'] = d_loss_recon.item()
        #loss['Augment/prob'] = self.pseudo_aug
        #loss['D/bcr_loss'] = (bcr_real + bcr_fake).item()
        #loss['D/zcr_loss'] = zcr_loss.item()
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        
        # Compute loss with fake images.
        fake_img, fake_128 = self.netG(random_data)
        #fake_img_aug = Util.augment_p(fake_img, self.pseudo_aug) # ADA
        #fake_128_aug = Util.augment_p(fake_128, self.pseudo_aug) # ADA
        fake_src_score, fake_128_score = self.netD((fake_img, fake_128))
        fake_src_loss = torch.sum((fake_src_score - c) ** 2)
        fake_128_loss = torch.sum((fake_128_score - c) ** 2)

        ## zCR
        #z_img, z_128 = self.netG(z)
        #zcr_loss = - mse_loss(fake_img, z_img) - mse_loss(fake_128, z_128)
        
        ## Mode Seeking Loss
        #lz = torch.mean(torch.abs(fake_img - _fake_img)) / torch.mean(torch.abs(random_data - _random_data))
        #eps = 1 * 1e-5
        #ms_loss = 1 / (lz + eps) * lambda_ms
        
        # Backward and optimize.
        g_loss = (0.5 * (fake_src_loss + fake_128_loss) / self.args.batch_size
                  #+ self.args.lambda_zcr_gen * zcr_loss
                  #+ ms_loss
                 )
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        loss['G/fake_128_loss'] = fake_128_loss.item()
        #loss['G/ms_loss'] = ms_loss.item()
        #loss['G/zcr_loss'] = zcr_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img, img_path)
            #Util.showImage(fake_img)
        
        return loss
        
    def trainGAN_WGANgp(self, epoch, iters, max_iters, real_img, lambda_ms=1):
        ### Train with WGAN-gp.
        
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        
        # ================================================================================ #
        #                             Train the discriminator                              #
        # ================================================================================ #
        
        # Compute loss with real images.
        real_src_score, d_loss_recon = self.netD(real_img, is_real=True)
        real_src_loss = - torch.mean(real_src_score)
        
        # Compute loss with fake images.
        fake_img, fake_128 = self.netG(random_data)
        fake_src_score, fake_128_score = self.netD((fake_img, fake_128))

        # Not APA
        fake_src_loss = torch.mean(fake_src_score)
        fake_128_loss = torch.mean(fake_128_score)
        ## APA
        #p = random.uniform(0, 1)
        #if 1 - self.pseudo_aug < p:
        #    fake_src_loss = - torch.mean(fake_src_score) # Pseudo: fake is real.
        #    fake_128_loss = - torch.mean(fake_128_score) # Pseudo: fake is real.
        #else:
        #    fake_src_loss = torch.mean(fake_src_score)
        #    fake_128_loss = torch.mean(fake_128_score)
        #
        ## Update Probability Augmentation.
        #lz = (torch.sign(torch.logit(real_src_score)).mean()
        #      - torch.sign(torch.logit(fake_src_score)).mean()) / 2
        #if lz > self.args.aug_threshold:
        #    self.pseudo_aug += self.args.aug_increment
        #else:
        #    self.pseudo_aug -= self.args.aug_increment
        #self.pseudo_aug = min(1, max(0, self.pseudo_aug))
        
        # Compute loss for gradient penalty
        alpha = torch.rand(real_img.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * real_img.data + (1 - alpha) * fake_img.data).requires_grad_(True)
        x_hat_score, _ = self.netD(x_hat, is_real=True)
        
        grad = torch.autograd.grad(outputs=x_hat_score,
                                   inputs=x_hat,
                                   grad_outputs=torch.ones(x_hat_score.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        
        grad = grad.view(grad.size()[0], -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        gp_loss = torch.mean((grad_l2norm - 1)**2)
        
        ## for Mode-Seeking
        #_fake_img = Variable(fake_img.data)
        #_random_data = Variable(random_data.data)
        
        # Backward and optimize.
        d_loss = real_src_loss + fake_src_loss + fake_128_loss + d_loss_recon + self.args.lambda_gp * gp_loss
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        
        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/fake_128_loss'] = fake_128_loss.item()
        loss['D/loss_recon'] = d_loss_recon.item()
        loss['D/loss_gp'] = gp_loss.item()
        #loss['Augment/prob'] = self.pseudo_aug
        
        # ================================================================================ #
        #                               Train the generator                                #
        # ================================================================================ #
        
        random_data = torch.randn(real_img.size(0), self.feed_dim, 2, 2).to(self.device)
        
        # Compute loss with fake images.
        fake_img, fake_128 = self.netG(random_data)
        fake_src_score, fake_128_score = self.netD((fake_img, fake_128))
        fake_src_loss = - torch.mean(fake_src_score)
        fake_128_loss = - torch.mean(fake_128_score)
        
        ## Mode Seeking Loss
        #lz = torch.mean(torch.abs(fake_img - _fake_img)) / torch.mean(torch.abs(random_data - _random_data))
        #eps = 1 * 1e-5
        #ms_loss = 1 / (lz + eps) * lambda_ms
        
        # Backward and optimize.
        g_loss = (fake_src_loss + fake_128_loss
                  #+ ms_loss
                 )
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        # Logging.
        loss['G/loss'] = g_loss.item()
        loss['G/fake_128_loss'] = fake_128_loss.item()
        
        # Save
        if iters == max_iters:
            self.save_state(epoch)
            img_name = str(epoch) + '_' + str(iters) + '.png'
            img_path = os.path.join(self.args.result_dir, img_name)
            save_image(fake_img, img_path)
            #Util.showImage(fake_img)
        
        return loss
    
    def train(self, resume=True):
        print(f'Use Device: {self.device}')
        torch.backends.cudnn.benchmark = True
        
        self.netG.train()
        self.netD.train()
        
        hyper_params = {}
        hyper_params['Image Dir'] = self.args.image_dir
        hyper_params['Result Dir'] = self.args.result_dir
        hyper_params['Weight Dir'] = self.args.weight_dir
        hyper_params['Image Size'] = self.args.image_size
        hyper_params['Learning Rate'] = self.args.lr
        hyper_params["Mul Discriminator's LR"] = self.args.mul_lr_dis
        hyper_params['Batch Size'] = self.args.batch_size
        hyper_params['Num Train'] = self.args.num_train
        hyper_params['Lambda_WGAN-gp'] = self.args.lambda_gp
        #hyper_params['Probability Aug-Threshold'] = self.args.aug_threshold
        #hyper_params['Probability Aug-Increment'] = self.args.aug_increment
        #hyper_params['bCR lambda_real'] = args.lambda_bcr_real
        #hyper_params['bCR lambda_fake'] = args.lambda_bcr_fake
        #hyper_params['zCR lambda_gen'] = args.lambda_zcr_gen
        #hyper_params['zCR lambda_dis'] = args.lambda_zcr_dis
        #hyper_params['zCR lambda_noise'] = args.lambda_zcr_noise

        for key in hyper_params.keys():
            print(f'{key}: {hyper_params[key]}')
        #experiment.log_parameters(hyper_params)
        
        while self.args.num_train > self.epoch:
            self.epoch += 1
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            
            for iters, (data, _) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                data = data.to(self.device, non_blocking=True)

                #loss = self.trainGAN_LSGAN(self.epoch, iters, self.max_iters, data)
                loss = self.trainGAN_WGANgp(self.epoch, iters, self.max_iters, data)
                
                epoch_loss_D += loss['D/loss']
                epoch_loss_G += loss['G/loss']
                #experiment.log_metrics(loss)
            
            epoch_loss = epoch_loss_G + epoch_loss_D
            
            print(f'Epoch[{self.epoch}]'
                  + f' LR[G({self.scheduler_G.get_last_lr()[0]:.5f}) D({self.scheduler_D.get_last_lr()[0]:.5f})]'
                  + f' Loss[G({epoch_loss_G}) + D({epoch_loss_D}) = {epoch_loss}]')
            
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            if resume:
                self.save_resume()
    
    def generate(self, num=100):
        self.netG.eval()
        
        for _ in range(num):
            random_data = torch.randn(1, self.feed_dim, 2, 2).to(self.device)
            fake_img = self.netG(random_data)[0][0,:]
            save_image(fake_img, os.path.join(self.args.result_dir, f'generated_{time.time()}.png'))
            #Util.showImage(fake_img)
        print('New picture was generated.')
        
    def showImages(self):
        Util.showImages(self.dataloader)


# In[ ]:


def main(args):
    solver = Solver.load(args, resume=not args.noresume)
    solver.load_state()
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
    
    #solver.showImages()
    solver.train(not args.noresume)
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--weight_dir', type=str, default='weights')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--lambda_gp', type=float, default=10)
    #parser.add_argument('--aug_threshold', type=float, default=0.6)
    #parser.add_argument('--aug_increment', type=float, default=0.01)
    #parser.add_argument('--lambda_bcr_real', type=float, default=10)
    #parser.add_argument('--lambda_bcr_fake', type=float, default=10)
    #parser.add_argument('--lambda_zcr_noise', type=float, default=0.07)
    #parser.add_argument('--lambda_zcr_dis', type=float, default=20)
    #parser.add_argument('--lambda_zcr_gen', type=float, default=0.5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)

