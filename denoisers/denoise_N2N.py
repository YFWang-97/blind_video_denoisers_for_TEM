#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# BEWARE this script implements the augmentations with a mandatory downsampling#
#--------------------remove by eliminating lines 143 to 151--------------------#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
import time
import numpy as np
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage import io
import tifffile
from denoisers.Neighbor2Neighbor.arch_unet import UNet
import random
from denoisers import utils
import time
import warnings
import os
warnings.filterwarnings("ignore")


logs = {}
time_init = time.time()

operation_seed_counter = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DataSet(torch.utils.data.Dataset):
    def __init__(self, filename, image_size = None, transforms = False, valid = False, type = 1):
        super().__init__()
        self.x = image_size
        self.img = io.imread(filename)
        self.transforms = transforms
        self.valid = valid
        self.type = type
    
    def __len__(self):
        return len(self.img)

    def __getitem__(self, inp_index):
        out = np.expand_dims(np.asarray(self.img[inp_index]), axis = 0)

        C, H, W = out.shape
        if self.valid:
            
            if self.type ==1:
                startx = H//2-(self.x//2)
                starty = W//2-(self.x//2)
                
                out = out[:, starty:starty+self.x, startx:startx+self.x]
            
            elif self.type == 2:
                out = out[:, :self.x, :self.x]
            elif self.type == 3:
                out = out[:, :self.x, -self.x:]
            elif self.type == 4:
                out = out[:, -self.x:, :self.x]
            elif self.type == 5:
                out = out[:, -self.x:, -self.x:]
            
            return torch.Tensor(np.float32(out)).to(device)
        
        if self.x is not None:
            h = np.random.randint(0, H-self.x)
            w = np.random.randint(0, W-self.x)
            out = out[:, h:h+self.x, w:w+self.x]
        
        if self.transforms:
            subsample = random.choice([0, 1, 2, 3])
            if subsample // 2 == 0:
                out = out[:,1::2,:]
            else:
                out = out[:,:-1:2,:]
            if subsample % 2 == 0:
                out = out[:,:,1::2]
            else:
                out = out[:,:,:-1:2]
                
            invert = random.choice([0, 1, 2])
            if invert == 1:
                out = out[:, :, ::-1]
            elif invert == 2:
                out = out[:, ::-1, :]

            rotate = random.choice([0, 1, 2, 3])
            if rotate != 0:
                out = np.rot90(out, rotate, (1, 2))
        out = out.astype(int)

        return torch.FloatTensor(out.copy()).to(device)

class denoiser():
    def __init__(self):
        self.parallel = True
        self.n_channel = 1
        self.n_feature = 48
        self.lr = 1e-4
        self.gamma = 0.5
        self.Lambda1 = 1.0
        self.Lambda2 = 1.0

    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size,
                            w // block_size)

    def generate_mask_pair(self, img):
        # prepare masks (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                            dtype=torch.bool,
                            device=img.device)
        mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                            dtype=torch.bool,
                            device=img.device)
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 3], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device)
        rd_idx = torch.ones(size=(n * h // 2 * w // 2, ),
                            dtype=torch.int64,
                            device=img.device)*random.randint(0, 7)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(start=0,
                                    end=n * h // 2 * w // 2 * 4,
                                    step=4,
                                    dtype=torch.int64,
                                    device=img.device).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2

    def generate_subimages(self, img, mask):
        n, c, h, w = img.shape
        subimage = torch.zeros(n,
                            c,
                            h // 2,
                            w // 2,
                            dtype=img.dtype,
                            layout=img.layout,
                            device=img.device)
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i:i + 1, :, :], block_size=2)
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
            subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
                n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        return subimage
    def train(self, args):
        num_epochs = args.num_epochs
        ratio = num_epochs / 10
        data = args.data
        batch_size = args.batch_size

        # Dataset
        ds = DataSet(data, args.image_size, args.transforms)
        p = int(0.7*len(ds))
        valid, train = torch.utils.data.random_split(ds, [len(ds)-p, p], generator=torch.Generator().manual_seed(314))
        
        # Network
        network = UNet(in_nc=self.n_channel,
                    out_nc=self.n_channel,
                    n_feature=self.n_feature)
        network.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'Neighbor2Neighbor/N2N.pt')), strict=False)
        if self.parallel:
            network = torch.nn.DataParallel(network)
        network = network.to(device)
        
        # about training scheme
        optimizer = optim.Adam(network.parameters(), lr=self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[
                                                int(20 * ratio) - 1,
                                                int(40 * ratio) - 1,
                                                int(60 * ratio) - 1,
                                                int(80 * ratio) - 1
                                            ],
                                            gamma=self.gamma)
        

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0)

        train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_upsnr", "train_ssim"])}
        valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_upsnr", "valid_ssim"])}
                    
        best_loss = 100000
        for epoch in range(num_epochs):
            t_0, t_data, t_den, t_loss, t_train = 0, 0, 0, 0, 0
            Lambda = 1
            
            
            for meter in train_meters.values():
                meter.reset()
            logs['loss1'] = []
            logs['loss2'] = []
            logs['loss'] = []
            logs['loss1v'] = []
            logs['loss2v'] = []
            logs['lossv'] = []
            
            network.train()
            
            train_bar = utils.ProgressBar(train_loader, epoch)
            loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
            for inputs in train_bar:
                
                t_0 = time.time()
                optimizer.zero_grad()
                # data
                noisy = inputs
                noisy = noisy.to(device)

                mask1, mask2 = self.generate_mask_pair(noisy)
                noisy_sub1 = self.generate_subimages(noisy, mask1)
                noisy_sub2 = self.generate_subimages(noisy, mask2)
                t_data += time.time() - t_0
                t_0 = time.time()
                with torch.no_grad():
                    noisy_denoised = network(noisy)
                t_den += time.time()-t_0
                t_0 = time.time()
                noisy_denoised_sub1 = self.generate_subimages(noisy_denoised, mask1)
                noisy_denoised_sub2 = self.generate_subimages(noisy_denoised, mask2)
                t_data += time.time() - t_0
                t_0 = time.time()
                noisy_output = network(noisy_sub1)
                noisy_target = noisy_sub2
                t_den += time.time()-t_0
                t_0 = time.time()
                
                diff = noisy_output - noisy_target
                exp_diff = noisy_denoised_sub1 - noisy_denoised_sub2

                loss1 = torch.mean(diff**2)
                loss2 = Lambda * torch.mean((diff - exp_diff)**2)
                loss_all = self.Lambda1 * loss1 + self.Lambda2 * loss2

                logs['loss1'].append(loss1.item())
                logs['loss2'].append(torch.mean((diff - exp_diff)**2).item())
                logs['loss'].append(loss_all.item())
                t_loss += time.time()-t_0
                t_0 = time.time()
                loss_all.backward()
                optimizer.step()
                t_train += time.time()-t_0
                t_0 = time.time()

                # calculate metrics
                cur_psnr = utils.psnr(noisy_sub1, noisy_output, False)
                cur_ssim = utils.ssim(noisy_sub1, noisy_output, False)
                cur_umse = loss1 - torch.mean((noisy_sub1 - noisy_sub2) ** 2) / 2
                cur_upsnr= 10 * torch.log10(255 ** 2 / cur_umse)
                
                train_meters["train_loss"].update(loss_all.item())
                train_meters["train_psnr"].update(cur_psnr.item())
                train_meters["train_ssim"].update(cur_ssim.item())
                train_meters["train_upsnr"].update(cur_upsnr.item())
            
                loss_avg += loss_all.item()
                psnr_avg += cur_psnr.item()
                ssim_avg += cur_ssim.item()
                count += 1
            
                train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)
            scheduler.step()

            logging.info(train_bar.print(dict(**train_meters, lr=optimizer.param_groups[0]["lr"])))

            network.eval()
            for meter in valid_meters.values():
                meter.reset()
            valid_bar = utils.ProgressBar(valid_loader)
        
            loss_avg, psnr_avg, ssim_avg, count = 0, 0, 0, 0
            for sample in valid_bar:
                with torch.no_grad():
                    t_0 = time.time()
                    noisy_im = sample
                    # padding to square
                    H = noisy_im.shape[-2]
                    W = noisy_im.shape[-1]
                    val_size = (max(H, W) + 31) // 32 * 32
                    noisy_im = torch.Tensor(np.pad(noisy_im.cpu().numpy(),
                                    [[0, 0], [0, 0], [0, val_size - H], [0, val_size - W]],
                                    'reflect'))
                    t_data += time.time() - t_0
                    t_0 = time.time()
                    prediction = network(noisy_im)
                    prediction = prediction[:, :, :H, :W]
                    t_den += time.time() - t_0
                    t_0 = time.time()
                    
                    # calculate metrics
                    cur_psnr = utils.psnr(sample, prediction, False)
                    cur_ssim = utils.ssim(sample, prediction, False)
                        
                    valid_meters["valid_psnr"].update(cur_psnr.item())
                    valid_meters["valid_ssim"].update(cur_ssim.item())
                    
                    mask1, mask2 = self.generate_mask_pair(noisy_im.to(device))
                    noisy_sub1 = self.generate_subimages(noisy_im, mask1)
                    noisy_sub2 = self.generate_subimages(noisy_im, mask2)
                    t_data += time.time() - t_0
                    t_0 = time.time()
                    
                    noisy_denoised = network(noisy_im)
                    t_den += time.time() - t_0
                    t_0 = time.time()
                    noisy_denoised_sub1 = self.generate_subimages(noisy_denoised, mask1)
                    noisy_denoised_sub2 = self.generate_subimages(noisy_denoised, mask2)
                    t_data += time.time() - t_0
                    t_0 = time.time()

                    noisy_output = network(noisy_sub1)
                    t_den += time.time() - t_0
                    t_0 = time.time()
                    noisy_target = noisy_sub2.to(device)
                    diff = noisy_output - noisy_target
                    exp_diff = noisy_denoised_sub1 - noisy_denoised_sub2

                    loss1 = torch.mean(diff**2)
                    loss2 = Lambda * torch.mean((diff - exp_diff)**2)
                    loss_all = self.Lambda1 * loss1 + self.Lambda2 * loss2
                    t_loss += time.time() - t_0
                    t_0 = time.time()
                    
                    cur_umse = loss1 - torch.mean((noisy_sub1 - noisy_sub2) ** 2) / 2
                    cur_upsnr= 10 * torch.log10(255 ** 2 / cur_umse)
                    valid_meters["valid_upsnr"].update(cur_upsnr.item())

                    logs['loss1v'].append(loss1.item())
                    logs['loss2v'].append(torch.mean((diff - exp_diff)**2).item())
                    logs['lossv'].append(loss_all.item())

                    loss_avg += loss_all / batch_size
                    psnr_avg += cur_psnr.item()
                    ssim_avg += cur_ssim.item()
                    count += 1
                    
            if loss_avg/count < best_loss:
                best_loss = loss_avg/count
                best_model = network.state_dict()
        
        network.load_state_dict(best_model)
        return network

    def denoise(self, model, args):
        data = args.data
        # Denoise the video
        ds = DataSet(data)
        denoised = np.zeros_like(ds.img)
        model.eval()
        for k in range(len(ds)):
            with torch.no_grad():
                H = ds[k].shape[-2]
                W = ds[k].shape[-1]
                
                cropped_ds = ds[k]
                o = model(cropped_ds.unsqueeze(0))
        
                o[:, :, :-1:2, :-1:2] = model(cropped_ds[ :, :-1:2, :-1:2].unsqueeze(0))
                o[:, :, 1::2, :-1:2] = model(cropped_ds[ :, 1::2, :-1:2].unsqueeze(0))
                o[:, :, :-1:2, 1::2] = model(cropped_ds[ :, :-1:2, 1::2].unsqueeze(0))
                o[:, :, 1::2, 1::2] = model(cropped_ds[ :, 1::2, 1::2].unsqueeze(0))  
                
                o = o.cpu().numpy()
                denoised[k] = o
        tensor_noisy = torch.Tensor(np.float32(ds.img)).unsqueeze(1)
        tensor_denoised = torch.Tensor(np.float32(denoised)).unsqueeze(1)
        uMSE, uPSNR = utils.uMSE_uPSNR(ds, model)

        print('MSE: ', utils.mse(tensor_noisy, tensor_denoised))    
        print('uMSE:', uMSE)
        print('uPSNR:', uPSNR)    
        print('PSNR: ', utils.psnr(tensor_noisy, tensor_denoised))
        print('SSIM: ', utils.ssim(tensor_noisy, tensor_denoised))
        return denoised
