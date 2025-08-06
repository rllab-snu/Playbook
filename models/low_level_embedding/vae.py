
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class VAE_Feature(nn.Module):
    def __init__(self, state_dim, z_dim, H, recon_type="origin"):
        super(VAE_Feature, self).__init__()

        self.nn_encoder = nn.Sequential(
            nn.Linear(state_dim*H, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim*2),
        )

        dec_H = H if recon_type == "origin" else 1
        self.nn_decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim*dec_H),
        )

        self.z_dim = z_dim
        self.recon_type = recon_type

    def forward(self, x_list):
        B = x_list.shape[0]
        x = x_list.view(B, -1)
        y = self.nn_encoder(x)
        mean, var = torch.split(y, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        var = F.softplus(var) + 1e-6
        log_var = torch.log(var)
        std = torch.exp(0.5*log_var)

        kld_loss = torch.mean(-0.5 * torch.sum(1. + log_var - mean**2 - torch.exp(log_var), dim=1))

        standard_samples = torch.normal(mean=torch.zeros(mean.shape), std=torch.ones(mean.shape))
        standard_samples = standard_samples.to(mean.device)
        z_samples = mean + std * standard_samples

        x_hat = self.nn_decoder(z_samples)
        if self.recon_type == "origin":
            rec_loss = torch.mean(torch.sum((x-x_hat)**2, dim=-1))
        elif self.recon_type == "delta":
            x_delta = x_list[:,-1,:] - x_list[:,0,:]
            rec_loss = torch.mean(torch.sum((x_delta-x_hat)**2, dim=-1))
        return mean, z_samples, kld_loss, rec_loss
    
    def get_zs(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        y = self.nn_encoder(x)
        mean, var = torch.split(y, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        var = F.softplus(var) + 1e-6
        log_var = torch.log(var)
        std = torch.exp(0.5*log_var)

        standard_samples = torch.normal(mean=torch.zeros(mean.shape), std=torch.ones(mean.shape))
        standard_samples = standard_samples.to(mean.device)
        z_samples = mean + std * standard_samples
        return z_samples.detach()
    
    def get_logp(self, x, z):
        B = x.shape[0]
        x = x.view(B, -1)
        y = self.nn_encoder(x)
        mean, var = torch.split(y, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        var = F.softplus(var) + 1e-6

        logp = -0.5*(math.log(2*torch.pi) \
                + 2.0*torch.log(var) + (mean-z)**2/var**2)
        logp = torch.sum(logp, axis=1)
        return logp


class VAE_Image(nn.Module):
    def __init__(self, in_dim, z_dim, H, setting):
        super(VAE_Image, self).__init__()

        kernel = setting["kernel"]
        stride = setting["stride"]
        padding = setting["padding"]
        n_selected = setting["n_selected"]
        h_dim = 64

        self.nn_encoder_im = nn.Sequential(
            nn.Conv2d(in_dim*H, h_dim // 2, kernel_size=kernel[0],
                      stride=stride[0], padding=padding[0]),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel[1],
                      stride=stride[1], padding=padding[1]),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel[2],
                      stride=stride[2], padding=padding[2]),
        )

        self.nn_encoder_fc = nn.Sequential(
            nn.Linear(h_dim*n_selected, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim*2),
        )

        self.nn_decoder_fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, h_dim*n_selected),
        )

        self.nn_decoder_im = nn.Sequential(
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel[2], stride=stride[2], padding=padding[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel[1], stride=stride[1], padding=padding[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, in_dim*H, kernel_size=kernel[0],
                               stride=stride[0], padding=padding[0])
        )

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.im_dim = int(math.sqrt(n_selected))

    def forward(self, x):
        y = self.nn_encoder_im(x)
        #y = torch.mean(y, axis=[2,3])
        y = torch.flatten(y, start_dim=1)
        y = self.nn_encoder_fc(y)

        mean, var = torch.split(y, self.z_dim, dim=-1)
        mean = torch.tanh(mean)
        var = F.softplus(var) + 1e-6
        log_var = torch.log(var)
        std = torch.exp(0.5*log_var)

        kld_loss = torch.mean(-0.5 * torch.sum(1. + log_var - mean**2 - torch.exp(log_var), dim=1))

        standard_samples = torch.normal(mean=torch.zeros(mean.shape), std=torch.ones(mean.shape))
        standard_samples = standard_samples.to(mean.device)
        z_samples = mean + std * standard_samples

        y_hat = self.nn_decoder_fc(z_samples)
        y_hat = y_hat.view(y_hat.shape[0],self.h_dim,self.im_dim,self.im_dim)
        x_hat = self.nn_decoder_im(y_hat)
        rec_loss = torch.mean(torch.sum((x-x_hat)**2, dim=[1,2,3]))
        return mean, z_samples, kld_loss, rec_loss

    def reconstruction(self, zs):
        y_hat = self.nn_decoder_fc(zs)
        y_hat = y_hat.view(y_hat.shape[0],self.h_dim,self.im_dim,self.im_dim)
        x_hat = self.nn_decoder_im(y_hat)
        return x_hat