import os
import torch
import numpy as np

def save_model(model, savedir, savename):
    if not os.path.exists(savedir): os.makedirs(savedir)
    results = {
        'model': model.state_dict(),
    }
    torch.save(results, savedir+'/{}.pth'.format(savename))
    
def load_model(model, loaddir, loadname):
    checkpoint = torch.load(loaddir + '/{}.pth'.format(loadname))
    model.load_state_dict(checkpoint['model'])


class Loss_Log():
    def __init__(self, num_losses, max_len=int(1e5)):
        self.losses = []
        for _ in range(num_losses): self.losses.append([])
        self.num_losses = num_losses
        self.max_len = max_len

    def insert_loss(self, loss_values):
        assert len(loss_values) == self.num_losses
        for i, v in enumerate(loss_values):
            self.losses[i].append(v)
            if len(self.losses[i]) > self.max_len:
                self.losses[i] = self.losses[i][-self.max_len:]

    def get_average_loss(self, num_average):
        averages = []
        for i in range(self.num_losses):
            averages.append(np.mean(self.losses[i][-num_average:]))
        return averages
