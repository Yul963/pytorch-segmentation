import torch

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape((3, 1, 1))
        self.std = torch.tensor(std).reshape((3, 1, 1))

    def __call__(self, tensor):
        with torch.no_grad():
            tensor.mul_(self.std).add_(self.mean)
        return tensor
