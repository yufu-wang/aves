import torch
import numpy as np

def evaluate_iou(proj_masks, masks):
    IOU = []
    for proj_mask, mask in zip(proj_masks, masks):
        
        stack = torch.stack([mask, proj_mask]).byte()
        I = torch.all(stack, 0).sum([0,1]).float()
        U = torch.any(stack, 0).sum([0,1]).float()
        score = (I/U).item()
        IOU.append(score)
        
    return IOU


def evaluate_pck(proj_kpts, keypoints, bboxes=None, size=256):
    PCK05 = []
    PCK10 = []
        
    err = proj_kpts[:,:,:2] - keypoints[:,:,:2]
    err = err.norm(dim=2, keepdim=True)
    
    if bboxes is not None:
        maxHW, ind = torch.max(bboxes[:,2:], dim=1)
    else:
        if type(size) == int:
            maxHW = [size] * len(err)
        else:
            maxHW = size
        
    for i in range(len(err)):
        valid = keypoints[i, :, 2:] > 0
        err_i = err[i][valid]
        err_i = err_i / maxHW[i]
        pck05 = (err_i < 0.05).float().mean().item()
        pck10 = (err_i < 0.10).float().mean().item()
        PCK05.append(pck05)
        PCK10.append(pck10)
    
    return PCK05, PCK10


class average_meter():
    def __init__(self, ):
        self.num = 0.
        self.sum = 0.
        self.convert_types = [torch.Tensor, np.ndarray]

    def collect(self, item):  
        if type(item) in self.convert_types:
            item = item.flatten()
            item = item.tolist()

        if type(item) is not list:
            item = [item]
                
        self.sum += sum(item)
        self.num += len(item)


    def report(self, reset=False):
        score = self.sum / self.num
        if reset is True:
            self.reset()

        return score


    def reset(self):
        self.sum = 0.
        self.num = 0.


