import os
from glob import glob
from PIL import Image 
import numpy as np
import torch

import constants


class CUB_Dataset(torch.utils.data.Dataset):
    '''
    
    '''
    def __init__(self, root='data/CUB', species_id=None, species=None):
        if species_id is None:
            if species is None:
                print('No specified species or id. Default: 0')
                species_id = 0
            else:
                species_id = constants.Species_to_id[species]

        self.root = root
        self.species = constants.Species[species_id]
        self.cub_category = constants.Cub_category[species_id]
        self.meta = constants.Species_meta[self.species]


        self.segmentations = np.load(root+'/segmentations/'+self.cub_category+'.npy')
        self.keypoints = np.load(root+'/keypoints/'+self.cub_category+'.npy')

        images = []
        imgpaths = sorted(glob(root+'/images/'+self.cub_category+'/*'))
        for imgpath in imgpaths:
            images.append(np.array(Image.open(imgpath)))
        self.images = np.stack(images, axis=0)
        
            
    def __getitem__(self, idx):
        img = self.images[idx]
        seg = self.segmentations[idx]
        kpt = self.keypoints[idx]
        
        return img, seg, kpt

    def __len__(self):
        return len(self.images)
        