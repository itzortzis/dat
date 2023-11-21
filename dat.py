import cv2
import torch
import random
from torch import nn
import numpy as np
from tqdm import tqdm
from skimage import exposure





class DAT():
    def __init__(self, paths, params):
        self.paths = paths
        self.params = params
        print("Executing DAT...")
        # self.execute()
        
    
    def execute(self):
        self.init_paths()
        self.load_dataset()
        self.init_comps()
        self.augment_dataset()
        self.save_dataset()
        
    
    def init_paths(self):
        self.path_to_dataset = self.paths["dataset"]
        
        
    def aug_dataset_len(self):
        return int(self.params['aug_factor']) * self.dataset.shape[0]
    
    
    def init_comps(self):
        l = self.aug_dataset_len()
        self.aug_dataset = np.zeros((l, self.params['p_size'], self.params['p_size'], 2))
        
        
    def load_dataset(self):
        self.dataset = np.load(self.path_to_dataset)
    
    
    def generate_hlf(self, img):
        m = nn.Conv2d(1, int(self.params['aug_factor']), 3, stride=2, padding=(1, 1))
        input = torch.from_numpy(img)
        input = torch.unsqueeze(input, 0)
        input = torch.unsqueeze(input, 0)
        input = input.to(torch.float32)
        output = m(input)
        output = torch.squeeze(output, 0)
        img_augs = output.cpu().detach().numpy()
        
        return img_augs
    
    
    def dataset_random_flip(self):
        for i in range(self.aug_dataset_len()):
            r = random.randint(0, 2)
            if r == 0:
                self.aug_dataset[i, :, :, 0] = np.fliplr(self.aug_dataset[i, :, :, 0])
                self.aug_dataset[i, :, :, 1] = np.fliplr(self.aug_dataset[i, :, :, 1])
            if r == 1:
                self.aug_dataset[i, :, :, 0] = np.flipud(self.aug_dataset[i, :, :, 0])
                self.aug_dataset[i, :, :, 1] = np.flipud(self.aug_dataset[i, :, :, 1])
    
    
    def augment_dataset(self):
        
        idx = 0
        for i in tqdm(range(len(self.dataset))):
            img = self.dataset[i, :, :, 0]
            mask = self.dataset[i, :, :, 1]
            mask = cv2.resize(mask, (256, 256))
            
            img_augs = self.generate_hlf(img)
            slot_s = idx
            slot_e = idx + int(self.params['aug_factor'])
            self.aug_dataset[slot_s: slot_e, :, :, 0] = img_augs
            for w in range(slot_s, slot_e):
                self.aug_dataset[w, :, :, 1] = mask
            
            idx += int(self.params['aug_factor'])
        self.dataset_random_flip()
            
    
    def save_dataset(self):
        np.save(self.paths['aug_dataset'], self.aug_dataset)