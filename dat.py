import numpy as np
from tqdm import tqdm
from skimage import exposure





class DAT():
    def __init__(self, paths):
        self.paths = paths
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
        return 3 * self.dataset.shape[0]
    
    
    def init_comps(self):
        l = self.aug_dataset_len()
        self.aug_dataset = np.zeros((l, 256, 256, 2))
        
        
    def load_dataset(self):
        self.dataset = np.load(self.path_to_dataset)
        
        
    def augment_img(self, img, mask):
        flippedlr = np.fliplr(img)
        flippedlr_mask = np.fliplr(mask)
        flippedud = np.flipud(img)
        flippedup_mask = np.flipud(mask)
        
        return flippedlr, flippedlr_mask, flippedud, flippedup_mask
    
    
    def augment_dataset(self):
        
        idx = 0
        for i in tqdm(range(len(self.dataset))):
            img = self.dataset[i, :, :, 0]
            mask = self.dataset[i, :, :, 1]
            flr, flr_mask, fud, fud_mask = self.augment_img(img, mask)
            self.aug_dataset[idx, :, :, 0] = img
            self.aug_dataset[idx, :, :, 1] = mask
            idx += 1
            self.aug_dataset[idx, :, :, 0] = flr
            self.aug_dataset[idx, :, :, 1] = flr_mask
            idx += 1
            self.aug_dataset[idx, :, :, 0] = fud
            self.aug_dataset[idx, :, :, 1] = fud_mask
            idx += 1
            
    
    def save_dataset(self):
        np.save(self.paths['aug_dataset'], self.aug_dataset)