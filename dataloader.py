from torch.utils.data import Dataset
from torchvision import transforms
from skimage import transform
import os
from utilz import load_files, norm
import numpy as np
import torch
import random

#data_path = '/data/OPUS_nerve_segmentation/data_2'
data_path = '/data/OPUS_nerve_segmentation/data 3/data 3_1'

print('current data_path:  ', data_path)


randomcrop_size = 224
#input_size = 256
input_size = 400
batch_size = 1
num_workers = 2
p = 0.0 # augmentation probability


# =============================================================================
# dataloader, augmentation, batch
# =============================================================================




# =============================================================================
# class for custom dataset
# read paths in __init__, actually load files in __getitem__
# =============================================================================

class OPUSDataset(Dataset):   

    def __init__(self, phase, transform=None):     
        
        self.transform = transform
        self.phase = phase
        
        
        if phase == 'train':
            self.image_train_list = list()
            self.us_train_list = list()
            self.labels_train_list = list()
            l_OA = os.listdir(os.path.join(data_path, 'Recon_train'))
            l_OA.sort()
            l_US = os.listdir(os.path.join(data_path, 'US_train'))
            l_US.sort()
            l_ROI = os.listdir(os.path.join(data_path, 'ROI_train'))
            l_ROI.sort()
            
            for (x, y, z) in zip(l_OA, l_US, l_ROI):
                self.image_train_list.append(os.path.join(data_path, 'Recon_train', x))    
                self.us_train_list.append(os.path.join(data_path, 'US_train', y))
                self.labels_train_list.append(os.path.join(data_path, 'ROI_train', z))

        
        if phase == 'val':
            self.image_val_list = list()
            self.us_val_list = list()
            self.labels_val_list = list()
            l_OA = os.listdir(os.path.join(data_path, 'Recon_val'))
            l_OA.sort()
            l_US = os.listdir(os.path.join(data_path, 'US_val'))
            l_US.sort()
            l_ROI = os.listdir(os.path.join(data_path, 'ROI_val'))
            l_ROI.sort()
            
            for (x, y, z) in zip(l_OA, l_US, l_ROI):
                self.image_val_list.append(os.path.join(data_path, 'Recon_val', x))
                self.us_val_list.append(os.path.join(data_path, 'US_val', y))
                self.labels_val_list.append(os.path.join(data_path, 'ROI_val', z))  
    
        
        if phase == 'test':
            self.image_test_list = list()
            self.us_test_list = list()
            self.labels_test_list = list()
            l_OA = os.listdir(os.path.join(data_path, 'Recon_test'))
            l_OA.sort()
            l_US = os.listdir(os.path.join(data_path, 'US_test'))
            l_US.sort()
            l_ROI = os.listdir(os.path.join(data_path, 'ROI_test'))
            l_ROI.sort()
            
            for (x, y, z) in zip(l_OA, l_US, l_ROI):
                self.image_test_list.append(os.path.join(data_path, 'Recon_test', x))
                self.us_test_list.append(os.path.join(data_path, 'US_test', y))
                self.labels_test_list.append(os.path.join(data_path, 'ROI_test', z))  
        
        
      
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.image_train_list)
        if self.phase == 'val':
            return len(self.image_val_list)
        if self.phase == 'test':
            return len(self.image_test_list)
        
    
   

    def __getitem__(self, idx):    
        
        if self.phase == 'train':
            
            image_train = load_files(self.image_train_list[idx])
            us_train = load_files(self.us_train_list[idx])
            labels_train = load_files(self.labels_train_list[idx])
            
            image_train = image_train[:,:,:20]
            
            
            ### check for nan
            #print('nan: ', idx, np.where(np.isnan(image_train)==True))
            #t = torch.from_numpy(image_train)
            #x = torch.isnan(t)
            #print('x == 1, nan ', idx, x[x == 1])
                        
            
            ### OPUS
            image_train = np.flip(image_train, axis=1)
            image_train = np.concatenate((image_train, us_train) , axis=2)       
            
            ### US only
            #train_image_list.append(us_train)
        
            image = image_train
            labels = labels_train


        if self.phase == 'val':       
            image_val = load_files(self.image_val_list[idx])
            us_val = load_files(self.us_val_list[idx])
            labels_val = load_files(self.labels_val_list[idx])
            
            ### "sanity check"
            #print(self.image_val_list[idx])
            #print(self.us_val_list[idx])
            #print(self.labels_val_list[idx])
            
            image_val = image_val[:,:,:20]
                      
            ### OPUS
            image_val = np.flip(image_val, axis=1)
            image_val = np.concatenate((image_val, us_val) , axis=2)       
            
            ### US only
            #image_val_list.append(us_val)
        
            image = image_val
            labels = labels_val


        if self.phase == 'test': 
            image_test = load_files(self.image_test_list[idx])
            us_test = load_files(self.us_test_list[idx])
            labels_test = load_files(self.labels_test_list[idx])
            
            image_test = image_test[:,:,:20]
            ### OPUS
            image_test = np.flip(image_test, axis=1)
            image_test = np.concatenate((image_test, us_test) , axis=2)       
            
            ### US only
            #image_test_list.append(us_test)
        
            image = image_test
            labels = labels_test
         
        
        sample = {'image' : image, 'labels' : labels}
        if self.transform:
            sample = self.transform(sample)
        return sample
    





# =============================================================================
# Transforms
# =============================================================================

"""         
class RandomRotation(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']
        
        for x in img:
            for y in lab:
                if random.random() < self.p:
                    r = random.randint(1,3)
                    img = np.rot90(img, k=r, axes=(0, 1))  
                    lab = np.rot90(lab, k=r, axes=(0, 1))   

 
                return {'image': img, 'labels': lab}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
"""      
    
    
class RandomHorizontalFlip(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']
        
        for x in img:
            for y in lab:
                if random.random() < self.p:
                    img = np.flip(img, axis=1)
                    lab = np.flip(lab, axis=1)
 
                return {'image': img, 'labels': lab}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w, d = image.shape[:3]
        new_h, new_w = self.output_size
       

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        labels = labels[top: top + new_h,
                      left: left + new_w] 
        
        return {'image': image, 'labels': labels}



"""
class RandomVerticalFlip(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']
        
        for x in img:
            for y in lab:
                if random.random() < self.p:
                    img = np.flip(img, axis=0)
                    lab = np.flip(lab, axis=0)
 
                return {'image': img, 'labels': lab}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
"""
  

    

class Rescale(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        
        h, w, d = image.shape[:3]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
      
        else:
            new_h, new_w = self.output_size
            
        new_h, new_w = int(new_h), int(new_w)
        
        img = transform.resize(image, (new_h, new_w, d), mode='constant')
        labels = transform.resize(labels, (new_h, new_w), mode='constant')

        
        return {'image': img, 'labels': labels}

        


class ToTensor(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        
        image = norm(image)
        
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}
        
    


# =============================================================================
# instantiate custom dataset, load data, apply transforms, 
# =============================================================================
        
transformed_dataset_train = OPUSDataset('train', transform=transforms.Compose([
                                          #RandomHorizontalFlip(p),
                                          #RandomVerticalFlip(p),
                                          #RandomRotation(p),
                                          Rescale(input_size), 
                                          #RandomCrop(randomcrop_size), 
                                          ToTensor()
                                  ]))

          
transformed_dataset_val = OPUSDataset('val', transform=transforms.Compose([
                                          #RandomHorizontalFlip(p),
                                          #RandomVerticalFlip(p),
                                          #RandomRotation(p),
                                          Rescale(input_size),
                                          #RandomCrop(randomcrop_size),
                                          ToTensor()
                                  ]))
              
transformed_dataset_test = OPUSDataset('test', transform=transforms.Compose([
                                          Rescale(input_size),
                                          ToTensor()
                                  ]))
            



dataload_train = torch.utils.data.DataLoader(transformed_dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataload_val = torch.utils.data.DataLoader(transformed_dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataload_test = torch.utils.data.DataLoader(transformed_dataset_test, num_workers=num_workers)


print('initializing datasets and dataloaders')


            
            
            
            