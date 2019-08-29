from torch.utils.data import Dataset
from torchvision import transforms
from skimage import transform
import os
from utilz import load_files, norm, elastic_deformation
import numpy as np
import torch
import random


data_path = '/data/OPUS_nerve_segmentation/OPUS_data'



randomcrop_size = 224
#input_size = 256
input_size = 400
batch_size = 1
num_workers = 3
p = 0.3 # augmentation probability


# =============================================================================
# dataloader, augmentation, batch
# class for custom dataset
# read paths in __init__, actually load files in __getitem__
# patient_001 - patient_012
# =============================================================================

class OPUSDataset(Dataset):   

    def __init__(self, phase, transform=None):     
        
        self.transform = transform
        self.phase = phase


# =============================================================================
# TRAINING files
# =============================================================================

        self.image_list = list()
        self.us_list = list()
        self.labels_list = list()
        self.patients_list = list()
            
        if phase == 'train':
                    
            
            ## patients for training
            self.patients_list = ('patient_001','patient_002','patient_003','patient_004','patient_005','patient_006','patient_007', 'patient_008','patient_009','patient_010', )
            for x in self.patients_list:
                data_path_patient = os.path.join(data_path, x)
                 
                # nervus medianus
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconOA')):
                    self.image_list.append(os.path.join(data_path_patient, 'medianus/reconOA', filename))
                    self.image_list.append(os.path.join(data_path_patient, 'medianus/reconOA', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconUS')):
                    self.us_list.append(os.path.join(data_path_patient, 'medianus/reconUS', filename))
                    self.us_list.append(os.path.join(data_path_patient, 'medianus/reconUS', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/ROI')):
                    self.labels_list.append(os.path.join(data_path_patient, 'medianus/ROI', filename))
                    self.labels_list.append(os.path.join(data_path_patient, 'medianus/ROI', filename))
                    
#                # nervus ulnaris
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'ulnaris/reconOA', filename))
#                    #self.image_list.append(os.path.join(data_path_patient, 'ulnaris/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'ulnaris/reconUS', filename))
#                    #self.us_list.append(os.path.join(data_path_patient, 'ulnaris/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'ulnaris/ROI', filename))
#                    #self.labels_list.append(os.path.join(data_path_patient, 'ulnaris/ROI', filename))
#                
#                # nervus radialis
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'radialis/reconOA', filename))
#                    #self.image_list.append(os.path.join(data_path_patient, 'radialis/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'radialis/reconUS', filename))
#                    #self.us_list.append(os.path.join(data_path_patient, 'radialis/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'radialis/ROI', filename))
#                    #self.labels_list.append(os.path.join(data_path_patient, 'radialis/ROI', filename))
            
                self.image_list.sort()
                self.us_list.sort()
                self.labels_list.sort()
            
# =============================================================================
# VALIDATION files
# =============================================================================

        if phase == 'val':
            
            
            ### patients for validation
            self.patients_list = ('patient_011', )
            for x in self.patients_list:
                data_path_patient = os.path.join(data_path, x)

                # nervus medianus
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconOA')):
                    self.image_list.append(os.path.join(data_path_patient, 'medianus/reconOA', filename))
                    self.image_list.append(os.path.join(data_path_patient, 'medianus/reconOA', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconUS')):
                    self.us_list.append(os.path.join(data_path_patient, 'medianus/reconUS', filename))
                    self.us_list.append(os.path.join(data_path_patient, 'medianus/reconUS', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/ROI')):
                    self.labels_list.append(os.path.join(data_path_patient, 'medianus/ROI', filename))
                    self.labels_list.append(os.path.join(data_path_patient, 'medianus/ROI', filename))
                
#                # nervus ulnaris
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'ulnaris/reconOA', filename))
#                    #self.image_list.append(os.path.join(data_path_patient, 'ulnaris/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'ulnaris/reconUS', filename))
#                    #self.us_list.append(os.path.join(data_path_patient, 'ulnaris/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'ulnaris/ROI', filename))
#                    #self.labels_list.append(os.path.join(data_path_patient, 'ulnaris/ROI', filename))
#                
#                # nervus radialis
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'radialis/reconOA', filename))
#                    #self.image_list.append(os.path.join(data_path_patient, 'radialis/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'radialis/reconUS', filename))
#                    #self.us_list.append(os.path.join(data_path_patient, 'radialis/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'radialis/ROI', filename))
#                    #self.labels_list.append(os.path.join(data_path_patient, 'radialis/ROI', filename))
            
                self.image_list.sort()
                self.us_list.sort()
                self.labels_list.sort()            
            
                    
    
# =============================================================================
# TESTING files
# =============================================================================
        if phase == 'test':
            
            
            ### patients for testing
            self.patients_list = ('patient_012', )
            for x in self.patients_list:
                data_path_patient = os.path.join(data_path, x)
                 
                # nervus medianus
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconOA')):
                    self.image_list.append(os.path.join(data_path_patient, 'medianus/reconOA', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/reconUS')):
                    self.us_list.append(os.path.join(data_path_patient, 'medianus/reconUS', filename))
                for filename in os.listdir(os.path.join(data_path_patient, 'medianus/ROI')):
                    self.labels_list.append(os.path.join(data_path_patient, 'medianus/ROI', filename))
                
#                # nervus ulnaris
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'ulnaris/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'ulnaris/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'ulnaris/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'ulnaris/ROI', filename))
#                
#                # nervus radialis
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconOA')):
#                    self.image_list.append(os.path.join(data_path_patient, 'radialis/reconOA', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/reconUS')):
#                    self.us_list.append(os.path.join(data_path_patient, 'radialis/reconUS', filename))
#                for filename in os.listdir(os.path.join(data_path_patient, 'radialis/ROI')):
#                    self.labels_list.append(os.path.join(data_path_patient, 'radialis/ROI', filename))            
            
                self.image_list.sort()
                self.us_list.sort()
                self.labels_list.sort()            
            
   

        

    def __len__(self):
        return len(self.image_list)
            
   

    def __getitem__(self, idx):    
         
        image = load_files(self.image_list[idx])
        us = load_files(self.us_list[idx])
        labels = load_files(self.labels_list[idx])
        

        if labels.ndim < 3:
            labels = np.expand_dims(labels, axis=2)

        #image = image[:,:,0:1]
                         
        ### OPUS
        image = np.flip(image, axis=1)
        image = np.concatenate((image, us) , axis=2)  
        
        ### US only
        #train_image_list.append(us_train)
        
        sample = {'image' : image, 'labels' : labels}
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    





# =============================================================================
# Transforms
# =============================================================================

    
class RandomHorizontalFlip(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']
            
        if random.random() < self.p:
            img = np.flip(img, axis=1)
            lab = np.flip(lab, axis=1)
        
        return {'image': img, 'labels': lab}



class elastic_deform(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['labels']
               
        x_coo = np.random.randint(100, 300)
        y_coo = np.random.randint(100, 300)
        dx = np.random.randint(50, 200)
        dy = np.random.randint(50, 200)
        if random.random() < self.p:
            img = elastic_deformation(img, x_coo, y_coo, dx, dy)
            lab = elastic_deformation(lab, x_coo, y_coo, dx, dy)
        
            lab = np.where(lab <= 20, 0, lab)
            lab = np.where(lab > 20, 255, lab)
        
        return {'image': img, 'labels': lab}

  

   
    
    

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
        elastic_deform(p),                                  
        RandomHorizontalFlip(p),
        Rescale(input_size),
        ToTensor() ]))

          
transformed_dataset_val = OPUSDataset('val', transform=transforms.Compose([
        elastic_deform(p),
        RandomHorizontalFlip(p),
        Rescale(input_size),
        ToTensor() ]))
              
transformed_dataset_test = OPUSDataset('test', transform=transforms.Compose([
                                          Rescale(input_size),
                                          ToTensor()
                                  ]))
            



dataload_train = torch.utils.data.DataLoader(transformed_dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataload_val = torch.utils.data.DataLoader(transformed_dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=True)
dataload_test = torch.utils.data.DataLoader(transformed_dataset_test, num_workers=num_workers)


print('initializing datasets and dataloaders')


            
            
            
            