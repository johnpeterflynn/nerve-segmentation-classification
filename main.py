import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
import torch.optim as optim
import os
from nn_common_modules import losses as additional_losses
import dataloader as dl
import network as nt
import quicknat as qn
import utilz as ut
#from tensorboardX import SummaryWriter
from polyaxon_helper import (get_outputs_path)  
#import losses as lo

model_path = get_outputs_path()
results_path = get_outputs_path()


#to assure reproducability/ take out randomness
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# train params
# =============================================================================
model_name = "QuickNat"
num_epochs = 1
lr = 1e-5

# =============================================================================
# test params
# =============================================================================
model_name_test = "QuickNat"
num_epochs_test = 1
lr_test = 1e-5


# =============================================================================
# params
# =============================================================================

### Optimizer: SGD, Adam
opt = 'Adam'

### Loss function: combined, dice, crossentropy
loss_function = 'combined'

### QuickNat
params = {'num_channels':31,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'kernel_c':1,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_class':2,
                        'se_block': False,
                        'drop_out':0.2}




def initialize_model(model_name):

    model_ft = None
    input_size = 0
    
    if model_name == "SegNet":
        model_ft = nt.SegNet()    
        input_size = input_size
        print('SegNet')
        
    if model_name == "QuickNat":
        model_ft = qn.QuickNat(params)           
        input_size = input_size
        print('QuickNat')

    return model_ft, input_size



### training & validation function
def train_model(model, dataload_train, dataload_val, criterion, optimizer, 
                num_epochs):

    since = time.time()
    epoch_loss_train = np.array([])
    epoch_loss_val = np.array([])
    epochs_count = np.array([])
    
    dice_score = np.array([])
    dice_graph_train = list()
    dice_graph_val = list()
    
    loss_train = list()
    loss_val = list()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric_value = 0.0
    epoch = 0
    
    while epoch in range(num_epochs):
        epoch += 1
        print('-' * 20)
        print('epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        epochs_count = np.append(epochs_count, epoch)
        
        
        ### early stopping (10)
        #if epoch > 10 and loss_val[-10] <= loss_val[-9] <= loss_val[-8] <= loss_val[-7] <= loss_val[-6] <= loss_val[-5] <= loss_val[-4] <= loss_val[-3]<= loss_val[-2] <= loss_val[-1]:
        if epoch > 10 and loss_val[-5] <= loss_val[-4] <= loss_val[-3]<= loss_val[-2] <= loss_val[-1]:
            print('early stopping')
            epoch = num_epochs + 1
            epochs_count = epochs_count[:-1]

        
        else: ### proceed as usual
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() # set model to training mode
                    dataload = dataload_train
                if phase == 'val':
                    model.eval() # set model to evaluate mode
                    dataload = dataload_val
    
                running_loss = 0.0
                dice_sum = 0
                  
                
                for number, x in enumerate(dataload):
                    
                    dice_metric_per_ep = 0
                    
                    inputs = x['image'].float()
                    labels = x['labels'].float()

                    
                    ### against NaN values
                    inputs[torch.isnan(inputs)] = 0
                    labels[torch.isnan(labels)] = 0
                    
                    
                
                ### "sanity check"
#                    if number < 3:
#                        plt.figure()
#                        plt.xlabel('inputs' + phase)
#                        plt.imshow(inputs[0, -1, :, :], cmap='gray')
#                        plt.savefig(os.path.join(results_path, '_sanity check_' + phase + str(number) + '_inputs.png'))
#                        plt.figure()
#                        plt.xlabel('labels' + phase)
#                        plt.imshow(labels[0, :, :, 0], cmap='gray')
#                        plt.savefig(os.path.join(results_path, '_sanity check_' + phase + str(number) + '_labels.png'))
            
                    
# =============================================================================
#              median frequency balancing 
# =============================================================================
                    l = labels.numpy()  
                    l = np.concatenate((l, l), axis=0)
                    l = l[:, :, :, 0] 
                    #class_weights, weights = ut.estimate_weights_mfb(l)                  
                    #class_weights = torch.from_numpy(class_weights)
                    #weights = torch.from_numpy(weights)
                    labels = labels[:,:,:,0]
                    
                    
                  
                    inputs = inputs.to(device)
                    labels = labels.to(device)      
                    #labels_weighted = labels_weighted.to(device)                                                
                    
                    # zero param gradients
                    optimizer.zero_grad()                
                         
                    ### forward
                    outputs = model(inputs)
                    
                    
                    ### which size does output(1) have here ?
                    
                    
                    loss = criterion(outputs, labels.long()) 
#                    if loss_function == 'combined':
#                        loss = criterion(outputs, labels.long(), class_weights)  
#                    if loss_function == 'dice':
#                        loss = criterion(outputs, labels.long(), weights)
#                    if loss_function == 'crossentropy':
#                        loss = criterion(outputs, labels.long(), weights)   
                        
                    
                    
                    running_loss += loss.item() 
                     
                    
                    dice_score = 1 - di(outputs, labels.long())
                    dice_sum += dice_score.item()
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()                  
                        optimizer.step()
                        el = running_loss / (number + 1 )
                        epoch_loss_train = np.append(epoch_loss_train, el) 
                        
                        
                    if phase == 'val':
                        
                        elv = running_loss / (number + 1 )
                        epoch_loss_val = np.append(epoch_loss_val, elv)
                        
                        
                        ### "sanity check"
                        #plt.figure()
                        #plt.imshow(outputs_bin[0,0,:,:], cmap='gray')
                        #plt.xlabel('output training after binary 0')
                        
        
                                          
                if phase == "val":
                    dice_metric_per_ep = dice_sum / (number + 1)
                    if dice_metric_per_ep > best_metric_value:
                        best_metric_value = dice_metric_per_ep
                        best_model_wts = copy.deepcopy(model.state_dict())
                    dice_graph_val = np.append(dice_graph_val, dice_metric_per_ep)
                    
                    ep_lo_va = np.mean(epoch_loss_val) ## compute average loss
                    loss_val = np.append(loss_val, ep_lo_va)
                    
                    print('val loss per epoch: ', ep_lo_va) 
                    print('dice score_validation per epoch: ', dice_metric_per_ep)
    
                    
                if phase == "train":
                    dice_metric_per_ep = dice_sum / (number + 1)
                    dice_graph_train = np.append(dice_graph_train, dice_metric_per_ep)
                    
                    ep_lo_tr = np.mean(epoch_loss_train) ## compute average loss)
                    loss_train = np.append(loss_train, ep_lo_tr)
                    
                    print('train loss per epoch: ', ep_lo_tr)
                    print('dice score_training per epoch: ', dice_metric_per_ep)
    
    print('FINAL dice score_best value:: ', best_metric_value)
    print('FINAL train loss: ', ep_lo_tr)
    print('FINAL val loss: ', ep_lo_va)
    print()
    
    
    if num_epochs <= 10:
        steps = 1
    if 10 < num_epochs <= 100:
        steps = 10
    if 100 < num_epochs <= 1000:
        steps = 100
    if 1000 < num_epochs <= 10000:
        steps = 1000
    
    
    ### dice score, val and train
    plt.figure()
    plt.plot(epochs_count, dice_graph_val, label='dice_score_val', marker='x', color='g', markersize=3)
    plt.plot(epochs_count, dice_graph_train, label='dice_score_train', marker='<', color='m', markersize=1)
    plt.xticks(epochs_count[steps-1::steps])
    plt.legend()
    plt.xlabel('# of epochs')
    plt.ylabel('Dice Score')
    plt.savefig(os.path.join(results_path, '_Dice Score_' + str(lr) + '_' + str(num_epochs) + '.png'))


    ### train and val loss
    plt.figure()
    plt.plot(epochs_count, loss_train, label='Training loss', marker='o', markersize=10, linewidth=4)
    plt.plot(epochs_count, loss_val, label='Validation loss', marker='o', markersize=4)
    plt.xticks(epochs_count[steps-1::steps])
    plt.legend()
    plt.xlabel('# of epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(results_path, 'lr: ' + str(lr) + '_loss_' + '_' + str(num_epochs) + '.png'))


    time_elapsed = time.time() - since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load model weights
    model.load_state_dict(best_model_wts)
    
    print('finished: training function')
    torch.save(model.state_dict(), model_path + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs))
    
    return model


            



### testing function
def test(dataload_test, model_path, results_path):
    print('testing')       
    
    model.load_state_dict(torch.load(model_path + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs)))
    model.eval()

    model.to(device)
    
    i = 0
    for number, x in enumerate(dataload_test):
        i += 1
        inputs, labels = x['image'].float(), x['labels'].float()
        
        images = inputs.to(device)
        outputs = model(images)
        
#        probabilistic_pred = outputs.detach().cpu().numpy()
#        probabilistic_pred_reshaped = np.reshape(probabilistic_pred[0,1,:,:], (dl.input_size, dl.input_size))
#        plt.figure()
#        plt.axis('off')
#        plt.imshow(probabilistic_pred_reshaped, cmap='gray')
#        plt.savefig(os.path.join(results_path, '_probabilistic prediction' + '_' + str(i) + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs) + '.png'))
#        
        outputs = ut.binar(outputs) ### binary output
    
        ### save binary prediction
        binary_pred = np.reshape(outputs[0,1,:,:], (dl.input_size, dl.input_size))
        #plt.figure()
        #plt.axis('off')
        #plt.imshow(binary_pred, cmap='gray')
        #plt.savefig(os.path.join(results_path, '_binary prediction_' + str(i) + '_' + model_name_test + '_' + str(lr_test) + '_' + str(num_epochs_test) + '.png'))
        
        ### show labels + prediction in US image
        ut.show_labels(inputs, labels, binary_pred, results_path, i)
        

    return binary_pred





# =============================================================================
# initialize model
# =============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model,_ = initialize_model(model_name)
model = model.to(device)



# =============================================================================
# optimizer
# =============================================================================

if opt == 'SGD': 
    optimizer = optim.SGD(model.parameters(), lr, momentum = 0.9)
if opt == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr)



# =============================================================================
# loss function, dice loss as metric
# =============================================================================

### https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/losses.py
if loss_function == 'combined':
    criterion = additional_losses.CombinedLoss()
if loss_function == 'dice':
    criterion = additional_losses.DiceLoss()
if loss_function == 'crossentropy':
    criterion = torch.nn.CrossEntropyLoss()


di = additional_losses.DiceLoss()   # dice score/ metric


# =============================================================================
# training
# =============================================================================
model = train_model(model, dl.dataload_train, dl.dataload_val, criterion, optimizer, num_epochs=num_epochs)



# =============================================================================
# testing
# =============================================================================

result = test(dl.dataload_test, model_path, results_path)