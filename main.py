import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
import torch.optim as optim
import os
import dataset_loader.lidc_loader as ldl
import network as nt
import quicknat as qn
import utilz as ut
import losses as lo
import argparse


#to assure reproducability/ take out randomness
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser()
parser.add_argument(
        '--num_channels',
        type=int,
        default=1
    )
parser.add_argument(
        '--num_filters',
        type=int,
        default=64
    )
parser.add_argument(
        '--kernel_h',
        type=int,
        default=5
    )
parser.add_argument(
        '--kernel_w',
        type=int,
        default=5
    )
parser.add_argument(
        '--kernel_c',
        type=int,
        default=1
    )
parser.add_argument(
        '--pool',
        type=int,
        default=2
    )
parser.add_argument(
        '--num_class',
        type=int,
        default=2
    )
parser.add_argument(
        '--se_block',
        default=False
    )
parser.add_argument(
        '--drop_out',
        type=float,
        default=0.2
    )
parser.add_argument(
        '--stride_conv',
        type=int,
        default=1
    )
parser.add_argument(
        '--stride_pool',
        type=int,
        default=2
    )
parser.add_argument(
        '--lr',
        type=float,
        default=1e-6
    )
parser.add_argument(
        '--model_name',
        default='QuickNat'
    )
parser.add_argument(
        '--num_epochs',
        type=int,
        default=30
    )
parser.add_argument(
        '--opt',
        default='Adam'
    )
parser.add_argument(
        '--loss_function',
        default='dice'
    )

args = parser.parse_args()

args_dict = vars(args)
args_dict['kernel_h'] = args_dict['kernel_w']
print(args_dict)
print(args_dict)




# =============================================================================
# params
# =============================================================================
opt = args.opt
loss_function = args.loss_function
model_name = args.model_name
num_epochs = args.num_epochs
lr = args.lr

### QuickNat local
#params = {'num_channels':7,
#                        'num_filters':64,
#                        'kernel_h':3,
#                        'kernel_w':3,
#                        'kernel_c':1,
#                        'stride_conv':1,
#                        'pool':2,
#                        'stride_pool':2,
#                        'num_class':2,
#                        'se_block': False,
#                        'drop_out':0.2}


# =============================================================================
# Polyaxon
# =============================================================================
from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
model_path = 'outputs/'  # get_outputs_path()
results_path = 'outputs'  # get_outputs_path()
#experiment = Experiment()
#experiment.set_name(str(args.lr).replace('.', '_') + '-' + str(args.num_epochs) + '-' + args.model_name + '-' + args.opt + '-' + args.loss_function + '-' + str(args.kernel_h))
#experiment.log_params(log_learning_rate=args.log_learning_rate,
#                      max_depth=args.max_depth,
#                      num_rounds=args.num_rounds,
#                      min_child_weight=args.min_child_weight)







def initialize_model(model_name):

    model_ft = None
    input_size = 0
    
    if model_name == "SegNet":
        model_ft = nt.SegNet()    
        input_size = input_size
        print('SegNet')
        
    if model_name == 'QuickNat':
        model_ft = qn.QuickNat(args_dict)           
        input_size = input_size
        print('QuickNat')

    return model_ft, input_size



### training & validation function
def train_model(model, dataload_train, dataload_val, optimizer, 
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
        if epoch > 10 and loss_val[-10] <= loss_val[-9] <= loss_val[-8] <= loss_val[-7] <= loss_val[-6] <= loss_val[-5] <= loss_val[-4] <= loss_val[-3]<= loss_val[-2] <= loss_val[-1]:
        #if epoch > 10 and loss_val[-5] <= loss_val[-4] <= loss_val[-3]<= loss_val[-2] <= loss_val[-1]:
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
                    
                    inputs, labels = x[0], x[1]
                    
                    
                    ### against NaN values
                    inputs[torch.isnan(inputs)] = 0
                    labels[torch.isnan(labels)] = 0
                                       
# =============================================================================
# =============================================================================                                 
                    

                    inputs = inputs.to(device)
                    labels = labels.to(device)                                                  
                    
                    # zero param gradients
                    optimizer.zero_grad()                
                         
                    ### forward
                    outputs = model(inputs)
                    
                    
# =============================================================================
# loss functions, dice loss as metric
# =============================================================================


                    ### without mfb
                    if loss_function == 'combined-':
                        criterion = lo.CombinedLoss()
                        loss = criterion(outputs, labels.long()) 
                    if loss_function == 'crossentropy-':
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels.long()) 
                    
                    ### with mfb
                    if loss_function == 'combined+':
                        raise Exception("Hey that's not implemented!")
                        criterion = lo.CombinedLoss()
                        loss = criterion(outputs, labels.long(), combined_weights.cuda().float())
                    if loss_function == 'dice':
                        criterion = lo.DiceLoss()
                        loss = criterion(outputs, labels.long())
                    if loss_function == 'crossentropy+':                       
                        criterion = torch.nn.CrossEntropyLoss()
                        loss = criterion(outputs.double().cpu(), labels.long().cpu()) 
                        
                    # Dice score/ metric
                    metric = lo.DiceLoss()
                    
# =============================================================================
# =============================================================================
                                          

                    running_loss += loss.item() 
                     
                    
                    dice_score = 1 - metric(outputs, labels.long())
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
                    ep_lo_va = np.mean(epoch_loss_val) ## compute average loss per epoch
                    loss_val = np.append(loss_val, ep_lo_va)
                    
                    print('val loss per epoch: ', ep_lo_va) 
                    print('dice score_validation per epoch: ', dice_metric_per_ep)
                    #experiment.log_metrics(val_loss=ep_lo_va, dice_score_val=dice_metric_per_ep)
                    
                if phase == "train":
                    dice_metric_per_ep = dice_sum / (number + 1)
                    dice_graph_train = np.append(dice_graph_train, dice_metric_per_ep) 
                    ep_lo_tr = np.mean(epoch_loss_train) ## compute average loss per epoch
                    loss_train = np.append(loss_train, ep_lo_tr)
                    
                    print('train loss per epoch: ', ep_lo_tr)
                    print('dice score_training per epoch: ', dice_metric_per_ep)
                    #experiment.log_metrics(train_loss=ep_lo_tr, dice_score_train=dice_metric_per_ep)
    
    print('*** FINAL dice score_val_best value *** ', best_metric_value)
    print('*** FINAL train loss *** ', ep_lo_tr)
    print('*** FINAL val loss*** ', ep_lo_va)
    
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


            



def test(dataload_test, model_path, results_path):
    print('testing')       
    
    model.load_state_dict(torch.load(model_path + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs)))
    model.eval()

    model.to(device)
    
    i = 0
    for number, x in enumerate(dataload_test):
        i += 1
        inputs, labels = x[0], x[1]
        
        images = inputs.to(device)
        outputs = model(images)
        
#        probabilistic_pred = outputs.detach().cpu().numpy()
#        probabilistic_pred_reshaped = np.reshape(probabilistic_pred[0,1,:,:], (dl.input_size, dl.input_size))
#        plt.figure()
#        plt.axis('off')
#        plt.imshow(probabilistic_pred_reshaped, cmap='gray')
#        plt.savefig(os.path.join(results_path, '_probabilistic prediction' + '_' + str(i) + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs) + '.png'))
#        
        outputs = ut.binary(outputs) ### binary output
    
        ### save binary prediction
        binary_pred = np.reshape(outputs[0,1,:,:], (dl.input_size, dl.input_size))
        #plt.figure()
        #plt.axis('off')
        #plt.imshow(binary_pred, cmap='gray')
        #plt.savefig(os.path.join(results_path, '_binary prediction_' + str(i) + '_' + model_name + '_' + str(lr) + '_' + str(num_epochs) + '.png'))
        
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
# training
# =============================================================================
dataload_train, dataload_val, dataload_test = ldl.get_lidc_loaders()
model = train_model(model, dataload_train, dataload_val, optimizer, num_epochs=num_epochs)



# =============================================================================
# testing
# =============================================================================

result = test(dataload_test, model_path, results_path)
