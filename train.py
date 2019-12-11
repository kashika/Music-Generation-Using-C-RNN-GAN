#import relevant libraries
# The project in implemented using Pytorch and using CPU
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import h5py 
import sys
import os
import json
from model import PerformanceNet
cuda = torch.device("cuda")

# Declared a class to specify the hyperparameters
class hyperparams(object):
    def __init__(self):
        self.instrument = sys.argv[1]  
        self.train_epoch = int(sys.argv[2]) #default training epochs = 300
        self.test_freq = int(sys.argv[3])  #default test frequency = 10 
        self.exp_name = sys.argv[4]
        self.iter_train_loss = []    #variable for storing training loss
        self.iter_test_loss = []     #variable for storing test loss
        self.loss_history = []       #loss history logs
        self.test_loss_history = []     
        self.best_loss = 1e10        #store the minimum loss generated
        self.best_epoch = 0          #stores the epoch with minimum loss   

# Function to process the data. For this project, i have used hdf5 data which has music compositions
def Process_Data(instr, exp_dir):
    #import data from folder
    dataset = h5py.File('data/train_data.hdf5','r')    
    #set score, specefications and status of the piono instruments records
    score = dataset['{}_pianoroll'.format(instr)][:]
    spec = dataset['{}_spec'.format(instr)][:]
    onoff = dataset['{}_onoff'.format(instr)][:]
    #preprocessing
    score = np.concatenate((score, onoff),axis = -1)
    score = np.transpose(score,(0,2,1))
    
    # split the preprocessed data into training and testing set
    X_train, X_test, Y_train, Y_test = train_test_split(score, spec, test_size=0.2) 
    
    #specefy directory for the test data
    test_data_dir = os.path.join(exp_dir,'test_data')
    os.makedirs(test_data_dir)
    
    np.save(os.path.join(test_data_dir, "test_X.npy"), X_test)
    np.save(os.path.join(test_data_dir, "test_Y.npy"), Y_test)    
    
    #implemented the project in CPU using tensorflow DataLoader
    #batch size =16, shuffle = True
    train_dataset = utils.TensorDataset(torch.Tensor(X_train, device=cuda), torch.Tensor(Y_train, device=cuda))
    train_loader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = utils.TensorDataset(torch.Tensor(X_test, device=cuda), torch.Tensor(Y_test,device=cuda))
    test_loader = utils.DataLoader(test_dataset, batch_size=16, shuffle=True) 
    
    return train_loader, test_loader


#Function to perforn training of the model
def train(model, epoch, train_loader, optimizer,iter_train_loss):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):        
        #loss minimized using gradient descent
        optimizer.zero_grad()
        split = torch.split(data, 128, dim=1)
        y_pred = model(split[0].cuda(),split[1].cuda())
        #Loss is claculated using Mean Square Error
        loss_function = nn.MSELoss()
        loss = loss_function(y_pred, target.cuda())
        #Here backpropagation of the loss is performed
        loss.backward()
        iter_train_loss.append(loss.item())
        #training loss is added
        train_loss += loss
        optimizer.step()    
         
        if batch_idx % 2 == 0:
            #Loss per epoch is printed here
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
    return train_loss/ len(train_loader.dataset)

#Function to perform test steps
def test(model, epoch, test_loader, scheduler, iter_test_loss):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for idx, (data, target) in enumerate(test_loader):
            split = torch.split(data,128,dim = 1)
            y_pred = model(split[0].cuda(),split[1].cuda())
            #Loss is claculated using Mean Square Error
            loss_function = nn.MSELoss() 
            loss = loss_function(y_pred,target.cuda())    
            iter_test_loss.append(loss.item())
            test_loss += loss   
            #testing loss is added
        test_loss/= len(test_loader.dataset)
        scheduler.step(test_loss)
        print ('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


def main():    
    #check for exceptions here - file exists, directory exists
    hp = hyperparams()
    try:
        exp_root = os.path.join(os.path.abspath('./'),'experiments')
        os.makedirs(exp_root)
    except FileExistsError:
        pass
    
    exp_dir = os.path.join(exp_root, hp.exp_name)
    os.makedirs(exp_dir)
    # PerformanceNet library is used to evaluate performance of the model
    model = PerformanceNet()
    model.cuda()
    # Adam Optimization performs well for reducing loss in addition to the gradient descent
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.zero_grad()
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #instrument data is provided
    train_loader, test_loader = Process_Data(hp.instrument, exp_dir)
    print ('start training')
    
    # data from file is preprocessed and trained using tensorflow and pytorch
    for epoch in range(hp.train_epoch):
        loss = train(model, epoch, train_loader, optimizer,hp.iter_train_loss)
        hp.loss_history.append(loss.item())
        if epoch % hp.test_freq == 0:
            test_loss = test(model, epoch, test_loader, scheduler, hp.iter_test_loss)
            hp.test_loss_history.append(test_loss.item())
            if test_loss < hp.best_loss:         
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(exp_dir, 'checkpoint-{}.tar'.format(str(epoch + 1 ))))
                hp.best_loss = test_loss.item()    
                hp.best_epoch = epoch + 1    
                with open(os.path.join(exp_dir,'hyperparams.json'), 'w') as outfile:   
                    json.dump(hp.__dict__, outfile)
       

if __name__ == "__main__":
    main()
