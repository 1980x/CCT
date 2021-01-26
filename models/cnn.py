'''
Aum Sri Sai Ram
Implementation of ECCT: Ensemble Consensual Collaborative Training for Facial Expression Recognition with Noisy Annotations Resnet models                
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-01-2021
Email: darshangera@sssihl.edu.in
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from model.resnet import *
from model.sharedresnet import sharedresnet18
import pickle
import os
from model.spherenet import SphereNet
from model.light_cnn import LightCNN_29Layers_v2, LightCNN_9Layers

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
def resModel(args): #resnet18
    
    model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False)).to(device)
    
    if not args.pretrained:
       
       checkpoint = torch.load('pretrained/res18_naive.pth_MSceleb.tar')
       pretrained_state_dict = checkpoint['state_dict']
       model_state_dict = model.state_dict()
       
       '''
       for name, param in pretrained_state_dict.items():
           print(name)
           
       for name, param in model_state_dict.items():
           print(name)
       '''  
       for key in pretrained_state_dict:
           if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
               print(key) 
               pass
           else:
               #print(key)
               model_state_dict[key] = pretrained_state_dict[key]

       model.load_state_dict(model_state_dict, strict = False)
       print('Model loaded from Msceleb pretrained')
    else:
       print('No pretrained resent18 model built.')
    return model   


