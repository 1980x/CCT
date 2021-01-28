'''
Aum Sri Sai Ram
Implementation of ECCT: Ensemble Consensual Collaborative Training for Facial Expression Recognition with Noisy Annotations           
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-01-2021
Email: darshangera@sssihl.edu.in
'''
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import sys
import datetime
import cv2
import argparse,random
from PIL import Image
import pandas as pd

from algorithm.noisyfer import noisyfer 
import image_utils

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--fec_path', type=str, default='data/FEC/', help='FEC dataset path.')
    
parser.add_argument('--pretrained', type=str, default='pretrained/res18_naive.pth_MSceleb.tar',
                        help='Pretrained weights')

parser.add_argument('--resume', type=str, default='checkpoints')           
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.1)

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='pairflip')

parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
                    
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
                    
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='fec')

parser.add_argument('--noise_file', type=str, help='trainfile path', default='fec_train_expression_by_SCN.txt')

parser.add_argument('--test_file', type=str, help='test path', default='fec_result_file_PMCRH_16.csv') 

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

parser.add_argument('--beta', type=float, default=.65,  help='..based on ')
                 
parser.add_argument('--num_models', type=int, default=3)

parser.add_argument('--num_classes', type=int, default=7, help='number of expressions(class)')

parser.add_argument('--co_lambda_max', type=float, default=.9,    help='..based on ')

parser.add_argument('--n_epoch', type=int, default=25)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--co_lambda', type=float, default=0.1)

parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--evaluate', type=int, default=1)

parser.add_argument('--model_type', type=str, help='[res]', default='res')

parser.add_argument('--save_model', type=str, help='save model?', default="False")

parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

parser.add_argument('--model_dir','-m', default='checkpoints/fecrepeat', type=str)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

                         
class FECDataset(data.Dataset):
    def __init__(self, fec_path, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.fec_path = fec_path

        NAME_COLUMN = 1              
        LABEL_COLUMN = 3
        df_train_clean = pd.read_csv(os.path.join(self.fec_path, args.noise_file), sep=',', header=None)        
                
        if phase == 'train':            
            dataset_train_clean = df_train_clean
            dataset_train_clean = dataset_train_clean[dataset_train_clean[LABEL_COLUMN]  < 7 ]
            dataset_train_clean = dataset_train_clean[dataset_train_clean[LABEL_COLUMN]  >=0 ]   
            self.label = dataset_train_clean.iloc[:, LABEL_COLUMN].values              
            file_names = dataset_train_clean.iloc[:, NAME_COLUMN].values
                        
        else:
            
            NAME_COLUMN = 0              
            LABEL_COLUMN = 1
            df_test = pd.read_csv(os.path.join(self.fec_path, args.test_file), sep=',', header=None)
            df_test = df_test[df_test[LABEL_COLUMN]  < 7 ]
            dataset = df_test[df_test[LABEL_COLUMN]  >=0 ]        
            
            self.label = dataset.iloc[:, LABEL_COLUMN].values             
            file_names = dataset.iloc[:, NAME_COLUMN].values
            
        
        self.noise_or_not = (self.label == self.label) 
        self.file_paths = []
        
        for f in file_names:
            
            if phase == 'train': 
               f = 'aligned_train_Data_FEC/'+f
            else:
               f = 'aligned_test_Data_FEC/'+str(f)+'.jpg'
               
            path = os.path.join(self.fec_path,  f)
            
            self.file_paths.append(path)
            
       
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
 
        
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx                         
                            
       
if  args.dataset == 'fec':   
    input_channel = 3
    num_classes = 7
    init_epoch = 5
  

    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   std=[0.229, 0.224, 0.225]),
        ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])])
                                 
    train_dataset = FECDataset(args.fec_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())                                                                            
    test_dataset = FECDataset(args.fec_path, phase = 'test', transform = data_transforms_val)  
      
    print('Validation set size:', test_dataset.__len__())
    
                            

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate



def main():
    print('\n\t\t\tAum Sri Sai Ram\n')
    print('FER with noisy annotations on FEC annotated using Self-Cure-network\n')
    print(args)
    print('\n\nNoise level:', args.noise_file)
  
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               drop_last=True,
                                               shuffle = True,  
                                               pin_memory = True) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)                                    
    # Define models
    print('building model...')
    model= noisyfer(args, train_dataset, device, input_channel, num_classes)
    
    epoch = 0
   
    for epoch in range(0, args.n_epoch):
        
        train_acc1, train_acc2, train_acc3, acc = model.train(train_loader, epoch)
        test_acc1, test_acc2, test_acc3, acc = model.evaluate(test_loader)
        print(  'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Avg Accuracy %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2,  test_acc3,acc))
        
                

if __name__ == '__main__':
    main()
