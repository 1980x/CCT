# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram
Implementation of CCT: Consensual Collaborative Training for Facial Expression Recognition with Noisy Annotations                
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
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')

parser.add_argument('--fplus_path', type=str, default='../data/FERPLUS/Dataset', help='Fplus dataset path.')
    
parser.add_argument('--pretrained', type=str, default='pretrained/res18_naive.pth_MSceleb.tar', help='Pretrained weights')
                        
parser.add_argument('--resume', type=str, default='', help='Use FEC trained models')                  

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--dataset', type=str, help='rafdb, ferplus, affectnet', default='fplus')

parser.add_argument('--beta', type=float, default= 0.65,    help='..based on ')
                    
parser.add_argument('--noise_file', type=str, help='NoisyLabels', default='NoisyLabels/0.4noise_ferplus_trainvalid_list.txt')

parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)                    

parser.add_argument('--co_lambda_max', type=float, default=.9,   help='..based on ')

parser.add_argument('--n_epoch', type=int, default=40)

parser.add_argument('--num_models', type=int, default=2)

parser.add_argument('--num_classes', type=int, default=7)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=50)

parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--co_lambda', type=float, default=0.1)

parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--model_type', type=str, help='[res]', default='res')

parser.add_argument('--save_model', type=str, help='save model?', default="False")

parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')


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

                            
                         
class FplusDataSet(data.Dataset):
    def __init__(self, fplus_path, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.fplus_path = fplus_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join(self.fplus_path, 'NoisyLabels/ferplus_trainvalid_list.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join(self.fplus_path, args.noise_file), sep=' ', header=None)
       
        os.path.join(self.fplus_path, args.noise_file)
        
        df_test = pd.read_csv(os.path.join(self.fplus_path, 'NoisyLabels/ferplus_test.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset_train_noisy = df_train_noisy
            dataset_train_clean = df_train_clean
            
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values 
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values            
             
            self.label = self.noisy_label
            
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
            print(dataset_train_noisy.groupby([1]).size())
        else: 
           
            dataset = df_test
            self.label = dataset.iloc[:, LABEL_COLUMN].values             
            file_names = dataset.iloc[:, NAME_COLUMN].values
            print(dataset.groupby([1]).size())
        self.new_label = [] 
        
        for label in self.label:
            self.new_label.append(self.change_emotion_label_same_as_affectnet(label))
            
        self.label = self.new_label
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f+'.png'
            if phase == 'train':            
               path = os.path.join(self.fplus_path, 'Images/FER2013TrainValid', f)
            else:
               path = os.path.join(self.fplus_path, 'Images/FER2013Test', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def change_emotion_label_same_as_affectnet(self, emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """    
        if emo_to_return == 2:
            emo_to_return = 3
        elif emo_to_return == 3:
            emo_to_return = 2
        elif emo_to_return == 4:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 4

        return emo_to_return 
           
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
                            
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        #print(self.indices)    
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        #print(self.num_samples)              
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            #print(label)
            # spdb.set_trace()
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        #print(dataset_type)
        #pdb.set_trace()
        if dataset_type is FplusDataSet:
            return dataset.label[idx]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
       
if  args.dataset == 'fplus':   
    input_channel = 3
    num_classes = 8
    init_epoch = 5
    args.epoch_decay_start = 100
    # args.n_epoch = 200
    filter_outlier = False
    args.model_type = "res"
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                        transforms.RandomAffine(degrees=0, translate=(.1, .1),
                                               scale=(1.0, 1.25),
                                               resample=Image.BILINEAR)],p=0.5),
        
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))
        ])
            
    
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
                                 
    train_dataset = FplusDataSet(args.fplus_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('\n Train set size:', train_dataset.__len__())                                                                            
    test_dataset = FplusDataSet(args.fplus_path, phase = 'test', transform = data_transforms_val)    
    print('\n Validation set size:', test_dataset.__len__())
    
    
def main():
   
    print('\n\t\t\tAum Sri Sai Ram\n')
    print('FER with noisy annotations on FERPlus\n')
    print(args)
    
    
    print('\n\nNoise level:', args.noise_file)                       

    print('\n \n')
    print(args)
   
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
    
    # training
    for epoch in range(0, args.n_epoch):
        # training
    for epoch in range(0, args.n_epoch):
        
        train_acc1, train_acc2, train_acc3, acc = model.train(train_loader, epoch)
        test_acc1, test_acc2, test_acc3, acc = model.evaluate(test_loader)
        print(  'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Model3 %.4f %% Avg Accuracy %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2,  test_acc3,acc))
        
                
        # save results
        if acc >= 90.0: 
           model.save_model(epoch, acc, args.noise_file.split('/')[-1])
if __name__ == '__main__':
    main()
