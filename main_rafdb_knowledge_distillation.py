# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram
Implementation of CCT: Consensual Collaborative Training and Knowledge distillation for Facial Expression Recognition with Noisy Annotations         
Authors: Darshan Gera and Dr. S. Balasubramanian, SSSIHL
Date: 20-03-2021
Email: darshangera@sssihl.edu.in
'''
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse, sys
import datetime

from PIL import Image
import torch.nn.functional as F

import pandas as pd
import image_utils
import cv2
import argparse,random
from model.cnn import resModel

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--raf_path', type=str, default='../data/RAFDB', help='Raf-DB dataset path.')
    
parser.add_argument('--pretrained', type=str, default='pretrained/res18_naive.pth_MSceleb.tar',
                        help='Pretrained weights')

parser.add_argument('--resume', type=str, default='checkpoints/rafdb/epoch_73_noise_train_label.txt_acc_90.54.pth', help='Use FEC trained models')                     
                        
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.4)

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')

parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
                    
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
                    
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='rafdb')

parser.add_argument('--noise_file', type=str, help='EmoLabel/', default='EmoLabel/train_label.txt')

parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,  metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--beta', type=float, default=.65,
                    help='..based on ')

parser.add_argument('--co_lambda_max', type=float, default=.9,
                    help='..based on ')

parser.add_argument('--n_epoch', type=int, default=200)

parser.add_argument('--num_classes', type=int, default=7)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=30)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--batch_size', type=int, default=64, help='batch_size')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--co_lambda', type=float, default=0.1)

parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--num_models', type=int, default=3)

parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')

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


                         
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False, ):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/train_label.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join(self.raf_path, args.noise_file), sep=' ', header=None)
        
        df_test = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/test_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset_train_noisy = df_train_noisy[df_train_noisy[NAME_COLUMN].str.startswith('train')]
            dataset_train_clean = df_train_clean[df_train_clean[NAME_COLUMN].str.startswith('train')]
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.label = self.noisy_label
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
        else:             
            dataset = df_test[df_test[NAME_COLUMN].str.startswith('test')]
            self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral            
            file_names = dataset.iloc[:, NAME_COLUMN].values
        
        self.new_label = [] 
        
        for label in self.label:
            self.new_label.append(self.change_emotion_label_same_as_affectnet(label))
            
        self.label = self.new_label
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
        
        
    def change_emotion_label_same_as_affectnet(self, emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 3
        elif emo_to_return == 1:
            emo_to_return = 4
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 1
        elif emo_to_return == 4:
            emo_to_return = 2
        elif emo_to_return == 5:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 0

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
            image =  self.transform(image)
            #image2 =  self.transform2(image)
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
        if dataset_type is RafDataSet:
            return dataset.label[idx]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples  
       
if  args.dataset == 'rafdb':   
    input_channel = 3
    num_classes = args.num_classes
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
        ])#transforms.RandomErasing(scale=(0.02,0.25))

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
                                 
    train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    #print('Train set size:', train_dataset.__len__())                                                                            
    test_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)    
    #print('Validation set size:', test_dataset.__len__())
    
                            

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate



def main():
    # Data Loader (Input Pipeline)
    print('\n\t\t\tAum Sri Sai Ram\n\n\n')
        
    #print('\n\nNoise level:', args.noise_file)   
    
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
    #print('building models'+ str(args.num_models)+' with noise file: '+args.noise_file)
    model1 = resModel(args)     
    model2 = resModel(args)
    model3 = resModel(args)
    
    studentmodel = resModel(args)
    
    model1.to(device)
    model2.to(device)
    model3.to(device)
    studentmodel.to(device)
    
    best_prec1 = 0.
    
    pretrained = torch.load(args.resume)
    pretrained_state_dict1 = pretrained['model_1']   
    pretrained_state_dict2 = pretrained['model_2']    
    pretrained_state_dict3 = pretrained['model_3']  
    model1.load_state_dict(pretrained_state_dict1) 
    model2.load_state_dict(pretrained_state_dict2)
    model3.load_state_dict(pretrained_state_dict3)
           
    print('All teacher models loaded from ',args.resume)
    
    optimizer = torch.optim.Adamax(studentmodel.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 100])
    
    for epoch in range(0, args.n_epoch):
        # train for one epoch        
        train(train_loader, model1, model2, model3, studentmodel,optimizer, scheduler, epoch) 
        prec1 = validate(test_loader , studentmodel,  epoch)
        print("Epoch: {}   Test Acc: {}".format(epoch, prec1))
        is_best = prec1.item() > best_prec1
        best_prec1 = max(prec1.to(device).item(), best_prec1)
        
        if is_best:
           torch.save(studentmodel,'checkpoints/rafdb/student_ensemble_rafdb_T_2.pth.tar')
        
    
    

def kd_loss(teacherlogits, studentlogits, labels, T = 2, lambda_= 0.5):
    with torch.no_grad():
        outputTeacher = (1.0 / T) * teacherlogits 
        outputTeacher = F.softmax(outputTeacher, dim =1)
    cost_1 = F.cross_entropy(studentlogits, labels)
    pred = F.softmax(studentlogits, dim = 1)
    logp = F.log_softmax(studentlogits/T, dim=1)
    cost_2 = -torch.mean(torch.sum(outputTeacher * logp, dim=1))
    cost = ((1.0 - lambda_) * cost_1 + lambda_ * cost_2)
    return cost      

def train(train_loader, model1, model2, model3, studentmodel, optimizer, scheduler, epoch):
    print('Training ...')
    
    model1.eval() 
    model2.eval() 
    model3.eval() 
    train_correct = 0
    train_total = 0
    overall_loss = AverageMeter()
    top1 = AverageMeter()
    studentmodel.train()
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits1 = model1(images)
            logits2 = model2(images)
            logits3 = model3(images)
        logits = studentmodel(images)
        prec = accuracy(logits, labels, topk=(1,))
        
                 
        loss = kd_loss(logits1, logits, labels) + kd_loss(logits2, logits, labels) + kd_loss(logits3, logits, labels)
        
        overall_loss.update(loss.item(), images.size(0)) 
        top1.update(prec[0], images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
            
        optimizer.step()
            
        if i % args.print_freq == 0:
            print('Training Epoch: [{0}][{1}/{2}]\t'                  
                  'overall_loss ({overall_loss.avg:.3f})\t' 
                  'Prec1  ({top1.avg:.3f}) \t'.format(
                   epoch, i, len(train_loader),                  
                   overall_loss=overall_loss,  top1=top1))    
    scheduler.step()
        
    
def validate(val_loader,  basemodel, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
        
    region_prec = []
    mode =  'Testing'

    # switch to evaluate mode
    basemodel.eval()
   
   

    with torch.no_grad():         
        for i, (input1, target,_) in enumerate(val_loader):                    
            input1 = input1.to(device)
            
            target = target.to(device)
            region_preds = basemodel(input1)
            
            avg_prec = accuracy(region_preds,target,topk=(1,))       
            top1.update(avg_prec[0], input1.size(0))
        print('\n{0} [{1}/{2}]\t'                  
                  'Prec@1  ({top1.avg})\t'
                  .format(mode, i, len(val_loader),   top1=top1))

       
      
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
  
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res        

if __name__ == '__main__':
    main()
    
