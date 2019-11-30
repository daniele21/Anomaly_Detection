# -*- coding: utf-8 -*-

#%% IMPORTS
import time
from collections import OrderedDict
from network import Encoder, Decoder, weights_init
from loss import contextual_loss
from evaluate import evaluate
from utils import Paths
from utils import Checkpoint, saveInfoAE, addInfoAE
from utils import ensure_folder
from utils import EarlyStopping
#import utils as ut

import torch
import torch.nn as nn
from torchvision import transforms as Transforms

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from tqdm import tqdm
import sys
from torch.autograd import Variable
#import sys

paths = Paths()
#%% NETWORK

device = torch.device('cuda:0')

class Autoencoder(nn.Module):
    
    def __init__(self, opt):
        super().__init__()
        
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, x):
        
        z = self.encoder(x)
        x_prime = self.decoder(z)
        
        return x_prime

def generateAutoencoder(opt, cuda=True):
    
    if(cuda):
        return Autoencoder(opt).cuda()        
    else:
        return Autoencoder(opt).cpu()

def loadAEmodel(path_file):
    return torch.load(path_file)

class AutoencoderModel():
    
    def __init__(self, opt, optimizer, trainloader=None, validationloader=None):
        
        self.model      = generateAutoencoder(opt)
        self.optimizer  = optimizer(self.model.parameters(), opt.lr)
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.opt = opt
    
#    def loadCheckPoint(self, path_file):
#        self.model = torch.load(path_file)
#        self.optimizer = checkpoint.optimizer
    
    def loadTrainloader(self, trainloader):
        self.trainloader = trainloader
        
    def loadValidationLoader(self, validationloader):
        self.validationloader = validationloader
     
    def _trainOneEpoch(self, loss_function):
    
        trainLoss = []
        
        self.model.train()
        
        start = time.time()
        
        n = len(self.trainloader)
        for image, label in tqdm(self.trainloader, leave=True, position=0, total=n, desc='Training\t',
                                 file = sys.stdout):
            
            img_var = Variable(image).cuda()
        #        print(img_var.shape)
            
            # FORWARD
            output = self.model(img_var)        
#            output.shape                    
            loss = loss_function(output, img_var)
                    
            # BACKWARD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # VISUALIZATION
            trainLoss.append(loss.item())
                  
        time_spent = time.time() - start
            
        return trainLoss, time_spent
        
    def _validation(self, loss_function):
    
        validationLoss = []
    
        self.model.eval()      
        
        start = time.time()
        
        with torch.no_grad():
            count=0
            self.curr_steps += self.batch_size
            
            n = len(self.validationloader)
            
            for image, label in tqdm(self.validationloader, leave=True, position=0, total=n,
                                     file = sys.stdout, desc='Validation\t'):
                
                count += 1
                
                img_var = Variable(image).cuda()
                img_var.shape
                
                # FORWARD
                output = self.model(img_var)        
                output.shape
                        
                loss = loss_function(output, img_var)
                
                # VISUALIZATION
                
                validationLoss.append(loss.item())
            
            time_spent = time.time() - start
            
        return validationLoss, time_spent
    
    def _test(self):
        
        start = time.time()
        
        with torch.no_grad():
            
            i=0
            curr_epoch = 0
            times = []
            n_iter = len(self.validationloader)

            anomaly_scores = torch.zeros(size=(len(self.validationloader.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(self.validationloader.dataset),), dtype=torch.long,    device=device)
            
            for images, labels in tqdm(self.validationloader, leave=True, total=n_iter, desc='Test\t\t', file = sys.stdout):
                
                curr_epoch += self.opt.batch_size
                
                time_in = time.time()
                
                x = torch.Tensor(images).cuda()
                tensor_labels = torch.Tensor(labels).cuda()                
                
                
                x_prime = self.model(x)
#                print('\n----Z----')
#                                       torch.Size([64, 100, 1, 1])
#                print(z.shape)
                
                # ANOMALY SCORE
                x = x.reshape(x.size(0), x.size(1)*x.size(2)*x.size(3))
                x_prime = x_prime.reshape(x_prime.size(0), x_prime.size(1)*x_prime.size(2)*x_prime.size(3))
                
                score = torch.mean(torch.pow((x-x_prime), 2), dim=1)
                
                time_out = time.time()
                
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))
              
                times.append(time_out - time_in)

                i += 1
            
            # Measure inference time
            times = np.array(times)
            times = np.mean(times[:100] * 1000)
            
            # NORMALIZATION - Scale error vector between [0, 1]
            anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
            auc, threshold_norm = evaluate(gt_labels, anomaly_scores_norm)
#            print(threshold_norm)
            _, threshold = evaluate(gt_labels, anomaly_scores)
            
            performance = dict({'AUC':auc,
                                'Threshold':threshold})
            
            eval_data = dict({'gt_labels':gt_labels,
                              'scores':anomaly_scores})
    
#            performance = OrderedDict([('Avg Run Time (ms/batch)', times), ('AUC', auc)])                
            
            spent_time = time.time() - start
                
            return performance, eval_data,  spent_time
        
    
    def train_autoencoder(self, save=True):
        
        self.curr_steps = 0
        self.printing_freq = self.opt.printing_freq
        self.batch_size = self.opt.batch_size
        
        es = EarlyStopping(self.opt)
        
        loss_function = contextual_loss()
        
        self.loss = {}
        self.loss['train'] = []
        self.loss['validation'] = []
        
        self.avg_loss = {}
        self.avg_loss['train'] = []
        self.avg_loss['validation'] = []
        
        for self.epoch in range(self.opt.epochs):
            print('\n')
            print('Epoch {}/{}'.format(self.epoch+1, self.opt.epochs))
            
            # TRAINING
            train_losses, train_time = self._trainOneEpoch(loss_function)
            train_loss = np.average(train_losses)
#            self.loss['train'].append(train_losses) 
            self.avg_loss['train'].append(train_loss)
            
            # VALIDATION
            val_losses, val_time     = self._validation(loss_function)
            val_loss = np.average(val_losses)
#            self.loss['validation'].append(val_losses)
            self.avg_loss['validation'].append(val_loss)
            
            # TEST
            performance, eval_data, spent_time = self._test()
            
            self.auc, self.threshold = performance['AUC'], performance['Threshold']
            self.gt_labels, self.anomaly_scores = eval_data['gt_labels'], eval_data['scores']            
            
            if(self.epoch % self.opt.printing_freq == 0):
                self.plotting(save=False)
                self.evaluateRoc()
            
            print('\n')
            print('>- Training Loss:   {:.4f} in {:.2f} sec'.format(train_loss, train_time) )
            print('>- Validation Loss: {:.4f} in {:.2f} sec'.format(val_loss, val_time))
            
            # SAVING CHECKPOINT
            self.folder_save = paths.checkpoint_folder
            ensure_folder(self.folder_save)
            
#            filename = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.3f}.pth.tar'.format(folder_save,
#                                                                                self.opt.name,
#                                                                                self.opt.lr,
#                                                                                epoch, train_loss)
            
            saveCkp = es(val_loss)
            if(saveCkp and save):
                self.saveCheckPoint(val_loss)
            
            if(es.early_stop):
                print('-> Early stopping now')
#                self.plotting()
                break
        
        self.saveCheckPoint(val_loss)
        self.plotting()
        self.evaluateRoc(folder_save=self.folder_save)
        self.saveInfo()
        
        return val_loss
    
    def evaluateRoc(self, folder_save=None):
        if(folder_save is not None):
            folder_save = folder_save
        
        auc, _ = evaluate(self.gt_labels, self.anomaly_scores, plot=True, folder_save=folder_save)
        
        print('\n')
        print('AUC: {:.3f} \t Thres. : {:.3f} '.format(auc, self.threshold))
    
    def saveInfo(self):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        saveInfoAE(self.opt, folder_save, self.auc)
    
    def addInfo(self, info):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        addInfoAE(self.opt, folder_save, info)
        
    
    def saveCheckPoint(self, val_loss):
        
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
        
        path_file = '{0}/{1}_lr:{2}|Epoch:{3}|Loss:{4:.4f}.pth.tar'.format(self.folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr,
                                                                             self.epoch,
                                                                             val_loss)
        
        torch.save(self, path_file)
     
    def plotting(self, save=True):
                
        # PLOTTING LOSSES
        plt.figure()
        plt.title('Average Loss')
        plt.plot(self.avg_loss['train'], color='r', label='train')
        plt.plot(self.avg_loss['validation'], color='b', label='validation')
        plt.legend()
        
        
        if(save):
#            plt.savefig(self.folder_save + self.opt.name + '/'+ 'plot')
            plt.savefig(self.folder_save + 'plot')
       
        plt.show()
        
    def plotAnomScores(self):
        anom_scores = []
        for elem in self.anomaly_scores:
            anom_scores.append(elem.item())
        
        plt.hist(anom_scores)
        
    def predict(self, image, info=None, verbose=1):
        
        if(self.opt.in_channels == 1):
            cmap = 'gray'
            grayMode = True
            
        elif(self.opt.in_channels == 3):
            cmap = None
            grayMode = False
        
        
        imagePIL = Image.fromarray(image)
        transf = Transforms.Compose([Transforms.Grayscale(num_output_channels=self.opt.in_channels),
                                     Transforms.ToTensor()])
        
#        image_tensor = torch.FloatTensor(image)
#        image_tensor = Transforms.ToTensor()(image)
        image_tensor = transf(imagePIL)
        image_tensor = image_tensor.unsqueeze_(0)
        x = Variable(image_tensor).cuda()

#        plt.imshow(image)

        with torch.no_grad():
            x_prime = self.model(x)
            
        # SCORE
        
        #torch.Size([1, 3, 32, 32])
        
        x_score = x.reshape(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x_prime_score = x_prime.reshape(x_prime.size(0), x_prime.size(1)*x_prime.size(2)*x_prime.size(3))
        
        score = torch.mean(torch.pow((x_score-x_prime_score), 2), dim=1)
        
        # NORM SCORE
#        anomaly_score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        anomaly_score = score
        
        output = x_prime.cpu().numpy()
        output = np.transpose(output[0], (0,1,2))

        if(grayMode):
            output = output[0].reshape(self.opt.img_size, self.opt.img_size)

        final_output = output
        final_output = (output * 0.5) / 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1)
        
#        plt.imshow(image, cmap='gray')
#        plt.show()
#        plt.imshow(image)
        
        prediction = ['Anomalous Image', 1] if score >= self.threshold else ['Normal Image', 0]
#        print(prediction)
        if(verbose):
            
            fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,10))
            results = '------------ RESULTS -------------\n' + \
                       'Threshold: {:.3f}\n'.format(self.threshold) + \
                       'Score: {:.3f}\n'.format(anomaly_score.item()) + \
                       '---------------------------------\n\n' + \
                       'Original image --> {}'.format(prediction[0])
                       
            ax1.set_title(results)
            ax1.imshow(image, cmap=cmap)
            ax2.set_title('Reconstructed image')
            ax2.imshow(final_output, cmap=cmap)
            
            print('')
            print('\n------------ RESULTS -------------')
            print('Threshold: \t{:.3f}'.format(self.threshold))
            print('Score: \t\t{:.3f}'.format(anomaly_score.item()))
            print('')
            print('Original image --> ', prediction[0])
            print('----------------------------------')
            
            if(info is not None):
                print('..Saving..')
                if(prediction[0] == 'Normal Image'):    
                    plt.savefig(self.folder_save + 'Normal_{}'.format(info))
                elif(prediction[0] == 'Anomalous Image'):
                    plt.savefig(self.folder_save + 'Anomaly_{}'.format(info))
                else:
                    raise Exception('Wrong Predicion')
            
        return prediction
        
    def tuneLearningRate(self, inf_bound, sup_bound):
        
        max_count = 10
        result = []
        
        for count in range(max_count):
            
            print('Model n.', count)
            
            self.opt.epochs = 5
            self.optimizer.lr = 10**np.random.uniform(sup_bound, inf_bound)
            loss = self.train_autoencoder(save=False)
            
            result.append(str(self.optimizer.lr) + ' : ' + str(loss))
        
        return result
                       
        
        
        
        











