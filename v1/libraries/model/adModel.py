# -*- coding: utf-8 -*-
#%% IMPORTS
import time
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
from PIL import Image
import pandas as pd

import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms as Transforms
from torch.autograd import Variable

from libraries.model.ganomaly_network import GanomalyModel
from libraries.model.evaluate import evaluate
from libraries.utils import EarlyStopping, saveInfoGanomaly, addInfoGanomaly, LR_decay
from libraries.utils import Paths, ensure_folder, getNmeans
paths = Paths()

from libraries.dataset_package.dataset_manager import generatePatches
#%% CONSTANTS
GENERATOR = 'GENERATOR'
DISCRIMINATOR = 'DISCRIMINATOR'

device = torch.device('cuda:0')
#%%

def loadModel(filename):
        
    model_name = filename.split('_')[0] + '_' + filename.split('_')[1]
    path_file = paths.checkpoint_folder + model_name + '/' + filename
    
    return torch.load(path_file)

class AnomalyDetectionModel():
    
    def __init__(self, opt, optim_gen, optim_discr, optim_weights,
                 trainloader=None, validationloader=None, testloader=None):
        
        self.model              = GanomalyModel(opt)
#        optimizer_gen           = optim_gen(self.model.generator.parameters(), opt.lr_gen)
#        optimizer_discr         = optim_discr(self.model.discriminator.parameters(), opt.lr_discr)
#        optimizer_weights       = optim_gen(self.model.w_losses, opt.lr_gen)
        self.model.init_optim(optim_gen, optim_discr, optim_weights)
        self.trainloader        = trainloader
        self.validationloader   = validationloader
        self.testloader         = testloader
        self.opt                = opt
    
    def loadTrainloader(self, trainloader):
        self.trainloader = trainloader
        
    def loadValidationLoader(self, validationloader):
        self.validationloader = validationloader
    
    def get_loss(self):
        
        losses = OrderedDict([
                    
                    ('loss_gen', self.model.err_gen.item()),
                    ('loss_discr', self.model.err_discr.item()),
                    ('loss_gen_adv', self.model.err_gen_adv.item()),
                    ('loss_gen_con', self.model.err_gen_con.item()),
                    ('loss_gen_enc', self.model.err_gen_enc.item()),
                
                ])
    
        return losses

    def _trainOneEpoch(self):    

        self.model.train()
        train_loss = {}
        train_loss[GENERATOR] = []
        train_loss[DISCRIMINATOR] = []
        
        adv_loss = []
        con_loss = []
        enc_loss = []        
        
        n_iter = len(self.trainloader)
#        printing_freq = n_iter // self.opt.loss_per_epoch
#        print(n_iter)
        start = time.time()
        
#        for images, labels in tqdm(self.trainloader, leave=True, total=n_iter, desc='Training', file = sys.stdout):
        for images, labels in self.trainloader:

            x = torch.Tensor(images).cuda()
                
            # GENERATOR FORWARD
            x_prime, z, z_prime = self.model.forward_gen(x)

            # DISCRIMINATOR FORWARD
            pred_real, feat_real, pred_fake, feat_fake = self.model.forward_discr(x, x_prime) 
            
            
            # GENERATOR LOSS
            loss_gen, losses = self.model.loss_function_gen(x, x_prime, z, z_prime, feat_fake, feat_real, self.opt)
            if(self.epoch == 0):
                self.l0 = loss_gen.data
            
            # DISCRIMINATOR LOSS
            loss_discr = self.model.loss_function_discr(pred_real, pred_fake)
            
            # BACKWARDS
            self.model.optimize_gen(loss_gen, self.l0)
            self.model.optimize_discr(loss_discr)
            
            train_loss[GENERATOR].append(loss_gen.item()*images.size(0))
            train_loss[DISCRIMINATOR].append(loss_discr.item()*images.size(0))
            adv_loss.append(losses[0].item()*images.size(0))
            con_loss.append(losses[1].item()*images.size(0))
            enc_loss.append(losses[2].item()*images.size(0))
        
        spent_time = time.time() - start
        print('\n------------------------\n')
        print('> Loss weights')
        try:
            print('w_adv: {}'.format(self.model.w_adv[0]))
            print('w_con: {}'.format(self.model.w_con[0]))
            print('w_enc: {}'.format(self.model.w_enc[0]))
        except:
            print('w_adv: {}'.format(self.model.w_adv))
            print('w_con: {}'.format(self.model.w_con))
            print('w_enc: {}'.format(self.model.w_enc))
        print('----------------------------')
        return train_loss, [adv_loss, con_loss, enc_loss], spent_time
            
    def _validation(self):
        
        curr_epoch = 0
        steps = 0
        
        valid_loss = {}
        valid_loss[GENERATOR] = []
        valid_loss[DISCRIMINATOR] = []
        
        adv_loss = []
        con_loss = []
        enc_loss = []
        
        n_iter = len(self.validationloader)
        
        start = time.time()
        
        self.model.evaluate()
        with torch.no_grad():
            
#            for images, labels in tqdm(self.validationloader, leave=True, total=n_iter, desc='Validation', file = sys.stdout):
            for images, labels in self.validationloader:
                
                steps += 1
                curr_epoch += self.opt.batch_size
                
#                x = torch.Tensor(images).cuda()
                x = Variable(images).cuda()
                
                # GENERATOR FORWARD
                x_prime, z, z_prime = self.model.forward_gen(x)
                # DISCRIMINATOR FORWARD
                pred_real, feat_real, pred_fake, feat_fake = self.model.forward_discr(x, x_prime)    
                
                # GENERATOR LOSS
                loss_gen, losses = self.model.loss_function_gen(x, x_prime, z, z_prime, feat_fake, feat_real, self.opt)
                # DISCRIMINATOR LOSS
                loss_discr = self.model.loss_function_discr(pred_real, pred_fake)
                
                valid_loss[GENERATOR].append(loss_gen.item()*images.size(0))
                valid_loss[DISCRIMINATOR].append(loss_discr.item()*images.size(0))
                
                adv_loss.append(losses[0].item()*images.size(0))
                con_loss.append(losses[1].item()*images.size(0))
                enc_loss.append(losses[2].item()*images.size(0))
        
            spent_time = time.time() - start
                
        return valid_loss, [adv_loss, con_loss, enc_loss], spent_time
        
    def _test(self):
        
        start = time.time()
        
        if(self.testloader is None):
            test_loader = self.validationloader
        else:
            test_loader = self.testloader
        
        with torch.no_grad():
            
            i=0
            curr_epoch = 0
            times = []
#            n_iter = len(self.validationloader)

            anomaly_scores = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.long,    device=device)
            
            
#            for images, labels in tqdm(test_loader, leave=True, total=n_iter, desc='Test', file = sys.stdout):
            for images, labels in test_loader:
                
                curr_epoch += self.opt.batch_size
                
                time_in = time.time()
                
                x = torch.Tensor(images).cuda()
                tensor_labels = torch.Tensor(labels).cuda()                
                
                
                _, z, z_prime = self.model.forward_gen(x)
#                print('\n----Z----')
#                                       torch.Size([64, 100, 1, 1])
#                print(z.shape)
                
                # ANOMALY SCORE
                score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
                
#                print('Score: ', score)
                
                time_out = time.time()
                                
                
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_labels.reshape(score.size(0))

                times.append(time_out - time_in)

                i += 1
                
            
            # NORMALIZATION - Scale error vector between [0, 1]
            anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
            # auc, eer = roc(self.gt_labels, self.anomaly_scores)
#            print('\nAnom_Score Shape:')
#            print(anomaly_scores_norm.shape)
            
            auc, threshold_norm = evaluate(gt_labels, anomaly_scores_norm)
            
            _, threshold = evaluate(gt_labels, anomaly_scores)
            
#            print(np.where(gt_labels.cpu() == 1.0))
#            len(np.where(gt_labels.cpu() == 1.0))
            
            performance = dict({'AUC':auc,
                                'Threshold':threshold})
            
            eval_data = dict({'gt_labels':gt_labels,
                              'scores':anomaly_scores})

                
            spent_time = time.time() - start
                
            return performance, eval_data, spent_time
        
    def _training_step(self, epochs, save=True, lr_decay_value=None):
        
        plotUnit = 1
        start_epoch = self.epoch
        end_epochs = epochs
        
        self.es = EarlyStopping(self.opt)
        self.lrDecay = LR_decay(self.opt.lr_gen)
        
        for self.epoch in range(self.epoch, epochs):
            print('\n')
            print('Epoch {}/{}'.format(self.epoch+1, epochs))
            
            # TRAINING
            loss, losses, train_time = self._trainOneEpoch()
            self.train_loss[GENERATOR] = np.concatenate((self.train_loss[GENERATOR], getNmeans(loss[GENERATOR], plotUnit)))
            self.train_loss[DISCRIMINATOR] = np.concatenate((self.train_loss[DISCRIMINATOR], getNmeans(loss[DISCRIMINATOR], plotUnit)))
#            print(losses)
#            print(losses[0])
            
            self.train_adv_loss = np.concatenate((self.train_adv_loss, getNmeans(losses[0], plotUnit)))
            self.train_con_loss = np.concatenate((self.train_con_loss, getNmeans(losses[1], plotUnit)))
            self.train_enc_loss = np.concatenate((self.train_enc_loss, getNmeans(losses[2], plotUnit)))
            
#            train_loss[GENERATOR] = np.concatenate((train_loss[GENERATOR], loss[GENERATOR]))
#            print(len(train_loss[GENERATOR]))
            train_time = adjustTime(train_time)
            
            # VALIDATION
            loss, losses, val_time = self._validation()
            self.val_loss[GENERATOR] = np.concatenate((self.val_loss[GENERATOR], getNmeans(loss[GENERATOR], plotUnit)))
            self.val_loss[DISCRIMINATOR] = np.concatenate((self.val_loss[DISCRIMINATOR], getNmeans(loss[DISCRIMINATOR], plotUnit)))
            
            self.valid_adv_loss = np.concatenate((self.valid_adv_loss, getNmeans(losses[0], plotUnit)))
            self.valid_con_loss = np.concatenate((self.valid_con_loss, getNmeans(losses[1], plotUnit)))
            self.valid_enc_loss = np.concatenate((self.valid_enc_loss, getNmeans(losses[2], plotUnit)))
#            val_loss[GENERATOR] = np.concatenate((val_loss[GENERATOR], loss[GENERATOR]))
#            print(len(val_loss[GENERATOR]))
            val_time = adjustTime(val_time)

            valid_loss = self.val_loss['GENERATOR'][-1]
            
            performance, eval_data, spent_time = self._test()
            test_time = adjustTime(spent_time)
            
            self.auc, self.threshold = performance['AUC'], performance['Threshold']
            
            self.gt_labels, self.anomaly_scores = eval_data['gt_labels'], eval_data['scores']
            
            
            if(self.epoch % 5 == 0):
                self.plotting()
                self.evaluateRoc()
            
            if(self.auc > self.best_auc['AUC']):
                self.best_auc['AUC'] = self.auc
                self.best_auc['Loss'] = valid_loss
                self.best_auc['Threshold'] = self.threshold
                
#            self.visualizer.print_current_performance(result, best_auc)
                            
            print('\n')
            print('>- Training Loss:   {:.3f} in {} sec'.format(self.train_loss[GENERATOR][-1], train_time) )
            print('>- Validation Loss: {:.3f} in {} sec'.format(self.val_loss[GENERATOR][-1], val_time))
            print('>- AUC: {:.3f} \n>- Threshold: {:.3f} in {} sec'.format(self.auc, self.threshold, test_time))            
            
            
            saveCkp = self.es(valid_loss)
            if(saveCkp and save):
                self.saveCheckPoint(valid_loss)
                
            if(self.es.early_stop):
                print('-> Early stopping now')
                if(lr_decay_value):
                    print('-> LR factor decay: {}'.format(lr_decay_value))
                    self.lrDecay(lr_decay_value)
                    self.model.optimize_gen.params_groups[0]['lr'] = self.lrDecay.lr
                    print('New Learning rate for generator is {}'.format(self.lrDecay.lr))
                else:
                    break
        
        self.saveCheckPoint(valid_loss)
        self.plotting()
        self.saveInfo()
        self.evaluateRoc(folder_save=self.folder_save)
        
        print('\n> From {} to {} epochs'.format(start_epoch, end_epochs))
        
        return {'Validation_Loss' : valid_loss,
                'AUC': self.auc,
                'Threshold' : self.threshold}
        
        
    def train_model(self, epochs, save=True, lr_decay_value=None):
        
        print('-> Training model: ', self.opt.name)
        self.model.generator.train()
        self.model.discriminator.train()
        
        self.epoch = 0
        
        self.train_loss = {GENERATOR:[], DISCRIMINATOR:[]}
        
        self.val_loss = {GENERATOR:[], DISCRIMINATOR:[]}

        self.train_adv_loss = []
        self.train_con_loss = []
        self.train_enc_loss = []
        
        self.valid_adv_loss = []
        self.valid_con_loss = []
        self.valid_enc_loss = []
        
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
        
        self.best_auc = {'AUC':0, 'Loss':0, 'Threshold':0}
        
        performance = self._training_step(epochs, save, lr_decay_value)
        
        return performance
    
    
    def resumeTraining(self, epochs, save=True, lr_decay_value=None):
        
        performance = self._training_step(epochs, save)
        
        return performance
    
    def plotting(self, save=True):
        
        # PLOTTING LOSSES
        fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(5,1, figsize=(8,16))
        
        _subplot(ax1, self.train_adv_loss, self.valid_adv_loss, 'ADV loss')
        _subplot(ax2, self.train_con_loss, self.valid_con_loss, 'CON loss')
        _subplot(ax3, self.train_enc_loss, self.valid_enc_loss, 'ENC loss')
        _subplot(ax4, self.train_loss[GENERATOR], self.val_loss[GENERATOR], 'Generator')
        _subplot(ax5, self.train_loss[DISCRIMINATOR], self.val_loss[DISCRIMINATOR], 'Discriminator')
#        plt.legend()        
        
        if(save):
#            plt.savefig(self.folder_save + self.opt.name + '/'+ 'plot')
            plt.savefig(self.folder_save + 'plot')
       
        plt.show()
    
    def evaluateRoc(self, folder_save=None):
        if(folder_save is not None):
            folder_save = folder_save
        
        auc, _ = evaluate(self.gt_labels, self.anomaly_scores, plot=True, folder_save=folder_save)
        
        print('\n')
        print('AUC: {:.3f} \t Thres. : {:.3f} '.format(auc, self.threshold))
    
    def saveInfo(self):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        saveInfoGanomaly(self.opt, folder_save, self.auc)
    
    def addInfo(self, info):
        folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(folder_save)
        
        addInfoGanomaly(self.opt, folder_save, info)
    
    
    def predict(self, image, target=None, info=None, verbose=0):
#        image_tensor = torch.FloatTensor(image)
        image_transf = Transforms.ToTensor()(image)
        image_unsqueeze = image_transf.unsqueeze_(0)
        x_image = Variable(image_unsqueeze).cuda()
        
        transf = Transforms.Compose([Transforms.ToTensor(), Transforms.Normalize((0.5,),(0.5,))])
        
        imagePIL = Image.fromarray(image)
        x = transf(imagePIL)
        x = x.unsqueeze_(0)
        x = Variable(x).cuda()
#        image = image_tensor
        
#        plt.imshow(image)

        with torch.no_grad():
            _, z, z_prime = self.model.forward_gen(x)
            x_prime, _, _ = self.model.forward_gen(x_image)

        score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
#        print(score)
        anomaly_score = score
        
        output = x_prime.cpu().numpy()
        final_output = np.transpose(output[0], (2,1,0))
        
#        final_output = (output * 0.5) + 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1)        
        
        prediction = ['Anomalous Image', 1] if score >= self.threshold else ['Normal Image', 0]
        
        
        if(target is not None):
            real_outcome = 'Anomalous Image' if target == 1 else 'Normal Image'
            

        if(verbose):
            
            fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,13))
            results = '\n------------ RESULTS -------------\n' + \
                       'Threshold: {:.3f}\n'.format(self.threshold) + \
                       'Score: {:.3f}\n'.format(anomaly_score.item()) + \
                       'Real Outcome: {}\n'.format(real_outcome) + \
                       '---------------------------------\n\n' + \
                       'Original image --> {}'.format(prediction[0])
                       
            ax1.set_title(results)
            ax1.imshow(image)
            ax2.set_title('Reconstructed image')
            ax2.imshow(final_output)
            
            print('')
            print('\n------------ RESULTS -------------')
            print('Threshold: \t{:.3f}'.format(self.threshold))
            print('Score: \t\t{:.3f}'.format(anomaly_score.item()))
            print('From \t\t{}'.format(real_outcome))
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
            
        return prediction, anomaly_score.item(), self.threshold
        
    
    def predictImage(self, dataTest, folder_save=None, N=10):
        
        i = 0
        
        with torch.no_grad():
            anomaly_scores = torch.zeros(size=(len(dataTest.dataset),), dtype=torch.float32, device=device)
            gt_labels = torch.zeros(size=(len(dataTest.dataset),), dtype=torch.long, device=device)
                
            start = time.time()
            for patches, labels in tqdm(dataTest, total=len(dataTest)):
                
                x = torch.Tensor(patches).cuda()
                tensor_label = torch.Tensor(labels).cuda()
                
                
                x_prime, z, z_prime = self.model.forward_gen(x)
                
                # ANOMALY SCORE
                score = torch.mean(torch.pow((z-z_prime), 2), dim=1)
                    
                anomaly_scores[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = score.reshape(score.size(0))
                gt_labels[i*self.opt.batch_size : i*self.opt.batch_size + score.size(0)] = tensor_label.reshape(score.size(0))
        
                i += 1
    
            anomaly_scores_norm = (anomaly_scores - torch.min(anomaly_scores)) / (torch.max(anomaly_scores) - torch.min(anomaly_scores))
            auc, threshold_auc = evaluate(gt_labels, anomaly_scores_norm, plot=True, folder_save=folder_save)
            
            avg_prec = evaluate(gt_labels, anomaly_scores_norm, metric='prec_rec_curve', plot=True, folder_save=folder_save)
#            return evaluate(gt_labels, anomaly_scores_norm, metric='prec_rec_curve', plot=True)
            end = time.time()
            
            performance = {'AUC': auc,
                           'Thr': threshold_auc,
                           'Avg_prec': avg_prec}
            
            print('Prediction time: {}'.format(adjustTime(end-start)))
            
            gt_labels = dataTest.dataset.targets
            pred_labels = computeAnomalyDetection(anomaly_scores_norm, threshold_auc)
            # PRECISION
            precision = evaluate(gt_labels, pred_labels, metric='precision')
            
            
            pred_patches = generatePatches(dataTest.dataset.data, pred_labels)
            
            
            print('Precision: {}'.format(precision))
            samples = getSamples(patches, x_prime, labels, anomaly_scores_norm, N=10)
            
        return pred_patches, samples, performance
            
    def saveImages(self, dataloader):
        
        reals, fakes, fixed = self.get_images(dataloader)
        self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
        if self.opt.display:
            self.visualizer.display_current_images(reals, fakes, fixed)
    
        
    def saveCheckPoint(self, valid_loss):
        self.folder_save = paths.checkpoint_folder + self.opt.name + '/'
        ensure_folder(self.folder_save)
        
        path_file = '{0}/{1}_lr:{2}|Epoch:{3}|Auc:{4:.3f}|Loss:{5:.4f}.pth.tar'.format(self.folder_save,
                                                                             self.opt.name,
                                                                             self.opt.lr_gen,
                                                                             self.epoch,
                                                                             self.auc,
                                                                             valid_loss)
        
        torch.save(self, path_file)
#        print('3.Model: ', self.model)
        
    def tuneLearningRate(self, inf_bound_gen, sup_bound_gen, inf_bound_discr, sup_bound_discr):
        
        max_count = 10
        self.result = []
        
        for count in range(max_count):
            
            print('Model n.', count)
            
            self.opt.epochs = 4
            self.model.optimizer_gen.lr = 10**np.random.uniform(sup_bound_gen, inf_bound_gen)
            self.model.optimizer_discr.lr = 10**np.random.uniform(sup_bound_discr, inf_bound_discr)
            loss = self.train_model(save=False)
            
            lr_gen_label = 'Gen_Lr:\t{}\n'.format(self.model.optimizer_gen.lr)
            lr_discr_label = 'Discr_Lr:\t{}\n'.format(self.model.optimizer_discr.lr)
            loss_label = 'Loss:\t{}\n\n'.format(loss)
            
            result_label = 'Results \n' + lr_gen_label + lr_discr_label + loss_label
            
            self.result.append(result_label)
        
        return self.result

        

def computeAnomalyDetection(scores, threshold):   

    pred_labels = []
    
    for i in range(0, len(scores)):
        if(scores[i] < threshold):
            pred_labels.append(np.float64(0))
        else:
            pred_labels.append(np.float64(1))
            
    return pred_labels
    
def outputSample(sample, threshold, info=None, folder_save=None):
    
    original = sample['originals']
    image_rec = sample['images']
    label = 'Anomaly' if(sample['labels']) else 'Normal'
    score = sample['scores']
    print(score)
    
    result = 'Normal patch' if score < threshold else 'Anomalous patch'
    
    fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,13))
    results = '\n------------ RESULTS -------------\n' + \
               'Threshold: {:.3f}\n'.format(threshold) + \
               'Score: {:.3f}\n'.format(score) + \
               'Real Outcome: {}\n'.format(label) + \
               '---------------------------------\n\n' + \
               'Original image --> {}'.format(result)
               
    ax1.set_title(results)
    ax1.imshow(original)
    ax1.grid(False)
    ax2.set_title('Reconstructed image')
    ax2.imshow(image_rec)
    ax2.grid(False)
    
    print('')
    print('\n------------ RESULTS -------------')
    print('Threshold: \t{:.3f}'.format(threshold))
    print('Score: \t\t{:.3f}'.format(score))
    print('From \t\t{}'.format(label))
    print('')
    print('Original image --> ', result)
    print('----------------------------------')
    
    if(info is not None and folder_save is not None):
        print('..Saving..')
        if(result == 'Normal patch'):    
            plt.savefig(folder_save + 'Normal_{}'.format(info))
        elif(result == 'Anomalous patch'):
            plt.savefig(folder_save + 'Anomaly_{}'.format(info))
        else:
            raise Exception('Wrong Predicion')  
    
    
def _subplot(ax, train, val, title):
    ax.set_title(title)
    ax.plot(train, color='r', label='Training')
    ax.plot(val, color='b', label='Validation')
    ax.legend()
        
def adjustTime(sample_time):
    
    minutes = sample_time / 60
    seconds = (minutes - (sample_time // 60)) * 60
    
    return '{} min {} sec'.format(int(minutes), int(seconds))
      
def getSamples(originals, patches, labels, scores, N=10):
    
    patch_image = []
    patch_recon = []
    patch_labels = []
    patch_scores = []
    
    for i in range(0, N):
        x = originals[i]
        x_prime = patches[i]
        label = labels[i]
        
        image = x.cpu().numpy()
        output = x_prime.cpu().numpy()
#        print(output.shape)
        image_input = np.transpose(image, (2,1,0))
        final_output = np.transpose(output, (2,1,0))
#        print(final_output.shape)
        
        image_input = (image_input * 0.5) + 0.5
        image_input = np.flip(image_input, 1)
        image_input = np.rot90(image_input, 1) 
        
        final_output = (final_output * 0.5) + 0.5
        final_output = np.flip(final_output, 1)
        final_output = np.rot90(final_output, 1) 
        
        start = len(scores) - len(patches)
#        print(start)
        score = scores[start + i]
        
        patch_image.append(image_input)
        patch_recon.append(final_output)
        patch_labels.append(label.cpu())
        patch_scores.append(score)
        
        
    samples = {'originals': patch_image,
               'images': patch_recon,
               'labels': patch_labels,
               'scores': patch_scores}
    
    return pd.DataFrame(samples)
        
        
        
        
        
        
        
        
        