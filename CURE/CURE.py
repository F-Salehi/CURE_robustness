import torch
import copy
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt
from utils import pgd
import torchvision
import os
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from utils import progress_bar
from torch.distributions import uniform

    
class CURE():
    def __init__(self, net, trainloader, testloader, device='cuda', lambda_ = 4, precentage = 1
                 ):
        '''
        CURE Class
        precentage: the precentage of the minimum eigens chosen
        '''
        if not torch.cuda.is_available() and device=='cuda':
            raise ValueError("cuda is not available")

        self.net = net.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.lambda_ = lambda_
        self.precentage = precentage

        self.train_loss, self.train_acc, self.train_curv = [], [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean, self.test_curv = [], [], [], []
        self.train_loss_best, self.train_acc_best, self.train_curv_best = 0, 0, 0
        self.test_loss_best, self.test_acc_adv_best, self.test_acc_clean_best, self.test_curv_best = 0, 0, 0, 0
    

    def set_optimizer(self, optim_alg='Adam', args={'lr':1e-4}, scheduler=None, args_scheduler={}):
        '''
        setting the optimizer of the network
        
        Arguments:
        
        optim_alg : string
            NAme of the optimizer
        args: dict
            Parameter of the optimizer
        scheduler: optim.lr_scheduler
            Learning rate scheduler
        args_scheduler : dict
            Parameters of the scheduler
        '''
        self.optimizer = getattr(optim, optim_alg)(self.net.parameters(), **args)
        if not scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10**6, gamma=1)
        else:
            self.scheduler = getattr(optim.lr_scheduler, scheduler)(self.optimizer, **args_scheduler)
        
            
    def train(self, h = [3], epochs = 15):
        '''
        Training the network

        Arguemnets:

        h : list
            Different h for different epochs of training
        epochs : int
            Number of epochs
        '''
        if len(h)>epochs:
            raise ValueError('Length of h should be less than number of epochs')

        h_all = epochs * [1.0]
        h_all[:len(h)] = list(h[:])
        h_all[len(h):] = h[-1]

        for epoch, h_tmp in enumerate(h_all):
            self._train(epoch, h=h_tmp)
            self.test(epoch, h=h_tmp)
            self.scheduler.step()
        
    def _train(self, epoch, h):
        '''
        training the model
        '''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        curv, curvature, norm_grad_sum  = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)
            outputs = self.net.train()(inputs)
            
            regularizer, grad_norm = self.regularizer(inputs, targets, h=h) 
                    
            curvature += regularizer.item()
            neg_log_likelihood = self.criterion(outputs, targets) 
            loss = neg_log_likelihood + regularizer
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            outcome = predicted.data == targets
            num_correct += outcome.sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | curvature: %.3f '% \
             (train_loss/(batch_idx+1), 100.*num_correct/total, num_correct, total, curvature/(batch_idx+1)  ))
            
        self.train_loss.append(train_loss/(batch_idx+1))
        self.train_acc.append(100.*num_correct/total)
        self.train_curv.append(curvature/(batch_idx+1))
                
    def test(self, epoch, h, num_pgd_steps=20):  
        test_loss, adv_acc, total, curvature, clean_acc, grad_sum = 0, 0, 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.testloader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net.eval()(inputs)
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(targets).sum().item()
            total += targets.size(0)

            inputs_pert = inputs + 0.
            eps = 5./255.*8
            r = pgd(inputs, self.net.eval(), epsilon=[eps], targets=targets, step_size=0.04,
                    num_steps=num_pgd_steps, epsil=eps)

            inputs_pert = inputs_pert + eps * torch.Tensor(r).to(self.device)
            outputs = self.net(inputs_pert)
            probs, predicted = outputs.max(1)
            adv_acc += predicted.eq(targets).sum().item()
            cur, norm_grad = self.regularizer(inputs, targets, h=h)
            grad_sum += norm_grad.item()
            curvature += cur.item()
            test_loss += cur.item()

        print(f'epoch = {epoch}, adv_acc = {100.*adv_acc/total}, clean_acc = {100.*clean_acc/total}, loss = {test_loss/num_batches}', \
            f'curvature = {curvature/num_batches}')

        self.test_loss.append(test_loss/num_batches)
        self.test_acc_adv.append(100.*adv_acc/total)
        self.test_acc_clean.append(100.*clean_acc/total)
        self.test_curv.append(curvature/num_batches)
        # if self.test_acc_adv[-1] > self.test_acc_adv_best:
            
        return test_loss/num_batches, 100.*adv_acc/total, 100.*clean_acc/total, curvature/num_batches           

    
    def _find_z(self, inputs, targets):
        '''
        regularizer
        '''
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(self.net.eval()(inputs), targets)                
        loss_z.backward(torch.ones(targets.size()).to(self.device))         
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
        zero_gradients(inputs) 
        self.net.zero_grad()

        return z, norm_grad
    
        
    def regularizer(self, inputs, targets, h = 3., lambda_ = 4, mode = 'normal'):
        z, norm_grad = self._find_z(inputs, targets)
        
        inputs.requires_grad_()
        outputs_pos = self.net.eval()(inputs + z)
        outputs_orig = self.net.eval()(inputs)

        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(self.device),
                                        create_graph=True)[0]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        self.net.zero_grad()

        return torch.sum(self.lambda_ * reg) / float(inputs.size(0)), norm_grad
        
            
    def save_model(self, path):
        '''
        save the model
        '''
        print('Saving...')
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        
    def import_model(self, path):
        '''
        Import the already trained model
        '''
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
    
    def write_file(self, name):
        with open(name, 'w') as f:
            f.write('[train_loss, train_acc, train_curv, test_loss, test_acc_clean, test_acc_adv, test_curv]\n')
            f.write('[')
            nums = [self.train_loss_best.item(), self.train_acc_best, self.train_curv_best.item(), self.test_loss_best,
                    self.test_acc_clean_best, self.test_acc_adv_best, self.test_curv_best]
            nums = ','.join([str(num) for num in nums])
            f.write(nums)
            f.write(']\n')
            f.write('[')
            nums = [self.train_loss[-1].item(), self.train_acc[-1], self.train_curv[-1].item(), self.test_loss[-1],
                    self.test_acc_clean[-1], self.test_acc_adv[-1], self.test_curv[-1]]
            nums = ','.join([str(num) for num in nums])
            f.write(nums)
            f.write(']\n')
           
            
    def plot_results(self):
        plt.figure(figsize=(15,12))
        plt.suptitle('Results',fontsize = 18,y = 0.96)
        plt.subplot(3,3,1)
        plt.plot(self.train_acc, Linewidth=2, c = 'C0')
        plt.plot(self.test_acc_clean, Linewidth=2, c = 'C1')
        plt.plot(self.test_acc_adv, Linewidth=2, c = 'C2')
        plt.legend(['train_clean', 'test_clean', 'test_adv'], fontsize = 14)
        plt.title('Accuracy', fontsize = 14)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14) 
        plt.grid()  
        plt.subplot(3,3,2)
        plt.plot(self.train_curv, Linewidth=2, c = 'C0')
        plt.plot(self.test_curv, Linewidth=2, c = 'C1')
        plt.legend(['train_curv', 'test_curv'], fontsize = 14)
        plt.title('Curvetaure', fontsize = 14)
        plt.ylabel('curv', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.subplot(3,3,3)
        plt.plot(self.train_loss, Linewidth=2, c = 'C0')
        plt.plot(self.test_loss, Linewidth=2, c = 'C1')
        plt.legend(['train', 'test'], fontsize = 14)
        plt.title('Loss', fontsize = 14)
        plt.ylabel('loss', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()
        




    


