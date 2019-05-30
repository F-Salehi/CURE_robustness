import torch
import copy
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt
from pgd_parallel import pgd
import torchvision
from resnet import *
import os
from wide_resnet import *
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from wide_resnet import *
from utils import progress_bar
from torch.distributions import uniform

    
class CURE():
    def __init__(self, net, trainloader, testloader, device='cuda', lambda_ = 4, precentage = 1,
                 mode = 'CURE'):
        '''
        CURE Class
        precentage: the precentage of the minimum eigens chosen
        '''
        
        self.net = net.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.lambda_ = lambda_
        self.precentage = precentage

        self.train_loss, self.train_acc, self.train_curv = [], [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean, self.test_curv = [], [], [], []
        self.train_loss_best, self.train_acc_best, self.train_curv_best = 0, 0, 0
        self.test_loss_best, self.test_acc_adv_best, self.test_acc_clean_best, self.test_curv_best = 0, 0, 0, 0
    

    def set_optimizer(optim_alg='Adam', args={'lr':1e-4}, scheduler=None, args_scheduler={}):
        '''
        setting the optimizer of the network
        '''
        self.optimizer = getattr(optim, optim_alg)([x], **args)
        if not scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10**6, gamma=1)
        else:
            self.scheduler = getattr(optim.lr_scheduler, scheduler)(self.optimizer, **args_scheduler)
        
            
    def train(self, h = 3, start_epoch = 0, end_epoch = 15):
        for epoch in range(start_epoch, end_epoch):
            if epoch == 0:
                h = 0.1
            elif epoch == 1:
                h = 0.4
            elif epoch == 2:
                h = 0.8
            elif epoch == 3:
                h = 1.8
            elif epoch == 4:
                h = 3
            self.train_(epoch, h)
            self.test(epoch, lambda_= 4, num_batches = 10, h=3)
            #self.scheduler.step()
        
    def train_(self, epoch, h):
        '''
        training the model
        '''
        print('\nEpoch: %d' % epoch)
        self.net.eval()
        train_loss = 0
        correct = 0
        total = 0
        curv, curveture, distance, norm_grad_sum  = 0, 0, 0, 0
        for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)
            outputs = self.net(inputs)
            _, predicted = outputs.max(1)
            X, id_max = torch.topk(outputs, 2, dim=1)
            a = X[:,0].data - X[:,1].data
            b = outputs[range(len(targets)),targets].data - X[:,0].data
            reward = (predicted.data == targets).float() * a + (1.0-(predicted.data==targets)).float() * b
            distance += sum(reward).item()
            outcome = predicted.data == targets
            if self.only_correct_flag:
                outcome_regual = outcome
            else:
                outcome_regual = torch.ones(len(targets), dtype=torch.short) 
                
            if self.mode == 'normal':
                loss_reg, loss_reg_all = 0, 0 
                grad_norm = self.gradient_loss_input(inputs, targets)[0]
            elif self.mode == 'CURE':
                loss_reg, loss_reg_all, grad_norm = self.regularizer_min_max(inputs, targets, h=h) 
            elif self.mode == 'fast':
                loss_reg, grad_norm = self.fast_regual(inputs, targets, h=h) 
                loss_reg_all = 0
            else:
                raise NameError('Unknown regularizer mode')
            norm_grad_sum += grad_norm                       
            curveture += loss_reg_all
            neg_log_likelihood = self.criterion(outputs, targets) 
            loss = neg_log_likelihood + loss_reg
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


            train_loss += neg_log_likelihood.item() + loss_reg_all 
            correct += outcome.sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | curv: %.3f | distance = %.3f |norm_grad = %.3f'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total,curveture/(batch_idx+1), distance/total, norm_grad_sum/(batch_idx+1)  ))
            
        self.train_loss.append(train_loss/(batch_idx+1))
        self.train_acc.append(100.*correct/total)
        self.train_curv.append(curveture/(batch_idx+1))
                
    def test(self, epoch, lambda_= 4, num_batches = 10, h=3):  
        test_loss, adv_acc, total, curveture, clean_acc, grad_sum = 0, 0, 0, 0, 0, 0
        data = iter(self.testloader)
        for i in range(num_batches):
            inputs, targets, _ = next(data)
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
                    num_steps=20, epsil=eps) # default parameters: 0.04, num_steps=20
            inputs_pert = inputs_pert + eps * torch.Tensor(r).to(self.device)
            outputs = self.net(inputs_pert)
            probs, predicted = outputs.max(1)
            adv_acc += predicted.eq(targets).sum().item()
            cur, norm_grad = self.regularizer(inputs, targets, h=h)
            grad_sum += norm_grad
            curveture += cur
            test_loss += cur

        print(f'epoch = {epoch}, adv_acc = {100.*adv_acc/total}, clean_acc = {100.*clean_acc/total}, loss = {test_loss/num_batches}, curvature = {curveture/num_batches}, norm_grad = {grad_sum/num_batches}')
        self.test_loss.append(test_loss/num_batches)
        self.test_acc_adv.append(100.*adv_acc/total)
        self.test_acc_clean.append(100.*clean_acc/total)
        self.test_curv.append(curveture/num_batches)
        if self.test_acc_adv[-1] > self.test_acc_adv_best:
            self.train_loss_best, self.train_acc_best, self.train_curv_best = self.train_loss[-1],\
                self.train_acc[-1],self.train_curv[-1]
            self.test_loss_best, self.test_acc_adv_best, self.test_acc_clean_best, self.test_curv_best = self.test_loss[-1],\
            self.test_acc_adv[-1], self.test_acc_clean[-1], self.test_curv[-1]
        return test_loss/num_batches, 100.*adv_acc/total, 100.*clean_acc/total, curveture/num_batches           

    def gradient_loss_input(self, inputs, targets):
        inputs = copy.deepcopy(inputs)
        inputs.requires_grad_()
        loss_z = self.criterion(self.net.eval()(inputs), targets)                
        loss_z.backward(torch.ones(targets.size()).to('cuda:0'))                
        grads = inputs.grad.data.detach() + 0.
        grads_norm = []
        for i in range(grads.size(0)):
            grads_norm.append(torch.norm(grads[i]).item())
        self.net.zero_grad()
        return torch.norm(loss_z).item(), grads_norm
    
    def _find_z(self, inputs, targets):
        '''
        regularizer
        '''
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(self.net.eval()(inputs), targets)                
        loss_z.backward(torch.ones(targets.size()).to('cuda:0'))         
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        zero_gradients(inputs) 
        self.net.zero_grad()

        return z, norm_grad
    
    def regularizer_min_max(self, inputs, targets, h = 2.):
        '''
        regularizer
        '''
        num_min = int(len(targets) * self.precentage)
        z, norm_grad = self._find_z(inputs, targets)
        if self.learn_h_flag:
            outputs = self.net.eval()(inputs)
            h = self.learn_h(outputs, inputs)
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)   
        inputs.requires_grad_()
        outputs_pos = self.net(inputs + z)
        outputs_orig = self.net(inputs)
        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)
        diff_loss = loss_pos #- loss_orig
        grad_diff = torch.autograd.grad(diff_loss, inputs, grad_outputs=torch.ones(targets.size()).to('cuda:0'),
                                        create_graph=True)[0]
        if self.learn_h:
            grad_diff /= h
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        ### Selecting the precentage of the data
        reg_select = copy.copy(reg.data.detach()).cpu().numpy() + 0.
        reg_selected_idx = np.argsort(reg_select)[:num_min]
        lambda_ = torch.ones_like(targets,dtype=torch.float32) * self.lambda_max
        lambda_[reg_selected_idx] = self.lambda_min
        loss_reg = torch.sum(lambda_ * reg) / 128.0 #float(inputs.size(0)) #float(len(reg_selected_idx))
        loss_reg_all = torch.sum( 4 * reg.detach().data) / 128.0
        self.net.zero_grad()
        return loss_reg, loss_reg_all, norm_grad
        
    def regularizer(self, inputs, targets, h = 3., lambda_ = 4, mode = 'normal'):
        z, norm_grad = self._find_z(inputs, targets)
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)  
        inputs.requires_grad_()
        outputs_pos = self.net(inputs + z)
        outputs_orig = self.net(inputs)

        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to('cuda:0'),
                                        create_graph=True)[0]
        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
        self.net.zero_grad()
        if mode == 'learn':
            return torch.sum(self.lambda_min * reg) / float(inputs.size(0)), norm_grad
        elif mode == 'eigen':
            return reg.detach()
        elif mode == 'curv_diff':
            _, predicted = outputs_orig.max(1)
            reg_correct = reg[targets == predicted]
            reg_false = reg[targets != predicted]
            return reg_correct.detach(), reg_false.detach(), norm_grad
        else:
            loss_reg = torch.sum(lambda_ * reg) / float(inputs.size(0))
            return loss_reg.detach().item(), norm_grad
        
    def fast_regual(self, inputs, targets, h = 1.):
        inputs = copy.deepcopy(inputs) 
        z, norm_grad = self._find_z(inputs, targets)
        if self.learn_h_flag:
            h = self.learn_h(outputs, inputs)
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
        inputs.requires_grad_()
        outputs_pos = self.net(inputs + z)
        loss_pos = self.criterion(outputs_pos, targets)
        loss_pos.backward(torch.ones(targets.size()).to('cuda:0'))         
        grad = inputs.grad.data + 0.0
        self.net.zero_grad()
        z1 = torch.sign(grad).detach() 
        h1 = self.lambda_min
        z1 = 1.*(h1) * (z1+1e-7) / (z1.reshape(z1.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
        inputs_aug = inputs + z1
        inputs_aug.detach_()
        inputs_aug.requires_grad = False
        outputs_aug = self.net(inputs_aug)
        loss_reg = self.criterion(outputs_aug, targets)
        return loss_reg, norm_grad
    

    def get_curvs(self, loader ,h, mode = 'all'):
        """
        mode == all means that the curv of all datapoints are returned
        mode == diff returns the curv of correctly classified datapoints and
        wrongly classified datapoints seperately.
        """
        if mode == 'all':
            curvs_all = torch.zeros(len(loader.dataset)).to(self.device)
        elif mode == 'diff':
            curv_correct = []
            curv_false = []
        else:
            raise NameError('Unknown mode')
        for batch_idx, (inputs, targets, indices) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if mode == 'all': 
                curv = self.regularizer(inputs, targets, h = h, lambda_ = 4, mode = 'eigen')
                curvs_all[indices] = curv
            elif mode == 'diff':
                c_corr, c_false = self.regularizer(inputs, targets, h = h, lambda_ = 4, mode = 'eigen')
                curv_correct.append(c_corr)
                curv_false.append(c_false)
        if mode == 'all': 
            return curvs_all
        else:
            curv_correct = np.hstack(curv_correct)
            curv_false = np.hstack(curv_false)
            return curv_correct, curv_false
        
    def set_only_correct_regual(self, flag = True):
        self.only_correct_flag = flag
    
    def return_h(self, dataloader): 
        '''
        returns the h computed for the network
        '''
        h = torch.zeros(len(dataloader.dataset))
        for batch_idx, (inputs, targets, indices) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad_()
            zero_gradients(inputs) 
            outputs = self.net.eval()(inputs)
            h_temp = self.learn_h(outputs, inputs)
            h[indices] = h_temp.view(-1).cpu()
        return h
            
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
        Importing the already trained model
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
        




    


