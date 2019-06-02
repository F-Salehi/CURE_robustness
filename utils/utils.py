import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import time
import sys

def read_vision_dataset(path, batch_size=128, num_workers=4, dataset='CIFAR10', transform=None):
    '''
    Read dataset available in torchvision
    
    Arguments:
        dataset : string
            The name of dataset, it should be available in torchvision
        transform_train : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
        transform_test : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
    Return: 
        trainloader, testloader
    '''
    if not transform and dataset=='CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    trainset = getattr(datasets,dataset)(root=path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = getattr(datasets,dataset)(root=path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def pgd(inputs, net, epsilon=[1.], targets=None, step_size=0.04, num_steps=20, epsil=5./255.*8):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    input_shape = inputs.shape
    pert_image = copy.deepcopy(inputs).to('cuda:0')
    w = torch.zeros(input_shape)
    r_tot = torch.zeros(input_shape)
    
    denormal = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.2023, 1/0.1994, 1/0.2010)),
                             transforms.Normalize((-0.4914, -0.4822, -0.4465), (1., 1., 1.))])                                   
    normal = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    pert_image = pert_image + (torch.rand(inputs.shape).to('cuda:0')-0.5)*2*epsil
    pert_image = pert_image.to('cuda:0')
    
    for ii in range(num_steps):
        pert_image.requires_grad_()    
        zero_gradients(pert_image)
        fs = net.eval()(pert_image)
        loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
        grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=True, retain_graph=False)[0]

        dr = torch.sign(grad.data)
        pert_image.detach_()
        pert_image += dr * step_size
        for i in range(inputs.size(0)):
            pert_image[i] = torch.min(torch.max(pert_image[i], inputs[i] - epsil), inputs[i] + epsil)
            pert_image[i] = pert_image[i] / torch.Tensor([1/0.2023, 1/0.1994, 1/0.2010]).view(3,1,1).cuda()
            pert_image[i] -= torch.Tensor([-0.4914, -0.4822, -0.4465]).view(3,1,1).cuda()
            pert_image[i] = torch.clamp(pert_image[i], 0., 1.)
            pert_image[i] = (pert_image[i] - torch.Tensor([0.4914, 0.4822, 0.4465]).view(3,1,1).cuda()) 
            pert_image[i] /= torch.Tensor([0.2023, 0.1994, 0.2010]).view(3,1,1).cuda()
                            
            #pert_image[i] = normal(torch.clamp(pert_image[i], 0., 1.))[None, :, :, :]
    
    r_tot = pert_image - inputs
    regul = np.linalg.norm(r_tot.cpu().flatten(start_dim=1, end_dim=-1), np.inf, axis=1)
    regul = torch.Tensor(regul).view(-1,1,1,1).cuda()
    r_tot = r_tot / regul
    
    return r_tot.cpu()


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
term_width, _ = shutil.get_terminal_size()
term_width = int(term_width)

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f