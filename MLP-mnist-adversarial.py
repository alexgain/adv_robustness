import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

from torchvision import datasets, transforms

#from fashion import fashion

import struct
from copy import deepcopy
from time import time, sleep
import gc

from sklearn.preprocessing import normalize

np.random.seed(1)

cuda_boole = torch.cuda.is_available()

###                               ###
### Data import and preprocessing ###
###                               ###

N = 60000
BS = 128
rbf_boole = True
ST = False
save = False

N2 = 200
epochs = 2000
LR = 0.01
width = 500


transform_data = transforms.ToTensor()

train_set = datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data)

# train_set.train_data = train_set.train_data[:N]

##adding noise:
##noise_level = 0
##train_set.train_data = train_set.train_data.float()
##train_set.train_data = train_set.train_data + noise_level*torch.abs(torch.randn(*train_set.train_data.shape))
##train_set.train_data = train_set.train_data / train_set.train_data.max()

train_loader = torch.utils.data.DataLoader(train_set, batch_size = BS, shuffle=False)

train_loader_bap = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transform_data),
    batch_size=N2, shuffle=False)

test_dataset = datasets.MNIST(root='./data', 
                            train=False, 
                            transform=transform_data,
                           )
##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)

test_loader_bap = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=N2, shuffle=False)


##adding noise to test:
##test_dataset.test_data = test_dataset.test_data.float()
##test_dataset.test_data = test_dataset.test_data + noise_level*torch.abs(torch.randn(*test_dataset.test_data.shape))
##test_dataset.test_data = test_dataset.test_data / test_dataset.test_data.max()

##test_loader = torch.utils.data.DataLoader(
##    datasets.MNIST('./data', train=False, transform=transforms.Compose([
##                       transforms.ToTensor(),
##                       transforms.Normalize((0.1307,), (0.3081,))
##                   ])),
##    batch_size=10000, shuffle=False)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
##ytrain = to_categorical(ytrain, 3)
##ytest = to_categorical(ytest, 3)

##entropy = np.load('entropy_mnist_train.npy')[:N2]
##entropy_test = np.load('entropy_mnist_test.npy')[:N2]
##np.random.shuffle(entropy)

###                      ###
### Define torch network ###
###                      ###

def rbf(x,beta,mu):
    return torch.exp(-beta*(x.view(x.shape[0],-1,1)-mu).norm(dim=2))


class Net(nn.Module):
    def __init__(self, input_size, width, num_classes,rbf_boole=True):
        super(Net, self).__init__()

        ##feedfoward layers:

        bias_ind = True

        self.ff1 = nn.Linear(input_size, width, bias = bias_ind) #input

        self.ff2 = nn.Linear(width, width, bias = bias_ind) #hidden layers
        self.ff3 = nn.Linear(width, width, bias = bias_ind)
        self.ff4 = nn.Linear(width, width, bias = bias_ind)
        self.ff5 = nn.Linear(width, width, bias = bias_ind)

##        self.ff_out = nn.Linear(width, num_classes, bias = bias_ind) #output     
        self.ff_out = nn.Linear(width, 10, bias = bias_ind) #output     
        
        ##activations:
        self.do = nn.Dropout()
        self.relu = nn.ReLU()
        self.sm = nn.Softmax()

        self.beta1 = nn.Parameter(torch.Tensor([0.01]))
        self.beta2 = nn.Parameter(torch.Tensor([0.01]))
        self.beta3 = nn.Parameter(torch.Tensor([0.01]))
        self.beta4 = nn.Parameter(torch.Tensor([0.01]))
        self.beta5 = nn.Parameter(torch.Tensor([0.01]))
        self.mu1 = nn.Parameter(torch.randn(1,width,width))
        self.mu2 = nn.Parameter(torch.randn(1,width,width))
        self.mu3 = nn.Parameter(torch.randn(1,width,width))
        self.mu4 = nn.Parameter(torch.randn(1,width,width))
        self.mu5 = nn.Parameter(torch.randn(1,width,width))
        
        self.rbf_boole=rbf_boole

        
    def forward(self, input_data):

        if not self.rbf_boole:
            out = self.relu(self.ff1(input_data)) #input
            out = self.relu(self.ff2(out)) #hidden layers
            out = self.relu(self.ff3(out))
            out = self.relu(self.ff4(out))
            out = self.relu(self.ff5(out))

        else:
            out = rbf(self.ff1(input_data),self.beta1,self.mu1) #input
            out = rbf(self.ff2(out),self.beta2,self.mu2) #hidden layers
            out = rbf(self.ff3(out),self.beta3,self.mu3)
            # out = rbf(self.ff4(out),self.beta4,self.mu4)
            # out = rbf(self.ff5(out),self.beta5,self.mu5)


        out = self.ff_out(out)

        return out 



###hyper-parameters:
input_size = 28*28
num_classes = 10

###defining network:        
my_net = Net(input_size, width, num_classes,rbf_boole=rbf_boole)
if cuda_boole:
    my_net = my_net.cuda()


###                       ###
### Loss and optimization ###
###                       ###

LR2 = 1.0
##loss_metric = nn.MSELoss()
loss_metric = nn.CrossEntropyLoss()
##loss_metric = torch.nn.NLLLoss()
# optimizer = torch.optim.SGD(my_net.parameters(), lr = LR, momentum = 0.9)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001)
##optimizer = torch.optim.RMSprop(my_net.parameters(), lr = 0.00001, momentum = 0.8)
optimizer = torch.optim.Adam(my_net.parameters(), lr = 0.001)

###                         ###
### Adversarial Attack code ###
###                         ###

class GradientAttack():
        def __init__(self, loss, epsilon):
##            super(GradientAttack, self).__init__()
##            self.model = model
            self.loss = loss
            self.epsilon = epsilon

        def forward(self, x, y_true, model):
            # Give x a gradient buffer
            x_adv = x
##            x_adv.requires_grad = True

            # Build the loss function at J(x,y)
            y = model.forward(x_adv)
            J = self.loss(y,y_true)# - 1*bap_val(0)

            # Ensure that the x gradient buffer is 0
            if x_adv.grad is not None:
                    x_adv.grad.data.fill_(0)

            # Compute the gradient 
            x_grad = torch.autograd.grad(J, x_adv)[0]

            # Create the adversarial example and ensure 
            ##		x_adv = x_adv + self.epsilon*x_grad
            x_adv = x + self.epsilon*x_grad.sign_()

            # Clip the results to ensure we still have a picture
            # This CIFAR dataset ranges from -1 to 1.
            x_adv = torch.clamp(x_adv, 0, 1)

            return x_adv


adv_attack = GradientAttack(loss_metric, 0.1)

###                 ###
### Attractor Algs. ###
###                 ###

#Some more hyper-params and initializations:

eps_ball = 3.0 #controls how big we want the attractor spaces


###                           ###
### Alternative BAP calc func ###
###                           ###

def CO_calc(model,x,y):
##    slopes = []
##    for i in range(x.shape[0]//100):
##        slopes = slopes + list(model.beta(x[i*100:(i+1)*100].cuda()).cpu().data.numpy())
    slopes = model.beta(x.cuda()).cpu().data.numpy()
    slopes = np.array(slopes)
    return np.corrcoef(slopes,y[:slopes.shape[0]])[0,1]


###          ###
### Training ###
###          ###

#Some more hyper-params and initializations:

##train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
##test_loader = torch.utils.data.DataLoader(test, batch_size=BS, shuffle=False)

##printing train statistics:
def train_acc():
    correct = 0
    total = 0
    for images, labels in train_loader:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long() 
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    print('Accuracy of the network on the train images: %f %%' % (100 * correct / total))
    
def test_acc():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1, 28*28))
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

def test_acc_adv():
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader_bap:
        if cuda_boole:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images.view(-1,28*28), requires_grad=True)
        images = adv_attack.forward(images, Variable(labels), my_net)
        outputs = my_net(images)
        _, predicted = torch.max(outputs.data, 1)
    ##    labels = torch.max(labels.float(),1)[1]
    ##    predicted = torch.round(outputs.data).view(-1).long()
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        break

    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
    
def bap_val(verbose = 1):
    for images, labels in train_loader_bap:
        images = Variable(images.view(-1, 28*28))
        if cuda_boole:
            images = images.cuda()
        bap = my_net.corr(images,entropy)
        break
    if verbose:
        print('BAP value:',bap)
    return bap#.cpu().data.numpy()[0]

##def bap_val_true():
##    for images, labels in train_loader_bap:
##        images = Variable(images.view(-1, 28*28))
##        if cuda_boole:
##            images = images.cuda()
##        bap = my_net.corr(images,entropy)
##        break
####    if verbose:
####        print('BAP value:',bap)
##    return bap


def bap_val_test(verbose = 1):
    for images, labels in test_loader_bap:
        images = Variable(images.view(-1, 28*28))
        if cuda_boole:
            images = images.cuda()
        bap = my_net.corr(images,entropy_test)
        break
    if verbose:
        print('BAP value:',bap)
    return bap#.cpu().data.numpy()[0]

    
##entropy = torch.Tensor(entropy)
##if cuda_boole:
##    entropy = entropy.cuda()
##entropy = Variable(entropy)
##
##entropy_test = torch.Tensor(entropy_test)
##if cuda_boole:
##    entropy_test = entropy_test.cuda()
##entropy_test = Variable(entropy_test)
    
if ST:
    my_net_teacher = Net(input_size, width, num_classes,rbf_boole=False)
    my_net_teacher.load_state_dict(torch.load('./teacher.state'))
    if cuda_boole:
        my_net_teacher = my_net_teacher.cuda()

train_acc()
test_acc()
test_acc_adv()

###training loop (w/corr):
t1 = time()
for epoch in range(epochs):

    state_distance = 0

    ##time-keeping 1:
    time1 = time()

    for i, (x,y) in enumerate(train_loader):

        ##some pre-processing:
        x = x.view(-1,28*28)
##        y = y.float()
##        y = y.long()
##        y = torch.Tensor(to_categorical(y.long().cpu().numpy(),num_classes)) #MSE

        ##cuda:
        if cuda_boole:
            x = x.cuda()
            y = y.cuda()

        ##data preprocessing for optimization purposes:        
        x = Variable(x)
        y = Variable(y) #MSE 1-d output version

        ##adding noise:
##        noise_level = 2
##        noise = Variable(torch.abs(noise_level*torch.randn(*x.shape)).float())
##        if cuda_boole:
##            noise = noise.cuda()
##        x += noise
##        x /= x.max()

##        for images_test, labels_test in test_loader_bap:
##            images_test = Variable(images_test.view(-1, 28*28))
##            if cuda_boole:
##                images_test = images_test.cuda()
##            bap_test = my_net.corr(images_test,entropy_test)
##            break

        ###regular BP gradient update:
        optimizer.zero_grad()
        if ST:
            y = torch.Tensor(my_net_teacher.forward(x).cpu().data.numpy())
            if cuda_boole:
                y = y.cuda()
        outputs = my_net.forward(x)
        if ST:
            loss = (outputs-y).norm()/outputs.shape[0]
        else:
            loss = loss_metric(outputs,y)
##        if bap_train_boole:
##            loss +=  bap_val_test(0)
        loss.backward()
                
        ##performing update:
        optimizer.step()

        ##Performing attractor update:
        # rand_vec = Variable(torch.randn(*list(x.shape)))
        # if cuda_boole:
        #     rand_vec = rand_vec.cuda()
        # x_pert = x + (eps_ball)*(rand_vec / rand_vec.norm())
        
        # optimizer.zero_grad()

        # ##getting two states:
        # state1 = my_net.forward(x)
        # state2 = my_net.forward(x_pert)

        # loss2 = LR2*(state1 - state2).norm()
        # loss2.backward(retain_graph = True)
                
        ##performing update:
        # optimizer.step()

        ##accumulating loss:
        # state_distance += float(loss.cpu().data.numpy())
        
        ##printing statistics:
        
        ##printing statistics:
        if (i+1) % np.floor(N/BS) == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, epochs, i+1, N//BS, loss.data.item()))
            # print('Avg Batch Distance:',state_distance/(i+1))

            train_acc()
            test_acc()
            test_acc_adv()
##            bap_train = bap_val(1).cpu().data.numpy()[0]
##            bap_test = bap_val_test(1).cpu().data.numpy()[0]
##            if bap_train_max < bap_train:
##                bap_train_max = bap_train
##            if bap_test_max < bap_test:
##                bap_test_max = bap_test

    ##time-keeping 2:
    time2 = time()
    print('Elapsed time for epoch:',time2 - time1,'s')
    print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
    print()

t2 = time()
print((t2 - t1)/60,'total minutes elapsed')

if save == True:
    torch.save(my_net.state_dict(),'teacher.state')