import torch
import os
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyDOE import lhs
from torchvision import  datasets,transforms
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import time
#Set default dtype to float32
torch.set_default_dtype(torch.float)
from tqdm import tqdm
#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)
DIFF=0.1
 ## parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{device} is available.")



class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)

class simplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten=nn.Flatten()
        self.sequential=nn.Sequential(
            nn.Linear(in_features=2, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=60,bias=True),
            Swish(),
            nn.Linear(in_features=60, out_features=1,bias=True)
            
            )
      
          
    def forward(self,x):
            output = self.sequential(x)
            return output
        

modelv2=simplePINN().to(device)
# print(modelv2)
#print(modelv2[1].state_dict()) 
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        print(m)
        #nn.init.xavier_normal_(m.weight)


    modelv2.apply(init_normal)
  
########################   

########################################    
    
    
    
    
    
def LossPDE(x,y,DIFF):
    x = torch.Tensor(x).to(device)
    y=  torch.Tensor(y).to(device)
    x.requires_grad = True
    y.requires_grad = True
    sample = torch.cat((x,y),1)
    U=modelv2(sample)
    U = U.view(len(U),-1)
    U_x=torch.autograd.grad(U,x,grad_outputs=torch.ones_like(x),create_graph=True,only_inputs=True)[0]
    U_xx=torch.autograd.grad(U_x,x,grad_outputs=torch.ones_like(x),create_graph=True,only_inputs=True)[0]
    U_y=torch.autograd.grad(U,y,grad_outputs=torch.ones_like(y),create_graph=True,only_inputs=True)[0]
    U_yy=torch.autograd.grad(U_y,y,grad_outputs=torch.ones_like(y),create_graph=True,only_inputs=True)[0]
    f_source=-DIFF*6
    loss_1 =-f_source-DIFF*(U_xx+U_yy)
    loss_f=nn.MSELoss()
    loss_PDE=loss_f(loss_1,torch.zeros_like(loss_1))
    return loss_PDE


def LossBondaryDirichlet(all_X_u_trainDirichlet,all_u_trainDirichlet):
    xB_Dirichlet=all_X_u_trainDirichlet[:,0][:,None]
    yB_Dirichlet=all_X_u_trainDirichlet[:,1][:,None]
    
    xB_Dirichlet= torch.Tensor(xB_Dirichlet).to(device)
    yB_Dirichlet=  torch.Tensor(yB_Dirichlet).to(device)
    sample1 = torch.cat((xB_Dirichlet,yB_Dirichlet),1)
  
    UB_Dirichlet=modelv2(sample1)
    all_u_trainDirichlet = (torch.Tensor(all_u_trainDirichlet).to(device))
    loss_f1=nn.MSELoss()
    loss_BD=loss_f1(UB_Dirichlet,all_u_trainDirichlet)

    return loss_BD

def LossBondaryNeumann(all_X_u_trainNeumann,all_u_trainNeumann):
    xB_Neumann=all_X_u_trainNeumann[:,0][:,None]
    yB_Neumann=all_X_u_trainNeumann[:,1][:,None]
    xB_Neumann= torch.Tensor(xB_Neumann).to(device)
    yB_Neumann=  torch.Tensor(yB_Neumann).to(device)
    xB_Neumann.requires_grad = True
    yB_Neumann.requires_grad = True
    sample1 = torch.cat((xB_Neumann,yB_Neumann),1)
    UB_Neumann=modelv2(sample1)
    UB_Neumann = UB_Neumann.view(len(UB_Neumann),-1)
    UB_y=torch.autograd.grad(UB_Neumann,yB_Neumann,grad_outputs=torch.ones_like(yB_Neumann),create_graph=True,only_inputs=True)[0]
    all_u_trainNeumann = (torch.Tensor(all_u_trainNeumann).to(device))
    loss_f1=nn.MSELoss()
    loss_BN=loss_f1(UB_y,all_u_trainNeumann)
    return loss_BN

lb = np.array([0, 0]) #lower bound
ub = np.array([1, 1])  #upper bound
#Set default dtype to float32
def Exact(x,y):
    usol = 1+x**2+2*y**2 #solution chosen for convinience
    return usol




def DatatraingDirichlet(N_bun_point):
    xy_left= np.zeros((N_bun_point, 2))
    xy_left[..., 0] = np.zeros(N_bun_point)                               # x = 0
    xy_left[..., 1] = np.linspace(0.0, 1.0, num=N_bun_point)  
    leftedge_u=Exact(xy_left[:,0], xy_left[:,1])[:,None]
    
    
    
    xy_right = np.zeros((N_bun_point, 2))
    xy_right[..., 0] = np.ones(N_bun_point)                               # x = 1
    xy_right[..., 1] = np.linspace(0.0, 1.0, num=N_bun_point)             # y = 0 ~ +1
    rightedge_u=Exact(xy_right[:,0], xy_right[:,1])[:,None]
    
   
    
    all_X_u_trainDirichlet = np.vstack([xy_left, xy_right])
    all_u_trainDirichlet = np.vstack([leftedge_u, rightedge_u])
    

    return all_X_u_trainDirichlet,all_u_trainDirichlet

def Datatraingcollocaton(N_col_point):
    X_f = lb + (ub-lb)*lhs(2,N_col_point)
    x=X_f[:,0][:,None]
    y=X_f[:,1][:,None]
    return x,y

def DatatraingNeumann(N_bun_point):
    
    xy_top = np.zeros((N_bun_point, 2))
    xy_top[..., 0] = np.linspace(0.0, 1.0, num=N_bun_point)                # x = 0 ~ +1
    xy_top[..., 1] = np.ones(N_bun_point)
    topedge_u=4*np.ones((N_bun_point,1))
         
    xy_bottom = np.zeros((N_bun_point, 2))
    xy_bottom[..., 0] = np.linspace(0.0, 1.0, num=N_bun_point)          # x = 0 ~ +1
    xy_bottom[..., 1] = np.zeros(N_bun_point)
    bottomedge_u=np.zeros((N_bun_point, 1))
    
    all_X_u_trainNeumann = np.vstack([ xy_top,xy_bottom])
    all_u_trainNeumann = np.vstack([topedge_u, bottomedge_u])
    
    return all_X_u_trainNeumann,all_u_trainNeumann





################ Main Program #######################

batchsize = 100
epochs  =1000
step_size=10000
gamma = .1
# Domain bounds
lb = np.array([0, 0]) #lower bound
ub = np.array([1, 1])  #upper bound
N_col_point=20000
N_bun_point=100

x,y=Datatraingcollocaton(N_col_point)
all_X_u_trainDirichlet,all_u_trainDirichlet=DatatraingDirichlet(N_bun_point)
all_X_u_trainNeumann,all_u_trainNeumann=DatatraingNeumann(N_bun_point)




x = torch.Tensor(x).to(device)
y=  torch.Tensor(y).to(device)

dataset = TensorDataset(x,y)
dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
learning_rate=10**-3
#optimizer = torch.optim.Adam(modelv2.parameters(), lr=learning_rate,eps = 10**-3)
optimizer = optim.Adam(modelv2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-20)
#optimizer = torch.optim.Adam(modelv2.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-04, weight_decay=0, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# max_iter = 16600
def train(x,y,epochs,dataloader,modelv2,optimizer):
    for epoch in tqdm(range( (epochs))):
       #for batch_idx,(x,y) in enumerate(dataloader):
            # print(f"{batch_idx}")
            loss_PDE=LossPDE(x,y,DIFF)
            loss_Dirichlet=LossBondaryDirichlet(all_X_u_trainDirichlet,all_u_trainDirichlet)
            loss_neumann=LossBondaryNeumann(all_X_u_trainNeumann,all_u_trainNeumann)
            # summation of loss function including boundary and PDE
            loss=loss_PDE+ loss_Dirichlet+loss_neumann
            # print(loss)
            loss.backward()
            #update parameters W,B
            optimizer.step()
            #reset gradients
            modelv2.zero_grad()
            if epoch % 100 ==0:
                print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE: {:.8f} \tLoss_neumann: {:.10f}\tLoss_Dieichlet: {:.10f}'.format(epoch, loss.item(),loss_PDE.item(),loss_neumann.item(),loss_Dirichlet.item()))

train(x,y,epochs,dataloader,modelv2,optimizer)

#saving and loading model
torch.save(modelv2.state_dict(),'modelv2_wight.pth')
modelv3=simplePINN()
modelv3.load_state_dict(torch.load('modelv2_wight.pth'))
modelv3.eval()


x1 = np.linspace(0, 1,100)
y1 = np.linspace(0, 1,100)
x1, y1 = np.meshgrid(x1, y1)
x1 = np.reshape(x1, (np.size(x1[:]),1))
y1 = np.reshape(y1, (np.size(y1[:]),1))
x1 = torch.Tensor(x1).to(device)
y1=  torch.Tensor(y1).to(device)
with torch.no_grad():
    sample1 = torch.cat((x1,y1),1)
    pred=modelv2(sample1)
    U_exat=1+x1**2+2*y1**2
    Err = torch.linalg.norm((pred-U_exat),2)/torch.linalg.norm(U_exat,2)
    print(f"Error: { Err}")
#
# fig, axs = plt.subplots(2, 2)
# plt.subplot(1, 1, 1)
# plt.scatter(x1.detach().numpy(), y1.detach().numpy(), c =torch.abs(pred-U_exat), cmap = 'jet')
# plt.title('PINN results')
# plt.colorbar()
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 
x1 = np.linspace(0, 1,100)
y1 = np.linspace(0, 1,100)
x1, y1 = np.meshgrid(x1, y1)


pred=pred.detach().numpy()
U_exat=U_exat.detach().numpy()
qq=np.abs(pred-U_exat)
Z=np.reshape(Z, (100,100))
 # Plot the surface.
surf = ax.plot_surface(x1, y1, Z, cmap=cm.jet,linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.25, aspect=5)

plt.show()