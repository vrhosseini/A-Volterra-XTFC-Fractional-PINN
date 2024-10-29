# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:00:38 2024

@author: vrhos
"""



import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split
import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
#Set default dtype to float32
torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
#%%%%%%%%%%%%%%%initializing ElectrodeProperties%%%%%%%%%%%%%%%%%%%%%%%%%

T=1



#%%



D1=1
D2=4




#%%
# class Swish(nn.Module):
# 		def __init__(self, inplace=True):
#  			super(Swish, self).__init__()
#  			self.inplace = inplace

# 		def forward(self, x):
#  			return x.mul_(torch.sigmoid(x)) 
class Swish1(nn.Module):
		def __init__(self, inplace=True):
 			super(Swish1, self).__init__()
 			self.inplace = inplace

		def forward(self, x):
 			return torch.sigmoid(x)
         
class Swish2(nn.Module):
		def __init__(self, inplace=True):
 			super(Swish2, self).__init__()
 			self.inplace = inplace

		def forward(self, x):
 			return torch.PReLU(x)
         
            
            
# class Swish(nn.Module):
# 		def __init__(self, inplace=True):
#  			super(Swish, self).__init__()
#  			self.inplace = inplace

# 		def forward(self, x):
#  			return torch.tanh(x)          
            
            
            

class simplePINN1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten=nn.Flatten()
        self.sequential=nn.Sequential(
            nn.Linear(in_features=2, out_features=20,bias=True),
            Swish1(),
            nn.Linear(in_features=20, out_features=20,bias=True),
            Swish1(),
            nn.Linear(in_features=20, out_features=20,bias=True),
            Swish1(),
            nn.Linear(in_features=20, out_features=20,bias=True),
            Swish1(),
            nn.Linear(in_features=20, out_features=20,bias=True),
            Swish1(),

            

            nn.Linear(in_features=20, out_features=1,bias=True)
            
            )
          
    def forward(self,x):
            output = self.sequential(x)
            return output
        
class simplePINN2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten=nn.Flatten()
        self.sequential=nn.Sequential(
            nn.Linear(in_features=2, out_features=100,bias=True),
            Swish2(),
            nn.Linear(in_features=100, out_features=100,bias=True),
            Swish2(),
            nn.Linear(in_features=100, out_features=100,bias=True),
            # Swish2(),
            # nn.Linear(in_features=50, out_features=50,bias=True),
            # Swish2(),
            # nn.Linear(in_features=50, out_features=50,bias=True),
            # Swish2(),
            # nn.Linear(in_features=50, out_features=50,bias=True),
            # Swish2(),
            nn.Linear(in_features=100, out_features=1,bias=True)
            
            )
      
          
    def forward(self,x):
            output = self.sequential(x)
            return output      
        

modelv1=simplePINN1().to(device)
modelv2=simplePINN1().to(device)

def init_normal1(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        print(m)
        #nn.init.xavier_normal_(m.weight)
def init_normal2(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        print(m)
        #nn.init.xavier_normal_(m.weight)

    modelv2.apply(init_normal1)
    modelv1.apply(init_normal2)
    
    
def f_real(x,t):
  return torch.exp(-t)*(torch.sin(np.pi*x))
def f_real_x(x,t):
  return np.pi*torch.exp(-t)*(torch.cos(np.pi*x))   
def Eq_exact(x,t):
  return torch.exp(-t)*(torch.sin(torch.pi*x))    
#%%
def LossPDE1(x, t):
    x = x.to(device)
    t = t.to(device)
    x.requires_grad = True
    t.requires_grad = True
    # print(x.shape)
    # print(t.shape)
    U_exact_PDE1= f_real(x,t)
    sample = torch.cat((x, t), 1)
    U1 = modelv1(sample)
    U1_t = torch.autograd.grad(U1, t, grad_outputs=torch.ones_like(U1), create_graph=True, only_inputs=True)[0]
    U1_x = torch.autograd.grad(U1, x, grad_outputs=torch.ones_like(U1), create_graph=True, only_inputs=True)[0]
    U1_xx = torch.autograd.grad(U1_x, x, grad_outputs=torch.ones_like(U1_x), create_graph=True, only_inputs=True)[0]
    f_source = U1_t  - D1*U1_xx-torch.exp(-t) * (-torch.sin(np.pi * x) +D1* np.pi**2 * torch.sin(np.pi * x))
    # print(f_source.shape)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_f1 = nn.MSELoss()
    loss_PDE1 = loss_f(f_source, torch.zeros_like(f_source))
    loss_U_exact_PDE1 = loss_f1(U1,  U_exact_PDE1)
    
    return loss_PDE1, loss_U_exact_PDE1

def LossPDE2(x, t):
    x = x.to(device)
    t = t.to(device)
    x.requires_grad = True
    t.requires_grad = True
    sample = torch.cat((x, t), 1)
    U2 = modelv2(sample)
    # print(x.shape)
    # print(t.shape)
    U_exact_PDE2= f_real(x,t)
    U2_t = torch.autograd.grad(U2, t, grad_outputs=torch.ones_like(U2), create_graph=True, only_inputs=True)[0]
    U2_x = torch.autograd.grad(U2, x, grad_outputs=torch.ones_like(U2), create_graph=True, only_inputs=True)[0]
    U2_xx = torch.autograd.grad(U2_x, x, grad_outputs=torch.ones_like(U2_x), create_graph=True, only_inputs=True)[0]
    f_source = U2_t  -D2* U2_xx-torch.exp(-t) * (-torch.sin(np.pi * x) +D2* np.pi**2 * torch.sin(np.pi * x))
    # print(f_source.shape)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_PDE2 = loss_f(f_source, torch.zeros_like(f_source))
    loss_U_exact_PDE2 = loss_f(U2,  U_exact_PDE2)
    # print(loss_PDE2)
    return loss_PDE2,loss_U_exact_PDE2

#loss for initiale  codition left 
def Loss_IC1(X0IC1,T0IC1, UIC1):
    X0IC1=torch.from_numpy(X0IC1).float().to(device)
    T0IC1=torch.from_numpy(T0IC1).float().to(device)
    UIC1=torch.from_numpy(UIC1).float().to(device)
    # print(X0IC1.shape)
    # print(T0IC1.shape)
    # print(T0IC1.shape)
    Ul_exact= f_real(X0IC1,T0IC1) 
    sample = torch.cat((X0IC1, T0IC1), 1)
    U_IC1 = modelv1(sample)
    loss_f = nn.MSELoss()
    loss_IC1 = loss_f(U_IC1,  UIC1)
    loss_Ul_exact = loss_f(U_IC1,  Ul_exact)

  
    return loss_IC1,loss_Ul_exact 

# #loss for initioal codition left 
def Loss_IC2(X0IC2,T0IC2, UIC2):
    X0IC2=torch.from_numpy(X0IC2).float().to(device)
    T0IC2=torch.from_numpy(T0IC2).float().to(device)
    UIC2=torch.from_numpy(UIC2).float().to(device)
    # print(X0IC1.shape)
    # print(T0IC1.shape)
    sample = torch.cat((X0IC2, T0IC2), 1)
    U_IC2 = modelv2(sample)
    loss_f = nn.MSELoss()
    loss_IC2 = loss_f(U_IC2,  UIC2) 
    # print(loss_IC1)
    return loss_IC2






#loss for boubdary codition left 
def Loss_BC1(xBC1,tBC1, UBC1):
    xBC1=torch.from_numpy(xBC1).float().to(device)
    tBC1=torch.from_numpy(tBC1).float().to(device)
    UBC1=torch.from_numpy(UBC1).float().to(device)
    # print(xBC1.shape)
    # print(tBC1.shape) 
    xBC1.requires_grad = True
    tBC1.requires_grad = True
    sample = torch.cat((xBC1, tBC1), 1)
    U_BC1 = modelv1(sample)
    # print(UBC2.shape)
    # U_BC1_x = torch.autograd.grad(U_BC1, xBC1, grad_outputs=torch.ones_like(U_BC1), create_graph=True)[0]
    loss_f = nn.MSELoss()
    loss_BC1 = loss_f(U_BC1 , UBC1)
    # print(loss_BC1)
    return loss_BC1

#loss for boundary condition right 
def Loss_BC2(xBC2,tBC2, UBC2):
    xBC2=torch.from_numpy(xBC2).float().to(device)
    tBC2=torch.from_numpy(tBC2).float().to(device)
    UBC2=torch.from_numpy(UBC2).float().to(device)
    # print(tBC2.shape)
    
    xBC2.requires_grad = True
    tBC2.requires_grad = True
    sample = torch.cat((xBC2, tBC2), 1)
    U_BC2 = modelv2(sample)

    # U_BC2_x = torch.autograd.grad(U_BC2, xBC2, grad_outputs=torch.ones_like(U_BC2), create_graph=True)[0]
    loss_f = nn.MSELoss()
    loss_BC2 = loss_f(U_BC2  , UBC2)
    # print(loss_BC2 )
    return loss_BC2



def Loss_IFC( xface,tface):
    xface=torch.from_numpy(xface).float().to(device)
    tface=torch.from_numpy(tface).float().to(device)
    # print(Ul_x.shape)
    # print(xface.shape)
    # print(tface.shape)
    xface.requires_grad = True
    tface.requires_grad = True
    sample = torch.cat((xface, tface), 1)
    Ur = modelv1(sample)
    Ul = modelv2(sample)
    Ur_x = torch.autograd.grad(Ur, xface, grad_outputs=torch.ones_like(Ur), create_graph=True)[0]
    Ul_x = torch.autograd.grad(Ul,xface, grad_outputs=torch.ones_like(Ul), create_graph=True)[0]
    Ur_t = torch.autograd.grad(Ur, tface, grad_outputs=torch.ones_like(Ur), create_graph=True)[0]
    Ul_t = torch.autograd.grad(Ul,tface, grad_outputs=torch.ones_like(Ul), create_graph=True)[0]
    U_exact_IFC= f_real(xface,tface)
    U_x_exact_IFC= f_real_x(xface,tface)
    
    u_avrag=(Ur+Ul)/2
    ux_avrag=(D1*Ur+D2*Ul)/2
    ut_avrag=(Ur_t+Ul_t)/2
    f1_source = -Ur+Ul
    f2_source =-D1* Ur_x+D2*Ul_x+(  D1-D2)*U_x_exact_IFC
    
    f3_source =-Ur_t+Ul_t
    loss_f1 =  nn.MSELoss(reduction ='mean')
    loss_f2 =  nn.MSELoss(reduction ='mean')
    loss_IFC = loss_f1(f1_source, torch.zeros_like(f1_source))
    loss_IFCN = loss_f2(f2_source, torch.zeros_like(f2_source))
    loss_IFCT = loss_f2(f3_source, torch.zeros_like(f2_source))
    lost_avara1=loss_f1(u_avrag  , Ur)
    lost_avara2=loss_f1(u_avrag  , Ul)
    lost_avara_x1=loss_f1(ux_avrag  ,D1* Ur_x)
    
    lost_avara_x2=loss_f1(ux_avrag  ,D2* Ul_x)
    lost_avara_t1=loss_f1(ut_avrag  ,Ur_t)
    lost_avara_t2=loss_f1(ut_avrag  , Ul_t)
    lost_U_exact_IFC=loss_f1(U_exact_IFC  , Ur)
    
    return loss_IFC,loss_IFCN,lost_avara1,lost_avara2,lost_avara_x1,lost_avara_x2,loss_IFCT,lost_avara_t1,lost_avara_t2


#%%
    #Initial Condition 0 =< x =<1 and t = 0
def Initial_Condition1(numInitialConditionPoints):
    X0IC1= np.linspace(0, 0.8, numInitialConditionPoints)[:,None]
    T0IC1 = np.zeros(numInitialConditionPoints)[:,None]
    UIC1 = np.sin(np.pi*X0IC1)
    return  X0IC1,T0IC1, UIC1

    #Initial Condition 1.5*10^-5=< x =<7*10^-5 and t = 0
def Initial_Condition2(numInitialConditionPoints):
    X0IC2= np.linspace( 0.8,1, numInitialConditionPoints)[:,None]
    T0IC2 = np.zeros(numInitialConditionPoints)[:,None]
    UIC2 = np.sin(np.pi*X0IC2)
    return X0IC2,T0IC2, UIC2

 

def Boundary_Condition1(numBoundaryConditionPoints): 
    #Boundary Condition x = 0 and 0 =< t =<1
    xBC1 = np.zeros(numBoundaryConditionPoints)[:,None]# x = -1                           
    tBC1= np.linspace(0, T, numBoundaryConditionPoints)[:,None] 
    
    UBC1 = np.zeros(numBoundaryConditionPoints)[:,None]
    return xBC1,tBC1, UBC1

def Boundary_Condition2(numBoundaryConditionPoints): 
    #Boundary Condition x = 7*10**-5 and 0 =< t =<1
    xBC2 = np.ones(numBoundaryConditionPoints)[:,None]# x = +1                           
    tBC2= np.linspace(0, T, numBoundaryConditionPoints)[:,None] 
    UBC2 = np.zeros(numBoundaryConditionPoints)[:,None]
    return xBC2,tBC2, UBC2

def interface(numinterfaceConditionPoints): 
    #Boundary Condition x = 1
    xface =0.8*np.ones(numinterfaceConditionPoints)[:,None]# x = +1                             
    tface= np.linspace(0, T, numinterfaceConditionPoints)[:,None]
    Inter_face=np.hstack((xface[:,0][:,None], tface[:,0][:,None]))
    return  xface,tface

#%%
def Datatraingcollocaton1(N_col_point1):
    X= lb1 + (ub1-lb1)*lhs(2,N_col_point1)
    x_col1=X[:,0][:,None]
    t_col1=X[:,1][:,None]
    return x_col1,t_col1

def Datatraingcollocaton2(N_col_point2):
    X= lb2 + (ub2-lb2)*lhs(2,N_col_point2)
    x_col2=X[:,0][:,None]
    t_col2=X[:,1][:,None]
    return x_col2,t_col2

#%%
batchsize =500
epochs  =500
# Domain bounds
lb1 = np.array([0, 0]) #lower bound
ub1 = np.array([0.8, T])  #upper bound
lb2 = np.array([0.8, 0]) #lower bound
ub2 = np.array([1, T])  #upper bound

N_col_point1=2000
N_col_point2=2000
numInitialConditionPoints=100
numBoundaryConditionPoints = 100
numinterfaceConditionPoints=200

x_col1,t_col1=Datatraingcollocaton1(N_col_point1)
x_col2,t_col2=Datatraingcollocaton2(N_col_point2)
x_col1 = torch.Tensor(x_col1).to(device)
t_col1=  torch.Tensor(t_col1).to(device)
x_col2 = torch.Tensor(x_col2).to(device)
t_col2=  torch.Tensor(t_col2).to(device)



print(x_col1.shape)
X0IC1,T0IC1, UIC1=Initial_Condition1(numInitialConditionPoints)
X0IC2,T0IC2, UIC2=Initial_Condition2(numInitialConditionPoints)
xBC1,tBC1, UBC1=Boundary_Condition1(numBoundaryConditionPoints)
xBC2,tBC2, UBC2=Boundary_Condition2(numBoundaryConditionPoints)
xface,tface=interface(numinterfaceConditionPoints)


#%%
learning_rate1=10**-4
learning_rate2=10**-4
# #optimizer1 = optim.Adam(modelv1.parameters(), lr=learning_rate1, betas = (0.9,0.99),eps = 10**-9)
# #optimizer2 = optim.Adam(modelv2.parameters(), lr=learning_rate2, betas = (0.9,0.99),eps = 10**-9)

# # Define optimizer
# #optimizer1 = optim.AdamW(modelv1.parameters(), lr=1e-4, weight_decay=1e-3)
# #optimizer2 = optim.AdamW(modelv2.parameters(), lr=1e-4, weight_decay=1e-3)
# # Define learning rate scheduler
# #scheduler = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.08)
# #scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.08)


# optimizer1 =torch.optim.AdamW(modelv1.parameters(), lr=learning_rate1, betas=(0.9, 0.99), eps=1e-20, weight_decay=1e-3)
# optimizer2 =torch.optim.AdamW(modelv2.parameters(), lr=learning_rate2,betas=(0.9, 0.99), eps=1e-20, weight_decay=1e-3)

# #scheduler = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.08)
# #scheduler = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.08)



# # # learning_rate=10**-3
# #optimizer1 = torch.optim.SGD(modelv1.parameters(), lr=learning_rate1)
# #optimizer2 = torch.optim.SGD(modelv2.parameters(), lr=learning_rate2)

# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer1, base_lr=learning_rate1, max_lr=0.001)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate2, max_lr=0.005)

pde1_losses = []
pde2_losses = []
bc_losses = []
l2_losses = []
#optimizer1 = torch.optim.LBFGS(modelv1.parameters(),lr=.1, max_iter=10000, max_eval=10000, tolerance_grad=1e-10, tolerance_change=1e-10, history_size=50, line_search_fn="strong_wolfe")
#optimizer2 = torch.optim.LBFGS(modelv2.parameters(),lr=.1, max_iter=10000, max_eval=10000, tolerance_grad=1e-10, tolerance_change=1e-10, history_size=50, line_search_fn="strong_wolfe")
optimizer1 = torch.optim.LBFGS(modelv1.parameters(), lr=1,  history_size=100, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
optimizer2 = torch.optim.LBFGS(modelv2.parameters(), lr=1,  history_size=100, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
#optimizer1 = torch.optim.LBFGS(modelv1.parameters(), lr=1.0, max_iter=500000, max_eval=500000,history_size=50,tolerance_grad=1e-5,tolerance_change=1.0 * np.finfo(float).eps,line_search_fn="strong_wolfe")
#optimizer2 = torch.optim.LBFGS(modelv2.parameters(), lr=1.0, max_iter=500000, max_eval=500000,history_size=50,tolerance_grad=1e-5,tolerance_change=1.0 * np.finfo(float).eps,line_search_fn="strong_wolfe",)
 
# optimizer1 = torch.optim.LBFGS(modelv1.parameters(), lr=1.0,
#             max_iter=500000,
#             max_eval=500000,
#             history_size=50,
#             tolerance_grad=1e-20,
#             tolerance_change=1e-20,
#             line_search_fn="strong_wolfe",)
# optimizer2 = torch.optim.LBFGS(modelv2.parameters(), lr=1.0,
#             max_iter=500000,
#             max_eval=500000,
#             history_size=50,
#             tolerance_grad=1e-20,
#             tolerance_change=1e-20,
#             line_search_fn="strong_wolfe",)


def train(x_col1,t_col1,x_col2,t_col2,X0IC1,T0IC1, UIC1,X0IC2,T0IC2, UIC2,xface,tface):
    for epoch in range(epochs):
        def closure():
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss_PDE1,loss_U_exact_PDE1=LossPDE1(x_col1,t_col1)
            loss_PDE2,loss_U_exact_PDE2=LossPDE2(x_col2,t_col2)
            loss_IC1,loss_Ul_exact=Loss_IC1(X0IC1,T0IC1, UIC1)
            
            loss_IC2=Loss_IC2(X0IC2,T0IC2, UIC2)
            loss_BC1 = Loss_BC1(xBC1,tBC1, UBC1)
            loss_BC2 = Loss_BC2(xBC2,tBC2, UBC2)
            loss_IFC,loss_IFCN,lost_avara1,lost_avara2,lost_avara_x1,lost_avara_x2,loss_IFCT,lost_avara_t1,lost_avara_t2=Loss_IFC( xface,tface)
            
            loss =loss_PDE1  +loss_PDE2+ loss_BC1+ loss_BC2+loss_IC1+loss_IC2+ loss_IFC+loss_U_exact_PDE1+loss_U_exact_PDE2+lost_avara1+lost_avara2+loss_IFCN
            
            loss.backward()
            
            return loss
        optimizer1.step(closure)
        optimizer2.step(closure)
        
        if epoch % 100 == 0:
            loss_PDE1,loss_U_exact_PDE1=LossPDE1(x_col1,t_col1)
            loss_PDE2,loss_U_exact_PDE2=LossPDE2(x_col2,t_col2)
            loss_IC1,loss_Ul_exact=Loss_IC1(X0IC1,T0IC1, UIC1)
            
            loss_IC2=Loss_IC2(X0IC2,T0IC2, UIC2)
            loss_BC1 = Loss_BC1(xBC1,tBC1, UBC1)
            loss_BC2 = Loss_BC2(xBC2,tBC2, UBC2)
            loss_IFC,loss_IFCN,lost_avara1,lost_avara2,lost_avara_x1,lost_avara_x2,loss_IFCT,lost_avara_t1,lost_avara_t2=Loss_IFC( xface,tface)
            loss =loss_PDE1  +loss_PDE2+ loss_BC1+ loss_BC2+loss_IC1+loss_IC2+loss_IFCN+ loss_IFC+loss_U_exact_PDE1+loss_U_exact_PDE2+lost_avara1+lost_avara2
            print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE1: {:.8f} \tLoss_PDE2: {:.8f} \tLoss_BC1: {:.10f} \tLoss_BC2: {:.10f} \tLoss_IC1: {:.10f} \tLoss_IC2: {:.10f}\tLoss_IFC: {:.10f}\tLoss_IFCN: {:.10f}'.format(epoch, loss.item(), loss_PDE1.item(),loss_PDE2.item(),loss_BC1.item(),loss_BC2.item(),loss_IC1.item(),loss_IC2.item(),loss_IFC.item(),loss_IFCN.item()))
        
train(x_col1,t_col1,x_col2,t_col2,X0IC1,T0IC1, UIC1,X0IC2,T0IC2, UIC2,xface,tface)    
    

    
    
    

     
    # if epoch % 100 == 0:
    #      print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE1: {:.8f} \tLoss_PDE2: {:.8f} \tLoss_BC1: {:.10f} \tLoss_BC2: {:.10f} \tLoss_IC1: {:.10f} \tLoss_IC2: {:.10f}\tLoss_IFC: {:.10f}\tLoss_IFCN: {:.10f}'.format(epoch, loss.item(), loss_PDE1.item(),loss_PDE2.item(),loss_BC1.item(),loss_BC2.item(),loss_IC1.item(),loss_IC2.item(),loss_IFC.item(),loss_IFCN.item()))
    # # # if epoch % 100 == 0:
    # #      print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE1: {:.8f} \tLoss_PDE2: {:.8f} \tLoss_BC1: {:.10f} \tLoss_BC2: {:.10f} \tLoss_IFC: {:.10f}'.format(epoch, loss.item(), loss_PDE1.item(),loss_PDE2.item(),loss_BC1.item(),loss_BC2.item(),loss_IFC.item()))
           
        # if epoch 

#saving and loading model
numPredictions = 50







        
tTest = np.array([0, 0.25, 0.5, 0.75, 1])
numPredictions = 100

for idx, t in enumerate(tTest):
    x1 = torch.linspace(0, 0.8, 50)[:, None]
    x2 = torch.linspace(0.8, 1, 50)[:, None]
    t1 = t * torch.ones(50)[:, None]
    t2 = t * torch.ones(50)[:, None]
    x1 = torch.Tensor(x1).to(device)
    t1 = torch.Tensor(t1).to(device)
    x2 = torch.Tensor(x2).to(device)
    t2 = torch.Tensor(t2).to(device)
    sample1 = torch.cat((x1, t1), 1)
    sample2 = torch.cat((x2, t2), 1)

    with torch.no_grad():
        u_pred1 = modelv1(sample1)
        u_pred2 = modelv2(sample2)

    u_pred = torch.cat([u_pred1, u_pred2])
    x = torch.cat([x1, x2])
    t=torch.cat([t1, t2])
    U_ext = f_real(x, t)
    
    u_pred = u_pred.cpu().detach().numpy()
    U_ext = U_ext.cpu().detach().numpy()
    x=x.cpu().detach().numpy()
    plt.figure(1)
    plt.plot(x, u_pred, '-b')
    plt.plot(x, u_pred, ':r')
    plt.savefig('vahid130.png')
    plt.show()
    
