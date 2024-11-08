

import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.utils.data import DataLoader, TensorDataset
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
R=7*10**-5
C_0=1000

#%%
L=55e-6
eps=0.6287064
r=5.0e-6

cs_max=43927 
soc_0=0.1738
soc_max=0.94680575
k=1.044546283497307e-11
F=96487
C_rate =0.1

cs=soc_0 *cs_max
A=1

Q= cs_max * A * L * eps * (soc_max-soc_0) * F
i_1C = Q/3600
a=3*eps/r
#29.242508064
I = 29.242508064 * C_rate/A
ampl=29.242508064*C_rate
J_coef=-1/(a*L*F); # modified
sum_A_anal = J_coef*ampl
# D1=1.142040936219013e-10
# D2=1.142040936219013e-10
# C_0=soc_0*cs_max
# delta=sum_A_anal*r/(D1*C_0)
Epsl_sep=0.70 # coefficient PDE1
Epsl_cat=0.70# coefficient PDE2
D1=1.95e-10
D2=1.95e-10
F=96487
D_ff1=D1*Epsl_sep**1.5
D_ff2=D2*Epsl_cat**1.5
t_0=0.47
L1=1-0.21428571428571427


#%%
class PINN:
    def __init__(self, layers):
        self.weights, self.biases, self.A = self.initialize_NN(layers)
        self.to_device()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        A = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = torch.nn.Parameter(self.xavier_init([layers[l], layers[l + 1]]))
            b = torch.nn.Parameter(torch.zeros(1, layers[l + 1], dtype=torch.float64))
            a = torch.nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
            weights.append(W)
            biases.append(b)
            A.append(a)
        return weights, biases, A

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = torch.sqrt(torch.tensor(2.0) / (in_dim + out_dim))
        return torch.randn(in_dim, out_dim, dtype=torch.float64) * xavier_stddev

    def neural_net_tanh(self, X):
        num_layers = len(self.weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(20 * self.A[l] * (H @ W + b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = H @ W + b
        return Y

    def to_device(self):
        # Move parameters to GPU if available
        for param in self.weights + self.biases + self.A:
            param.data = param.data.to(device)
            param.requires_grad = True



#%%
def LossPDE1(x, t):
    x = x.to(device)
    t = t.to(device)
    x.requires_grad = True
    t.requires_grad = True
    # print(x.shape)
    # print(t.shape)
    sample = torch.cat((x, t), 1)
    U1 = modelv1(sample)
    U1_t = torch.autograd.grad(U1, t, grad_outputs=torch.ones_like(U1), create_graph=True, only_inputs=True)[0]
    U1_x = torch.autograd.grad(U1, x, grad_outputs=torch.ones_like(U1), create_graph=True, only_inputs=True)[0]
    U1_xx = torch.autograd.grad(U1_x, x, grad_outputs=torch.ones_like(U1_x), create_graph=True, only_inputs=True)[0]
    f_source =U1_t  - (Epsl_sep**0.5)*U1_xx
    # print(f_source.shape)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_PDE1 = loss_f(f_source, torch.zeros_like(f_source))
    return loss_PDE1

def LossPDE2(x, t):
    x = x.to(device)
    t = t.to(device)
    
    x.requires_grad = True
    t.requires_grad = True
    
    sample = torch.cat((x, t), 1)
    U2 = modelv2(sample)
    # print(x.shape)
    # print(t.shape)
    U2_t = torch.autograd.grad(U2, t, grad_outputs=torch.ones_like(U2), create_graph=True, only_inputs=True)[0]
    U2_x = torch.autograd.grad(U2, x, grad_outputs=torch.ones_like(U2), create_graph=True, only_inputs=True)[0]
    U2_xx = torch.autograd.grad(U2_x, x, grad_outputs=torch.ones_like(U2_x), create_graph=True, only_inputs=True)[0]
    C_rate1=C_rate*np.ones(N_col_point2)[:,None] 
    C_rate1=torch.from_numpy(C_rate1).float().to(device)
    f_source = U2_t - (Epsl_cat**0.5)*U2_xx+(((1-t_0)*(29.242508064 * C_rate1/A)*R**2)/(D2*1000*Epsl_cat*L*F))
    
    # print(f_source.shape)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_PDE2 = loss_f(f_source, torch.zeros_like(f_source))
    # print(loss_PDE2)
    return loss_PDE2

#loss for initiale  codition left 
def Loss_IC1(X0IC1,T0IC1, UIC1):
    X0IC1=torch.from_numpy(X0IC1).float().to(device)
    T0IC1=torch.from_numpy(T0IC1).float().to(device)
    UIC1=torch.from_numpy(UIC1).float().to(device)
    # print(X0IC1.shape)
    # print(T0IC1.shape)
    sample = torch.cat((X0IC1, T0IC1), 1)
    U_IC1 = modelv1(sample)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_IC1 = loss_f(U_IC1,  UIC1)
    # print(loss_IC1)
    return loss_IC1

#loss for initioal codition left 
def Loss_IC2(X0IC2,T0IC2, UIC2):
    X0IC2=torch.from_numpy(X0IC2).float().to(device)
    T0IC2=torch.from_numpy(T0IC2).float().to(device)
    UIC2=torch.from_numpy(UIC2).float().to(device)
    
    # print(X0IC2.shape)
    # print(T0IC2.shape)
    sample = torch.cat((X0IC2, T0IC2), 1)
    U_IC2 = modelv2(sample)
    loss_f = nn.MSELoss(reduction ='mean')
    loss_IC2 = loss_f(U_IC2,  UIC2)
    # print(loss_IC2)
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
    U_BC1_x = torch.autograd.grad(U_BC1, xBC1, grad_outputs=torch.ones_like(U_BC1), create_graph=True)[0]
    loss_f = nn.MSELoss()
    loss_BC1 = loss_f(U_BC1_x , UBC1)
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
    # print(UBC2.shape)
    U_BC2_x = torch.autograd.grad(U_BC2, xBC2, grad_outputs=torch.ones_like(U_BC2), create_graph=True)[0]
    loss_f = nn.MSELoss()
    loss_BC2 = loss_f(U_BC2_x  , UBC2)
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
    f1_source = -Ur+Ul
    u_avrag=(Ur+Ul)/2
    f2_source =- Ur_x+Ul_x
    loss_f1 = nn.MSELoss(reduction ='mean')
    loss_f2 = nn.MSELoss(reduction ='mean')
    loss_IFC = loss_f1(f1_source, torch.zeros_like(f1_source))
    loss_IFCN = loss_f2(f2_source, torch.zeros_like(f2_source))
    lost_avara1=loss_f1(u_avrag  , Ur)
    lost_avara2=loss_f1(u_avrag  , Ul)
    
    return loss_IFC,loss_IFCN,lost_avara1,lost_avara2


#%%
    #Initial Condition 0 =< x =<15*10^-6 and t = 0
def Initial_Condition1(numInitialConditionPoints):
    X0IC1= np.linspace(0, 0.21428571428571427, numInitialConditionPoints)[:,None]
    T0IC1 = np.zeros(numInitialConditionPoints)[:,None]
    IC1=np.hstack((X0IC1[:,0][:,None], T0IC1[:,0][:,None])) #L1
    UIC1 = np.ones(numInitialConditionPoints)[:,None]
    return  X0IC1,T0IC1, UIC1

    #Initial Condition 1.5*10^-5=< x =<7*10^-5 and t = 0
def Initial_Condition2(numInitialConditionPoints):
    X0IC2= np.linspace( 0.21428571428571427,1, numInitialConditionPoints)[:,None]
    T0IC2 = np.zeros(numInitialConditionPoints)[:,None]
    IC2=np.hstack((X0IC2[:,0][:,None], T0IC2[:,0][:,None])) #L1
    UIC2 = np.ones(numInitialConditionPoints)[:,None]
    return X0IC2,T0IC2, UIC2

 

def Boundary_Condition1(numBoundaryConditionPoints): 
    #Boundary Condition x = 0 and 0 =< t =<1
    xBC1 = np.zeros(numBoundaryConditionPoints)[:,None]# x = -1                           
    tBC1= np.linspace(0, T, numBoundaryConditionPoints)[:,None] 
    BC1=np.hstack((xBC1[:,0][:,None], tBC1[:,0][:,None])) #L1
    UBC1 = ((-R*I*(1-t_0))/(1000*F*D1*Epsl_sep**1.5))*np.ones(numBoundaryConditionPoints)[:,None]
    return xBC1,tBC1, UBC1

def Boundary_Condition2(numBoundaryConditionPoints): 
    #Boundary Condition x = 7*10**-5 and 0 =< t =<1
    xBC2 = np.ones(numBoundaryConditionPoints)[:,None]# x = +1                           
    tBC2= np.linspace(0, T, numBoundaryConditionPoints)[:,None] 
    BC2=np.hstack((xBC2[:,0][:,None], tBC2[:,0][:,None])) #L1
    UBC2 = np.zeros(numBoundaryConditionPoints)[:,None]
    return xBC2,tBC2, UBC2

def interface(numinterfaceConditionPoints): 
    #Boundary Condition x = 1
    xface =0.21428571428571427*np.ones(numinterfaceConditionPoints)[:,None]# x = +1                             
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
batchsize =1000
epochs  =300
# Domain bounds
lb1 = np.array([0, 0]) #lower bound
ub1 = np.array([0.21428571428571425, T])  #upper bound
lb2 = np.array([0.21428571428571429, 0]) #lower bound
ub2 = np.array([1, T])  #upper bound

N_col_point1=200
N_col_point2=2500
numInitialConditionPoints=100
numBoundaryConditionPoints1 = 50
numBoundaryConditionPoints2 =50
numinterfaceConditionPoints=150






x_col1,t_col1=Datatraingcollocaton1(N_col_point1)
x_col2,t_col2=Datatraingcollocaton2(N_col_point2)
x_col1 = torch.Tensor(x_col1).to(device)
t_col1=  torch.Tensor(t_col1).to(device)
x_col2 = torch.Tensor(x_col2).to(device)
t_col2=  torch.Tensor(t_col2).to(device)


print(x_col1.shape)
X0IC1,T0IC1, UIC1=Initial_Condition1(numInitialConditionPoints)
X0IC2,T0IC2, UIC2=Initial_Condition2(numInitialConditionPoints)
xBC1,tBC1, UBC1=Boundary_Condition1(numBoundaryConditionPoints1)
xBC2,tBC2, UBC2=Boundary_Condition2(numBoundaryConditionPoints2)
xface,tface=interface(numinterfaceConditionPoints)

#%%
learning_rate1=10**-3
learning_rate2=10**-3
# optimizer1 = optim.Adam(modelv1.parameters(), lr=learning_rate1, betas = (0.9,0.999),eps = 10**-10)
# optimizer2 = optim.Adam(modelv2.parameters(), lr=learning_rate2, betas = (0.9,0.999),eps = 10**-10)


optimizer1 =torch.optim.Adam(modelv1.parameters(), lr=learning_rate1)
optimizer2 =torch.optim.Adam(modelv2.parameters(), lr=learning_rate2)
optimizer3 = torch.optim.LBFGS(modelv1.parameters(), lr=1,history_size=100, max_iter=20,line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
optimizer4 = torch.optim.LBFGS(modelv2.parameters(), lr=1,history_size=100, max_iter=20,line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
# optimizer1 = torch.optim.Adam(modelv1.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-02, weight_decay=0.001, amsgrad=False)
# optimizer2 = torch.optim.Adam(modelv2.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-02, weight_decay=0.001, amsgrad=False)
# optimizer1 = optim.Adam(modelv1.parameters(), lr=learning_rate1, eps = 10**-40)
# optimizer2 = optim.Adam(modelv2.parameters(), lr=learning_rate2,eps = 10**-40)


# learning_rate=10**-3
# optimizer1 = torch.optim.SGD(modelv1.parameters(), lr=learning_rate1)
# optimizer2 = torch.optim.SGD(modelv2.parameters(), lr=learning_rate2)
#optimizer = torch.optim.Adam(modelv2.parameters())
# optimizer = optim.Adam(modelv2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-30)
#optimizer = torch.optim.Adam(modelv2.parameters(), lr=0.01,betas=(0.9, 0.999), eps=1e-02, weight_decay=0.001, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# max_iter = 16600
def create_dataloader(x_col, t_col, batch_size):
    dataset = TensorDataset(x_col, t_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader






def train(x_col1,t_col1,x_col2,t_col2, X0IC1,T0IC1, UIC1,X0IC2,T0IC2, UIC2,xBC1,tBC1, UBC1,xBC2,tBC2, UBC2,xface,tface,epochs, modelv1,modelv2, optimizer1,optimizer2,optimizer3,optimizer4):
    # Create dataloaders for mini-batch training
    # dataloader1 = create_dataloader(x_col1, t_col1, batch_size)
    # dataloader2 = create_dataloader(x_col2, t_col2, batch_size)

    for epoch in range(epochs):
        # for (x_batch1, t_batch1), (x_batch2, t_batch2) in zip(dataloader1, dataloader2):
            def closure():
                if epoch <700:
                    optimizer3.zero_grad()
                    optimizer4.zero_grad()
                else: 
                    optimizer3.zero_grad()
                    optimizer4.zero_grad()   
                       
                loss_PDE1=LossPDE1(x_col1,t_col1)
                loss_PDE2=LossPDE2(x_col2,t_col2)
                loss_IC1=Loss_IC1(X0IC1,T0IC1, UIC1)
                loss_IC2=Loss_IC2(X0IC2,T0IC2, UIC2)
                loss_BC1 = Loss_BC1(xBC1,tBC1, UBC1)
                loss_BC2 = Loss_BC2(xBC2,tBC2, UBC2)
                loss_IFC,loss_IFCN,lost_avara1,lost_avara2=Loss_IFC( xface,tface)
            
                loss =loss_PDE1  +loss_PDE2+ loss_BC1+ loss_BC2+loss_IC1+loss_IC2+(loss_IFC+loss_IFCN)+lost_avara1+lost_avara2
                # loss = loss_PDE1  +loss_PDE2+loss_BC1+loss_BC2+loss_IFC+loss_IFCN
                
                loss.backward()
                return loss
              # Update parameters W, B
                
    
            if epoch <700:
                optimizer3.step(closure)
                optimizer4.step(closure)
            else: 
                optimizer3.step(closure)
                optimizer4.step(closure)   
    
                     
            if epoch % 100 == 0:
                loss_PDE1=LossPDE1(x_col1,t_col1)
                loss_PDE2=LossPDE2(x_col2,t_col2)
                loss_IC1=Loss_IC1(X0IC1,T0IC1, UIC1)
                loss_IC2=Loss_IC2(X0IC2,T0IC2, UIC2)
                loss_BC1 = Loss_BC1(xBC1,tBC1, UBC1)
                loss_BC2 = Loss_BC2(xBC2,tBC2, UBC2)
                loss_IFC,loss_IFCN,lost_avara1,lost_avara2=Loss_IFC( xface,tface)
            
                loss =loss_PDE1  +loss_PDE2+ loss_BC1+ loss_BC2+loss_IC1+loss_IC2+(loss_IFC+loss_IFCN)+lost_avara1+lost_avara2
                print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE1: {:.8f} \tLoss_PDE2: {:.8f} \tLoss_BC1: {:.10f} \tLoss_BC2: {:.10f} \tLoss_IC1: {:.10f} \tLoss_IC2: {:.10f}\tLoss_IFC: {:.10f}\tLoss_IFCN: {:.10f}'.format(epoch, loss.item(), loss_PDE1.item(),loss_PDE2.item(),loss_BC1.item(),loss_BC2.item(),loss_IC1.item(),loss_IC2.item(),loss_IFC.item(),loss_IFCN.item()))
            # # if epoch % 100 == 0:
                #      print('Train Epoch: {} \tLoss: {:.10f} \tLoss_PDE1: {:.8f} \tLoss_PDE2: {:.8f} \tLoss_BC1: {:.10f} \tLoss_BC2: {:.10f} \tLoss_IFC: {:.10f}'.format(epoch, loss.item(), loss_PDE1.item(),loss_PDE2.item(),loss_BC1.item(),loss_BC2.item(),loss_IFC.item()))
                       
batch_size = 2000  # Set your mini-batch size            # if epoch 
train(x_col1,t_col1,x_col2,t_col2, X0IC1,T0IC1, UIC1,X0IC2,T0IC2, UIC2,xBC1,tBC1, UBC1,xBC2,tBC2, UBC2,xface,tface,epochs, modelv1,modelv2, optimizer1,optimizer2,optimizer3,optimizer4)


#saving and loading model
# numPredictions = 50







        

#%%
import numpy as np
import math as m
from scipy import interpolate
from numpy.polynomial import polynomial as P
import matplotlib.pyplot as plt

T           = 298.15
c0          = 1000
constProp    = False
ce_ref   = 1000
burgCoeff_a  = 1.5
burgCoeff_s  = 1.5
burgCoeff_c  = 1.5
eps_s = 0.7
eps_c=0.7 #0.05 additives
raw_sige     = np.array([[0, 10000], [0.927, 0.927]])  # f(t,c)
raw_dfdlnc   = np.array([[0, 10000], [2.0, 2.0]])  # f(t,c)
raw_De       = np.array([[0, 10000], [1.95e-10, 1.95e-10]])  # f(t,c)
raw_tplus    = np.array([[0, 10000], [0.47, 0.47]])  # f(t,c)
c0=1000
SOC_init= 1.0
L_c= 55e-6 #L_cat
L_s= 15e-6 # L_sep
N_a=1
N_s=30 # n_sep =at least 2 points!n_sep
N_c=30 #n_cat
Nr=10#nr
R=8.31446262
F=96487
N_series=30
A = 1# cell.A
d1 =L_s # cell.L_s
d2 = L_c #cell.L_c
C0 = c0#cell.c0
i_1C =29.242508064
C_rate=0.1
dt0=100
ampl=C_rate* i_1C 
N_anal_sol = 30
N_anal_liq = 30
mesh=np.linspace(0, d1+d2,1000)
x_liq=np.linspace(0, d1+d2,1000)
ce = [c0*len(x_liq)]
N_t=N_a+N_s+N_c- 2
c=np.full((N_t),c0)
N_layers = 2
#%%


def signal_val(t):
   
    return  ampl
def signal_Lap_inv(A, t):

    if A == 0:
      inv = ampl*t
    else:
      inv = - ampl/A * (m.exp(-A*t)-1)
    return inv
#%%################################################################################################################



def updateProp_halfCell1( mesh, c, T):
    # Assume mesh is now a NumPy array with elements corresponding to ['N_s', 'N_t']
   

    # Convert concentration c to NumPy array
    c = np.array(c)
    c_mid = np.append(np.append(c[0], 0.5 * (c[0:-1] + c[1:])), c[-1])

    # Interpolate properties using provided concentration data
    tplus = np.interp(c, raw_tplus[0], raw_tplus[1])
    dfdlnc = np.interp(c_mid, raw_dfdlnc[0], raw_dfdlnc[1])

    De_ef_s = np.interp(c_mid[0:N_s], raw_De[0], raw_De[1]) * pow(eps_s, burgCoeff_s)
    De_ef_c = np.interp(c_mid[N_s:], raw_De[0], raw_De[1]) * pow(eps_c, burgCoeff_c)
    De_ef = np.append(De_ef_s, De_ef_c)

    sige_ef_s = np.interp(c_mid[0:N_s], raw_sige[0], raw_sige[1]) * pow(eps_s, burgCoeff_s)
    sige_ef_c = np.interp(c_mid[N_s:], raw_sige[0], raw_sige[1]) * pow(eps_c, burgCoeff_c)
    sige_ef = np.append(sige_ef_s, sige_ef_c)

    # Update properties in the class instance
    tplus = tplus
    De_ef = De_ef
    sige_ef = sige_ef
    dfdlnc = dfdlnc
    kD_ef = (2 * R * T * sige_ef / F) * (1 + dfdlnc) * (np.append(np.append(tplus[0], 0.5 * (tplus[0:-1] + tplus[1:])), tplus[-1]) - 1)
    return tplus,De_ef,sige_ef,dfdlnc,kD_ef

tplus,De_ef,sige_ef,dfdlnc,kD_ef= updateProp_halfCell1(mesh, c, T)

#%%###############################################################################################################
def solveDiff_anal_initialize_halfCell1():
    

    D = np.array([De_ef[0], De_ef[-1]])

    # Calculate coefficient g0
    g0_coef = - (1 - tplus[0]) / (A * F * De_ef[0])

    # Initialize roots array
    roots = np.zeros((2, N_series))

    # Define f2 function for finding roots
    f2 = lambda x: (
        m.sin(x * (d1 / np.sqrt(D[0] / eps_s) + d2 / np.sqrt(D[1] / eps_c)))
        + (1 - eps_c / eps_s * np.sqrt((D[1] / eps_c) / (D[0] / eps_s)))
        / (1 + eps_c / eps_s * np.sqrt((D[1] / eps_c) / (D[0] / eps_s)))
        * m.sin(x * (d1 / np.sqrt(D[0] / eps_s) - d2 / np.sqrt(D[1] / eps_c)))
    )

    # Finding roots using Newton's method
    dx_new = 2 * m.pi / (d1 / np.sqrt(D[0] / eps_s) + d2 / np.sqrt(D[1] / eps_c)) / 8
    tol_newton = 1e-7
    x0 = -1e-10
    n = 0

    while n < N_series:
        x1 = x0 + dx_new
        if f2(x1) * f2(x0) <= 0:
            X1, X0 = x1, x0
            while True:
                x_new = 0.5 * (X0 + X1)
                y_new = f2(x_new)
                if abs(y_new) < tol_newton:
                    roots[0, n] = x_new
                    roots[1, n] = y_new
                    n += 1
                    break
                elif y_new * f2(X0) < 0:
                    X1 = x_new
                else:
                    X0 = x_new
        x0 = x1

    # Calculate initial values
    g00 = g0_coef * signal_val(0)
    q2_coef = D[0] / eps_c / d2

    # Initialize arrays for coefficients
    a = np.zeros((N_layers, N_series))
    b = np.zeros((N_layers, N_series))
    c = np.zeros(N_series)
    Ac = np.zeros(N_series)
    Bc = np.zeros(N_series)
    A1 = np.zeros(N_series)
    A2 = np.zeros(N_series)
    
    # Loop to calculate A1, A2, a, b, c, Ac, and Bc
    
    for n in range(N_series):
        A1[n] =  roots[0, n] / np.sqrt(D[0] / eps_s)
        # print(f"Starting root-finding loop, n: {n}")

        A2[n] =  roots[0, n] / np.sqrt(D[1] /eps_c)
         
        a[0, n] = 0
        b[0, n] = 1
        a[1, n] = -eps_s / eps_c * np.sqrt((D[0] /eps_s) / (D[1] / eps_c)) * m.sin( A1[n] * d1)
        b[1, n] = m.cos( A1[n] * d1)

        if n == 0:
            coef_1a = eps_s * (b[0, n] * d1)
            coef_1b = eps_s * ((b[0, n] * d1 ** 2) / 6)
            coef_2a = eps_c * (b[1, n] * d2)
            int3 = eps_s * (b[0, n] ** 2 * d1)
            int4 = eps_c * (b[1, n] ** 2 * d2)
        else:
            coef_1a = eps_s * ( b[0, n] /  A1[n] * m.sin( A1[n] * d1))
            coef_1b = eps_s * ( b[0, n] / d1 / ( A1[n] ** 3) * (d1 *  A1[n] - m.sin( A1[n] * d1)))
            coef_2a = eps_c * ( a[1, n] /  A2[n] * (1 - m.cos( A2[n] * d2)) +  b[1, n] /  A2[n] * m.sin( A2[n] * d2))
            int3 = eps_s * ( b[0, n] ** 2 * (d1 / 2 + m.sin(2 *  A1[n] * d1) / (4 *  A1[n])))
            int4 = eps_c * (1 / A2[n] * ( A2[n] * d2 / 2 * ( a[1, n] ** 2 +  b[1, n] ** 2) +
                                              ( b[1, n] ** 2 -  a[1, n] ** 2) / 4 * m.sin(2 * A2[n] * d2) +
                                              a[1, n] *  b[1, n] * (m.sin( A2[n] * d2) ** 2)))

        c[n] = (C0 * coef_1a +  g00 * coef_1b + C0 * coef_2a) / (int3 + int4)
        Ac[n] = (-D[0] / eps_s / d1 * coef_1a + q2_coef * coef_2a) / (int3 + int4)
        Bc[n] = (1 * coef_1b) / (int3 + int4)
        

    return roots,g00,A1,A2,a,b,c,Ac,Bc

roots,g00,A1,A2,a,b,c,Ac,Bc= solveDiff_anal_initialize_halfCell1()
        
#%%
def solveDiff_anal_halfCell1( t):
    l1 = d1
    l2 = d1 + d2
    g0_coef = - (1 - tplus[0]) / (A * F * De_ef[0])

    # Split x_nodes into x0 and x1 using numpy array indexing
    x_nodes = np.array(x_liq)
    x0 = x_nodes[x_nodes <= l1 * (1 + 1e-10)]
    x1 = x_nodes[(x_nodes >= l1 * (1 + 1e-10)) & (x_nodes <= l2 * (1 + 1e-20))]

    # Initialize arrays for v0_sum, v1_sum, u0_val, u1_val, and v_all_coef
    v0_sum = np.zeros(len(x0))
    v1_sum = np.zeros(len(x1))
    u0_val = np.zeros(len(x0))
    u1_val = np.zeros(len(x1))
    v_all_coef = np.zeros(N_series)
    u0_val_av_cat = 0

    # Calculate g0
    g0 = g0_coef * signal_val(t)

    # Unpack self.anal (now an np.array) for readability
    # self.roots,  self.g00,  self.A1,  self.A2,  self.a,  self.b,  self.c,  self.Ac,  self.Bc = self.anal1

    # Calculate v_all_coef and u0_val_av_cat
    for n in range(N_series):
        
        integral = g0_coef * signal_Lap_inv(roots[0, n] ** 2, t)
        v_all_coef[n] = (Bc[n] * g0 + (-Bc[n] * g00 + c[n]) * m.exp(-t * roots[0, n] ** 2) +
                          (Ac[n] - Bc[n] * roots[0, n] ** 2) * integral)
        u0_val_av_cat += (v_all_coef[n] / d2 / A2[n] *(a[1, n] * (1 - m.cos(A2[n] * d2)) + b[1, n] * m.sin(A2[n] * d2)))

    # Calculate u0_val
    for i in range(len(x0)):
        for n in range(N_series):
            phi0_n = b[0, n] * m.cos(A1[n] * x0[i])
            v0_sum[i] += phi0_n * v_all_coef[n]
        u0_val[i] = (-g0 / (2 * l1) * x0[i] ** 2 + g0 * x0[i] - g0 * l1 / 2) + v0_sum[i]

    # Calculate u1_val
    for i in range(len(x1)):
        for n in range(N_series):
            phi1_n = a[1, n] * m.sin(A2[n] * (x1[i] - l1)) +b[1, n] * m.cos(A2[n] * (x1[i] - l1))
            v1_sum[i] += phi1_n * v_all_coef[n]
        u1_val[i] = v1_sum[i]

    u0_val_av_an = u0_val[0]

    return np.append(u0_val, u1_val), u0_val_av_an, u0_val_av_cat


            
#%%

for nt  in range(100):

        dt = dt0 # constant time steps for now!
        t = nt*dt
        # print(t)
        ce, ce_av_an, ce_av_cat = solveDiff_anal_halfCell1(t) #Cathode half-cell
        
        
        
        # print(f"t:{t},x_liq:{x_liq},cell.L_s:{cell.L_s},N_anal_liq:{N_anal_liq}")
        
        nt=nt+1

print('\n----------------------- Done! ----------------------')
plt.show()
tTest = np.array( [ 1])
numPredictions = 1000

# Start a new figure
plt.figure(1)

for idx, t in enumerate(tTest):

    x1 = np.linspace(0, 0.21428571428571427, 200)[:, None]
    x2 = np.linspace(0.21428571428571427, 1, 800)[:, None]
    t1 = t * np.ones(200)[:, None]
    t2 = t * np.ones(800)[:, None]
    XT1 = np.hstack((x1, t1))
    
    sample1 = torch.from_numpy(XT1).float().to(device)
    XT2 = np.hstack((x2, t2))
    sample2 = torch.from_numpy(XT2).float().to(device)    

    with torch.no_grad():
        u_pred1 = modelv1(sample1)
        u_pred2 = modelv2(sample2)

    u_pred1 = u_pred1.cpu().detach().numpy()
    u_pred2 = u_pred2.cpu().detach().numpy()
    u_pred = np.vstack([u_pred1, u_pred2])
    x = np.vstack([x1, x2])
    
    # Plotting on the same figure
    plt.plot(x * 7e-6, 1000 * u_pred, label=f't={t}')
    plt.plot(x * 7e-6,ce)
# Add legend and labels
plt.legend()
plt.xlabel('x * 7e-6')
plt.ylabel('1000 * u_pred')

# Save the final figure
plt.savefig('all_predictions.png')
plt.show()


