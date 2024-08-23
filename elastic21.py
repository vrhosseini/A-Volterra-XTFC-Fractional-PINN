import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F
from pyDOE import lhs

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 


# Constants
lam = 1
mu = 0.5
Q = 4

nu=20
N_col_point=5000
def Datatraingcollocaton(N_col_point):
    X= lb + (ub-lb)*lhs(2,N_col_point)
    x_col=X[:,0]
    y_col=X[:,1]
    
    return x_col,y_col


# Domain bounds
lb = np.array([0, 0]) #lower bound
ub = np.array([1, 1])  #upper bound
x_col,y_col=Datatraingcollocaton(N_col_point)
x_col=torch.Tensor(x_col[:,None] ).to(device)
y_col=torch.Tensor(y_col[:,None] ).to(device)
xy_points = torch.cat((x_col, y_col),  1)

# Convert xy_points to a PyTorch tensor
xy_points = torch.tensor(xy_points, requires_grad=True)
#%%
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Create boundary coordinates using PyTorch

left_coord_cpu = torch.cat([torch.zeros([nu, 1], dtype=torch.float32), 
                            torch.linspace(y_min, y_max, nu).reshape(-1, 1)], dim=1)
bottom_coord_cpu = torch.cat([torch.linspace(x_min, x_max, nu).reshape(-1, 1), 
                              torch.zeros([nu, 1], dtype=torch.float32)], dim=1)
right_coord_cpu = torch.cat([torch.ones([nu, 1], dtype=torch.float32), 
                             torch.linspace(y_min, y_max, nu).reshape(-1, 1)], dim=1)
top_coord_cpu = torch.cat([torch.linspace(x_min, x_max, nu).reshape(-1, 1), 
                           torch.ones([nu, 1], dtype=torch.float32)], dim=1)

# Move tensors to the specified device (either CPU or GPU)
left_coord = left_coord_cpu.to(device)
bottom_coord = bottom_coord_cpu.to(device)
right_coord = right_coord_cpu.to(device)
top_coord = top_coord_cpu.to(device)

# Store the coordinates in a dictionary
xy_boundary = {
    "left_coord": left_coord,
    "bottom_coord": bottom_coord,
    "right_coord": right_coord,
    "top_coord": top_coord
}

# Example of converting the boundary coordinates to torch variables with gradients enabled (if needed)
xy_boundary = {key: torch.autograd.Variable(value, requires_grad=True) for key, value in xy_boundary.items()}








class Model(nn.Module):
    def __init__(self, num_hidden_layers, num_units, activation):
        super(Model, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.activation = getattr(F, activation)
        
        # Input layers
        self.input_layer_1 = nn.Linear(2, num_units)
        self.input_layer_2 = nn.Linear(2, num_units)
        
        # Hidden layers
        self.hidden_layers_1 = nn.ModuleList([
            nn.Linear(num_units, num_units) for _ in range(num_hidden_layers)
        ])
        self.hidden_layers_2 = nn.ModuleList([
            nn.Linear(num_units, num_units) for _ in range(num_hidden_layers)
        ])
        
        # Output layers
        self.u_output = nn.Linear(num_units, 1)
        self.v_output = nn.Linear(num_units, 1)

        # Initialize weights using glorot normal
        self._initialize_weights()

    def _initialize_weights(self):
        def glorot_normal_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(glorot_normal_init)
    
    def forward(self, x):
        x1 = self.activation(self.input_layer_1(x))
        x2 = self.activation(self.input_layer_2(x))
        
        for i in range(self.num_hidden_layers):
            x1 = self.activation(self.hidden_layers_1[i](x1))
            x2 = self.activation(self.hidden_layers_2[i](x2))
        
        u = self.u_output(x1)
        v = self.v_output(x2)
        return u, v

# Example usage:

def loss_PDE(model, xy_points, lam, mu, Q):
    xy_points = xy_points.requires_grad_(True)
    
    u, v = model(xy_points)
    
    # Compute gradients with respect to inputs
    du = torch.autograd.grad(outputs=u, inputs=xy_points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(outputs=v, inputs=xy_points, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    du_dx, du_dy = du[:, 0], du[:, 1]
    dv_dx, dv_dy = dv[:, 0], dv[:, 1]
    
    # Compute second derivatives
    duu_dxx = torch.autograd.grad(du_dx, xy_points, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0]
    duu_dxy = torch.autograd.grad(du_dx, xy_points, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 1]
    duu_dyy = torch.autograd.grad(du_dy, xy_points, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1]
    dvv_dxx = torch.autograd.grad(dv_dx, xy_points, grad_outputs=torch.ones_like(dv_dx), create_graph=True)[0][:, 0]
    dvv_dxy = torch.autograd.grad(dv_dy, xy_points, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 0]
    dvv_dyy = torch.autograd.grad(dv_dy, xy_points, grad_outputs=torch.ones_like(dv_dy), create_graph=True)[0][:, 1]
    
    Fx = lam * (4*(torch.pi**2)*torch.cos(2*torch.pi*xy_points[:, 0])*torch.sin(torch.pi*xy_points[:, 1]) - torch.pi*torch.cos(torch.pi*xy_points[:, 0])*Q*xy_points[:, 1]**3 ) + mu * (9*(torch.pi**2)*torch.cos(2*torch.pi*xy_points[:, 0])*torch.sin(torch.pi*xy_points[:, 1]) - torch.pi*torch.cos(torch.pi*xy_points[:, 0])*Q*xy_points[:, 1]**3)
    
    Fy = lam*(-3*torch.sin(torch.pi*xy_points[:, 0])*Q*xy_points[:, 1]**2 + 2*(torch.pi**2)*torch.sin(2*torch.pi*xy_points[:, 0])*torch.cos(torch.pi*xy_points[:, 1])) + mu*(-6*torch.sin(torch.pi*xy_points[:, 0])*Q*xy_points[:, 1]**2 + 2*(torch.pi**2)*torch.sin(2*torch.pi*xy_points[:, 0])*torch.cos(torch.pi*xy_points[:, 1]) + (torch.pi**2)*torch.sin(torch.pi*xy_points[:, 0])*Q*(xy_points[:, 1]**4)/4)

    loss1 = mu * (duu_dxx + duu_dyy) + (mu + lam) * (duu_dxx + dvv_dxy) + Fx
    loss2 = mu * (dvv_dxx + dvv_dyy) + (mu + lam) * (dvv_dyy + duu_dxy) + Fy
    
    loss_pde = torch.mean(loss1**2) + torch.mean(loss2**2)
    
    return loss_pde
# Displacement and Stress Loss Functions

def bottom_dispLoss(model, xy_boundary):
    bottom_coord = xy_boundary["bottom_coord"]
    
    u, v = model(bottom_coord)
    return torch.mean(u ** 2) + torch.mean(v ** 2)

def top_dispLoss(model, xy_boundary):
    top_coord = xy_boundary["top_coord"]
    u, _ = model(top_coord)
    return torch.mean(u ** 2)

def left_dispLoss(model, xy_boundary):
    left_coord = xy_boundary["left_coord"]
    _, v = model(left_coord)
    return torch.mean(v ** 2)

def right_dispLoss(model, xy_boundary):
    right_coord = xy_boundary["right_coord"]
    _, v = model(right_coord)
    return torch.mean(v ** 2)

def left_stressLoss(model, xy_boundary):
    left_coord = xy_boundary["left_coord"]
    left_coord.requires_grad_(True)
    u, v = model(left_coord)
    
    du = torch.autograd.grad(outputs=u, inputs=left_coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(outputs=v, inputs=left_coord, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    du_dx = du[:, 0]
    dv_dy = dv[:, 1]
    
    sx = lam * (du_dx + dv_dy) + 2 * mu * du_dx
    return torch.mean(sx ** 2)

def right_stressLoss(model, xy_boundary):
    right_coord = xy_boundary["right_coord"]
    right_coord.requires_grad_(True)
    u, v = model(right_coord)
    
    du = torch.autograd.grad(outputs=u, inputs=right_coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(outputs=v, inputs=right_coord, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    du_dx = du[:, 0]
    dv_dy = dv[:, 1]
    
    sx = lam * (du_dx + dv_dy) + 2 * mu * du_dx
    return torch.mean(sx ** 2)

def top_stressLoss(model, xy_boundary):
    top_coord = xy_boundary["top_coord"]
    top_coord.requires_grad_(True)
    u, v = model(top_coord)
    
    du = torch.autograd.grad(outputs=u, inputs=top_coord, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(outputs=v, inputs=top_coord, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    du_dx = du[:, 0]
    dv_dy = dv[:, 1]
    
    sy = lam * (du_dx + dv_dy) + 2 * mu * dv_dy
    sy_top = (lam + 2 * mu) * Q * torch.sin(torch.pi * top_coord[:, 0])
    
    return torch.mean((sy - sy_top) ** 2)

# Boundary Condition Loss
def loss_BC(model, xy_boundary):
    loss1 = bottom_dispLoss(model, xy_boundary)
    loss2 = top_dispLoss(model, xy_boundary)
    loss3 = left_dispLoss(model, xy_boundary)
    loss4 = right_dispLoss(model, xy_boundary)
    loss5 = left_stressLoss(model, xy_boundary)
    loss6 = right_stressLoss(model, xy_boundary)
    loss7 = top_stressLoss(model, xy_boundary)
    
    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

# Total Loss
def Loss(model, xy_points, xy_boundary):
    loss1 = loss_PDE(model, xy_points)
    loss2 = loss_BC(model, xy_boundary)
    loss = loss1 + loss2
    
    return loss, loss1, loss2

# Define the custom loss function (this should be implemented based on your specific PDE problem)
def custom_loss(model, xy_points,xy_boundary):
    loss_pde = loss_PDE(model, xy_points, lam, mu, Q)  # Assuming you already defined loss_PDE
    loss_bc =  loss_BC(model, xy_boundary)  # This should be implemented
    total_loss = loss_pde+loss_bc 
    return total_loss, loss_pde,loss_bc

# Define the optimizer step
def run_optimizer(model, optimizer1, optimizer2, xy_points, xy_boundary, epoch):
    def closure():
        # Choose the optimizer based on the epoch count
        if epoch < 4600:
            optimizer1.zero_grad()
        else:
            optimizer2.zero_grad()
        
        # Forward pass and compute loss
        loss_val, loss_pde, loss_bc = custom_loss(model, xy_points, xy_boundary)
        
        # Backward pass
        loss_val.backward()
        
        return loss_val

    # Perform optimization step based on the epoch count
    if epoch < 4600:
        optimizer1.step(closure)
    else:
        optimizer2.step(closure)

    # Recalculate loss after optimization step for logging
    
    loss_val, loss_pde, loss_bc = custom_loss(model, xy_points, xy_boundary)

    return loss_val.item(), loss_pde.item(), loss_bc.item()

# Define the training loop
def train(xy_points, xy_boundary, epochs):
    # Initialize the model and optimizers
    model = Model(num_hidden_layers=5, num_units=64, activation='tanh')
    optimizer1 = optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.99), eps=1e-40)
    optimizer2 = torch.optim.LBFGS(model.parameters(), lr=1, history_size=100, 
                                   line_search_fn="strong_wolfe", tolerance_grad=1e-32, 
                                   tolerance_change=1e-32)

    # Lists to store loss values for each epoch
    losses = []
    loss_pde_vals = []
    loss_bc_vals = []

    for epoch in range(epochs):
        # Run the optimizer
        loss, loss_pde_val, loss_bc_val = run_optimizer(model, optimizer1, optimizer2, xy_points, xy_boundary, epoch)
        
        # Log the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Train Loss: {loss}")
        
        # Store the losses
        losses.append(loss)
        loss_pde_vals.append(loss_pde_val)
        loss_bc_vals.append(loss_bc_val)
    
    # Return the collected loss values and the trained model
    LossVal = {"losses": losses, "loss_pde": loss_pde_vals, "loss_bc": loss_bc_vals}
    return LossVal, model

# Assuming xy_points and xy_boundary have been defined as PyTorch tensors
LossVal, model = train(xy_points, xy_boundary, epochs=3000)

#saving and loading model
torch.save(model.state_dict(),'model_wight.pth')

model.load_state_dict(torch.load('model_wight.pth'))
model.eval()


# Constants
lam = 1
mu = 0.5
Q = 4
nx = 30  # number of points in x-direction
ny =30  # number of points in x-direction
nu=20
# Domain boundaries
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# PyTorch tensors for domain boundaries
lower_bound = torch.tensor([x_min, y_min], dtype=torch.float32)
upper_bound = torch.tensor([x_max, y_max], dtype=torch.float32)

# Generate coordinates
x_coord = torch.linspace(x_min, x_max, nx).reshape(-1, 1)
y_coord = torch.linspace(y_min, y_max, ny).reshape(-1, 1)

# Create meshgrid and flatten the coordinates into points
Grid = torch.meshgrid(x_coord.squeeze(), y_coord.squeeze(), indexing='ij')
xy_points = torch.cat([Grid[0].reshape(-1, 1), Grid[1].reshape(-1, 1)], dim=1)

# Convert xy_points to a PyTorch tensor
xy_points = torch.tensor(xy_points, requires_grad=True)
#%%
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Create boundary coordinates using PyTorch
left_coord = torch.cat([torch.zeros([nu, 1], dtype=torch.float32), 
                        torch.linspace(y_min, y_max, nu).reshape(-1, 1)], dim=1)
bottom_coord = torch.cat([torch.linspace(x_min, x_max, nu).reshape(-1, 1), 
                          torch.zeros([nu, 1], dtype=torch.float32)], dim=1)
right_coord = torch.cat([torch.ones([nu, 1], dtype=torch.float32), 
                         torch.linspace(y_min, y_max, nu).reshape(-1, 1)], dim=1)
top_coord = torch.cat([torch.linspace(x_min, x_max, nu).reshape(-1, 1), 
                       torch.ones([nu, 1], dtype=torch.float32)], dim=1)

# Store the coordinates in a dictionary
xy_boundary = {
    "left_coord": left_coord,
    "bottom_coord": bottom_coord,
    "right_coord": right_coord,
    "top_coord": top_coord
}

# Example of converting the boundary coordinates to torch variables with gradients enabled (if needed)
xy_boundary = {key: torch.autograd.Variable(value, requires_grad=True) for key, value in xy_boundary.items()}






def loss_PDE(model, xy_points):
    xy_points.requires_grad_(True)
    u, v = model(xy_points)
    
    # Gradients of deformation in x and y direction
    du = torch.autograd.grad(outputs=u, inputs=xy_points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(outputs=v, inputs=xy_points, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    du_dx, du_dy = du[:, 0], du[:, 1]
    dv_dx, dv_dy = dv[:, 0], dv[:, 1]
    
    e_x = du_dx
    e_y = dv_dy
    e_xy = (du_dy + dv_dx) / 2
    
    s_x = lam * (e_x + e_y) + 2 * mu * e_x
    s_y = lam * (e_x + e_y) + 2 * mu * e_y
    s_xy = 2 * mu * e_xy
    
    return s_x, s_y, s_xy, u, v

# Example usage
s_x, s_y, s_xy, u, v = loss_PDE(model, xy_points)
x1 = xy_points[:, 0]
y1 = xy_points[:, 1]

# Reshape arrays for plotting
arr_x1 = x1.detach().cpu().numpy().reshape(nx, ny)
arr_y1 = y1.detach().cpu().numpy().reshape(nx, ny)
arr_u = u.detach().cpu().numpy().reshape(nx, ny)
arr_v = v.detach().cpu().numpy().reshape(nx, ny)
arr_sx = s_x.detach().cpu().numpy().reshape(nx, ny)
arr_sy = s_y.detach().cpu().numpy().reshape(nx, ny)
arr_sxy = s_xy.detach().cpu().numpy().reshape(nx, ny)

# Exact displacement and stress
dispX = torch.cos(2 * np.pi * x1) * torch.sin(np.pi * y1)
dispY = torch.sin(np.pi * x1) * Q * (y1 ** 4) / 4

dispX_dx = -2 * np.pi * torch.sin(2 * np.pi * x1) * torch.sin(np.pi * y1)
dispY_dy = torch.sin(np.pi * x1) * Q * y1**3
dispX_dy = np.pi * torch.cos(2 * np.pi * x1) * torch.cos(np.pi * y1)
dispY_dx = np.pi * torch.cos(np.pi * x1) * Q * (y1 ** 4) / 4

sx = lam * (dispX_dx + dispY_dy) + 2 * mu * dispX_dx
sy = lam * (dispX_dx + dispY_dy) + 2 * mu * dispY_dy
sxy = mu * (dispX_dy + dispY_dx)

# Reshape for plotting
sx = sx.detach().cpu().numpy().reshape(nx, ny)
sy = sy.detach().cpu().numpy().reshape(nx, ny)
sxy = sxy.detach().cpu().numpy().reshape(nx, ny)
dispX = dispX.detach().cpu().numpy().reshape(nx, ny)
dispY = dispY.detach().cpu().numpy().reshape(nx, ny)

# Plotting
fig = plt.figure(figsize=(25, 25), dpi=100)

plt.subplot(5, 3, 1)
plt.contourf(arr_x1, arr_y1, sx, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title('Exact Stress in x direction')

plt.subplot(5, 3, 2)
plt.contourf(arr_x1, arr_y1, arr_sx, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted stress in x direction")

plt.subplot(5, 3, 3)
plt.contourf(arr_x1, arr_y1, np.square(sx - arr_sx), 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("error")

plt.subplot(5, 3, 4)
plt.contourf(arr_x1, arr_y1, sy, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact Stress in y direction")

plt.subplot(5, 3, 5)
plt.contourf(arr_x1, arr_y1, arr_sy, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted stress in y direction")

plt.subplot(5, 3, 6)
plt.contourf(arr_x1, arr_y1, np.square(sy - arr_sy), 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("error")

plt.subplot(5, 3, 7)
plt.contourf(arr_x1, arr_y1, dispX, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact displacement in x direction")

plt.subplot(5, 3, 8)
plt.contourf(arr_x1, arr_y1, arr_u, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted displacement in x direction")

plt.subplot(5, 3, 9)
plt.contourf(arr_x1, arr_y1, np.square(dispX - arr_u), 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("error")

plt.subplot(5, 3, 10)
plt.contourf(arr_x1, arr_y1, dispY, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact displacement in y direction")

plt.subplot(5, 3, 11)
plt.contourf(arr_x1, arr_y1, arr_v, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted displacement in y direction")

plt.subplot(5, 3, 12)
plt.contourf(arr_x1, arr_y1, np.square(dispY - arr_v), 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("error")

plt.subplot(5, 3, 13)
plt.contourf(arr_x1, arr_y1, sxy, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact stress in xy plane")

plt.subplot(5, 3, 14)
plt.contourf(arr_x1, arr_y1, arr_sxy, 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted stress in xy plane")

plt.subplot(5, 3, 15)
plt.contourf(arr_x1, arr_y1, np.square(arr_sxy - sxy), 50, cmap="seismic")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("error")
plt.savefig('elastic.png')
plt.show()