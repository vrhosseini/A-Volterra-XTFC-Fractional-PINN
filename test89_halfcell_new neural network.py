import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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




# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PINN1:
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
            a = torch.nn.Parameter(torch.tensor(1, dtype=torch.float64))
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
            H = torch.tanh(1 * self.A[l] * (H @ W + b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = H @ W + b
        return Y

    def to_device(self):
        # Move parameters to GPU if available
        for param in self.weights + self.biases + self.A:
            param.data = param.data.to(device)
            param.requires_grad = True

# Physics loss function outside the class
def physics_loss(model, x, t, alpha=0.01):
    x = x.to(device).requires_grad_(True)
    t = t.to(device).requires_grad_(True)
    u = model.neural_net_tanh(torch.cat([x, t], dim=1))

    # Compute derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    # Heat equation residual
    residual = u_t - alpha * u_xx
    return torch.mean(residual ** 2)

# Neural network configuration
layers = [2, 20, 20, 20, 1]  # Input layer (x, t), 3 hidden layers, output layer (u)
model = PINN1(layers)

# Define optimizer for weights, biases, and A
params = model.weights + model.biases + model.A
optimizer = optim.Adam(params, lr=0.001)

# Define initial and boundary conditions
x_ic = torch.rand(100, 1, dtype=torch.float64, device=device)  # Initial condition for x (e.g., x between 0 and 1)
t_ic = torch.zeros(100, 1, dtype=torch.float64, device=device)  # t = 0 (initial time)
u_ic = torch.sin(np.pi * x_ic).to(device)  # Initial condition u(0, x) = sin(pi * x)

x_bc = torch.cat([torch.zeros(50, 1, dtype=torch.float64), torch.ones(50, 1, dtype=torch.float64)]).to(device)
t_bc = torch.rand(100, 1, dtype=torch.float64, device=device)  # Random times for boundary conditions
u_bc = torch.zeros_like(x_bc)  # Boundary condition u(t, 0) = u(t, 1) = 0

# Training loop
epochs = 5000
loss_history = []
# Physics-informed loss
x_phys = torch.rand(100, 1, dtype=torch.float64, device=device)  # Collocation points in x
t_phys = torch.rand(100, 1, dtype=torch.float64, device=device)  # Collocation points in t
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Calculate the losses
    u_pred_ic = model.neural_net_tanh(torch.cat([x_ic, t_ic], dim=1))
    mse_ic = torch.mean((u_pred_ic - u_ic) ** 2)  # Initial condition loss

    u_pred_bc = model.neural_net_tanh(torch.cat([x_bc, t_bc], dim=1))
    mse_bc = torch.mean((u_pred_bc - u_bc) ** 2)  # Boundary condition loss

    
    mse_phys = physics_loss(model, x_phys, t_phys)

    # Total loss
    loss = mse_ic + mse_bc + mse_phys
    loss.backward()
    optimizer.step()
    
    # Store and print loss every 500 epochs
    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Plot training loss over epochs
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss of PINN')
plt.show()
