import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_Z(Rs, Rct, C, Rw, omega):
    return Rs + 1/(1j*omega*C + np.sqrt(omega)/(Rw+Rct*np.sqrt(omega)-Rw*1j))


def gradient(Rs, Rct, C, Rw, omega, Zre, Zim):
    """
    Z - impedance as function of current parameters
    Zre, Zim - measured impedance
    grad_1, grad_2, grad_3, grad_4 - partial derivatives with respect to
    Rs, Rct, C, Rw
    up1, up2, up3, up4 - value we update Rs, Rct, C, Rw
    """
    Z = calculate_Z(Rs, Rct, C, Rw, omega)
    diff_re = Z.real - Zre
    diff_im = Z.imag - Zim
    g1 = 1
    g2 = 1/((1+1j)*Rw*C*np.sqrt(omega) +1j*C*omega*Rct+1)**2
    g3 = -1j*omega/(np.sqrt(omega)/(Rct*np.sqrt(omega)+(1-1j)*Rw) + 1j*omega*C)**2
    g4 = (1j-1)/(np.sqrt(omega)*(Rct*C*omega+(1-1j)*C*np.sqrt(omega)*Rw-1j)**2)
    up1 = diff_re*g1.real + diff_im*g1.imag
    up2 = diff_re*g2.real + diff_im*g2.imag
    up3 = diff_re*g3.real + diff_im*g3.imag
    up4 = diff_re*g4.real + diff_im*g4.imag
    return up1, up2, up3, up4


def loss(Z_, Zre, Zim):
    return 1/2*(Z_.real - Zre)**2 + 1/2*(Z_.imag - Zim)**2


def gradient_avg(Rs, Rct, C, Rw, omegas, Zres, Zims):
    """
    Average paremeter updates over all measurements
    """
    n_samples = len(omegas)   
    u1, u2, u3, u4 = (0, 0, 0, 0)
    u_Rs, u_Rct, u_C, u_Rw = (0, 0, 0, 0)
    for i in range(n_samples):
        u1, u2, u3, u4 = gradient(Rs, Rct, C, Rw, omegas[i], Zres[i], Zims[i])
        u_Rs += u1
        u_Rct += u2
        u_C += u3
        u_Rw += u4
    return u_Rs/n_samples, u_Rct/n_samples, u_C/n_samples, u_Rw/n_samples


def loss_avg(Rs, Rct, C, Rw, omegas, Zres, Zims):
    """
    Average calculated loss over all measurements
    """
    n_samples = len(omegas)
    loss_ = 0
    for i in range(n_samples):
        Z_ = calculate_Z(Rs, Rct, C, Rw, omegas[i])
        loss_ += loss(Z_, Zres[i], Zims[i])
    return loss_/n_samples


def gradient_clipping(gradient, threshold):
    """
    Handle exploding gradients resulting with huge parameter update value
    """
    return gradient/abs(gradient)*min(threshold, abs(gradient))

def clrs(min_lr, max_lr, stepsize, policy="sine"):
    """
    Yields next cyclical learning rate between min_lr and max_lr with hal of
    period equal to stepsize. Implemented policies are "sine" and "triangular"
    """
    if policy == "sine":
        k = 1
        while True:
            lr = min_lr + (max_lr - min_lr)*(1 + np.cos(4*np.pi*k/stepsize))/2
            yield lr
            k += 1
    if policy == "triangular":
        k = 1
        while True:
            cycle = k//(2*stepsize)
            x = abs(k/stepsize - 2*cycle - 1)
            lr = min_lr + (max_lr - min_lr)*max(0, 1-x)
            yield lr
            k += 1
    else:
        raise ValueError("Implemented policies are 'sine' and 'triangular'")

# Use case example
# Load data, make sure ZIms is correct
# (we want negative values, ones coming right from measurement)
FILE = r'C:\Users\Lenovo\Desktop\SensDx\Pomiary\wirus_grypa\CSV z 10.11\v1[3].csv'
data = pd.read_csv(FILE)
Zres = data['ZRe']
Zims = [-Zim for Zim in data['ZIm']]
omegas = 2*np.pi*data['Frequency']

# Reasonable parameter values initialization for CLR
Rs = 1000
Rct = 1000
Rw = 1000
C = 1e-6
    

clr = clrs(0.001, 1.5, 5, policy='triangular') #step size multiplier generator
max_iters = 300 # maximum number of iterations
iters = 0 # iteration counter

# Optimize CLR
losses_clr = []
lrs = []
while  iters < max_iters:
    losses_clr.append(loss_avg(Rs, Rct, C, Rw, omegas, Zres, Zims))
    u1, u2, u3, u4 = gradient_avg(Rs, Rct, C, Rw, omegas, Zres, Zims)
    lr = next(clr)
    lrs.append(lr)
    Rs = Rs - lr*u1
    Rct = Rct - lr*u2
    C = C - gradient_clipping(lr*u3, 1e-8)
    Rw = Rw - lr*u4
    iters+=1



# Reasonable parameter values initialization
Rs = 1000
Rct = 1000
Rw = 1000
C = 1e-6
    
lr = .1  # step size multiplier
max_iters = 300 # maximum number of iterations
iters = 0 # iteration counter

# Optimize  
losses = []
while  iters < max_iters:
    losses.append(loss_avg(Rs, Rct, C, Rw, omegas, Zres, Zims))
    u1, u2, u3, u4 = gradient_avg(Rs, Rct, C, Rw, omegas, Zres, Zims)
    Rs = Rs - lr*u1
    Rct = Rct - lr*u2
    C = C - gradient_clipping(lr*u3, 1e-8)
    Rw = Rw - lr*u4
    iters+=1

# Convergence
plt.plot(losses[-100:])
plt.plot(losses_clr[50:150])

# Measured data vs fit
Z_est = [calculate_Z(Rs, Rct, C, Rw, omega) for omega in omegas]
plt.figure(figsize=(20,10))
plt.plot([Z.real for Z in Z_est], [-Z.imag for Z in Z_est], label='fit')
plt.plot(Zres, data['ZIm'], label = 'data')
plt.legend()