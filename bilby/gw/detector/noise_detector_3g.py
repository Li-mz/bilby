'''
Calculate PSD of 3rd-genertion GW detectors LISA, TianQin, Taiji and ET.
'''

import numpy as np

# %%  LISA noise
# reference: Niu, arXiv:1910.10592
def sacc_lisa(f):
    '''
    return acceleration noise of LISA
    f:Hz
    '''
    A = 9e-30/(2*np.pi*f)**4
    B = (6e-4/f)**2
    C = (2.22e-5/f)**8
    return A*(1+B*(1+C))


def scon_lisa(f):
    '''
    return confusion noise of unresolved binaries for LISA.
    f:Hz
    '''
    A = 3.0/20*3.2665e-44
    alpha = 1.183
    s1 = 3014.3
    s2 = 2957.7
    kappa = 2.0928e-3

    B = np.exp(-s1*(f**alpha))
    C = f**(-7.0/3)
    D = 1-np.tanh(s2*(f-kappa))

    return 0.5*A*B*C*D


def sn_lisa(f):
    '''
    return total PSD of LISA.
    f:Hz
    '''
    L = 2.5e9
    fstar = 0.019
    return (4*sacc_lisa(f) + 8.899e-23) / L**2 * (1+(f/(1.29*fstar))**2) + scon_lisa(f)

# %%  ET noise
# reference: Zhao, arXiv:1009.0206
def sn_et(f):
    S0 = 1.449e-52
    p1 = -4.05
    p2 = -0.69
    a1 = 185.62
    a2 = 232.56
    b1 = 31.18
    b2 = -64.72
    b3 = 52.24
    b4 = -42.16
    b5 = 10.17
    b6 = 11.53
    c1 = 13.58
    c2 = -36.46
    c3 = 18.56
    c4 = 27.43
    x = f/200.0
    return S0*(x**p1 + a1*x**p2 + a2*(1+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5+b6*x**6)/(1+c1*x+c2*x**2+c3*x**3+c4*x**4))

# %% TianQin noise
# Reference: arXiv:2002.06360
def sn_tianqin(f):
    Sx = 1e-24
    L = 1.73e8
    fstar = 0.28
    Sa = 1e-30

    S_N = Sx / L**2 + 4 * Sa / (2*np.pi*f)** 4 / L**2 * (1 + 1e-4/f)
    R = 1 / (1 + 0.6 * (f / fstar)** 2)

    return S_N / R

# %% Taiji noise
# Reference: arXiv:2002.06360
def sn_taiji(f):
    c = 299792458.0
    L = 3e9
    fstar = c / (2 * np.pi * L)
    Sx = 64e-24
    Sa = 9e-30

    S_N = Sx / L**2 + 4 * Sa / (2*np.pi*f)** 4 / L**2 * (1 + 1e-4/f)
    R = 1 / (1 + 0.6 * (f / fstar)** 2)
    return S_N / R
