#%%
'''
Calculate PSD & antenna pattern functions of 3rd-genertion GW detectors LISA, TianQin and ET.
'''

import numpy as np
import matplotlib.pyplot as plt 
import bilby
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

def sother_lisa(f):
    '''
    return other noise of LISA which is a const.
    '''
    return 8.899e-23

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
    return (4*sacc_lisa(f)+8.899e-23)/(L**2) * (1+(f/(1.29*fstar))**2) +scon_lisa(f)


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
# reference: Niu, arXiv:1910.10592
def sn_tianqin(f):
    Sx = 1e-24
    L = 1.73e8
    fstar = 0.28
    Sa = 1e-30

    A = Sx/L**2
    B = 4*Sa/(2*np.pi*f)**4/L**2*(1+1e-4/f)
    C = 1+(f/1.29/fstar)**2

    return (A+B)*C


#%%
'''
freqs = np.linspace(1,2000,1000)
plt.figure(figsize=(10,6))
plt.loglog(freqs,Sn_ET(freqs)**0.5,label='ET')
plt.xlabel('frequency(Hz)')
plt.ylabel('$\sqrt{S_n(f)}$')
plt.legend()
plt.savefig('ET ASD.png')
plt.show()


freqs = np.linspace(1e-4,1,10000)
plt.figure(figsize=(10,6))
plt.loglog(freqs,Sn_LISA(freqs)**0.5,label='LISA')
plt.loglog(freqs,Sn_LISAoffical(freqs)**0.5,label='LISA offical')
plt.loglog(freqs,Sn_TianQin(freqs)**0.5,label='TianQin')
plt.xlabel('frequency(Hz)')
plt.ylabel('$\sqrt{S_n(f)}$')
plt.legend()
plt.savefig('LISA & TianQin ASD.png')
plt.show()
'''
# %% 
'''
duration = 4e8
sampling_frequency = 0.1
injection_parameters = dict(
    mass_1=3.6e6, mass_2=2.9e6, a_1=0.4, a_2=0.3, tilt_1=0.5, tilt_2=1.0,
    phi_12=1.7, phi_jl=0.3, luminosity_distance=2000., theta_jn=0.4, psi=2.659,
    phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108)

waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                          reference_frequency=1e-2, minimum_frequency=1e-4)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments)

plt.figure(figsize=(10,6))
plt.plot(waveform_generator.frequency_domain_strain(injection_parameters)['cross'])
plt.title('non-PV')
plt.show()

hc = waveform_generator.time_domain_strain(injection_parameters)['cross']
plt.figure(figsize=(10,6))
plt.plot(hc)
plt.title('non-PV')
plt.show()
'''
# %% Antenna pattern function in detector frame
# References:
# Cutler, https://arxiv.org/abs/gr-qc/9703068v1
# Zhao, https://arxiv.org/abs/1009.0206v4
# Liang, arXiv:1901.09624v3
def fplus(gamma, theta, phi, psi):
    '''
    gamma: angle between interferometer arms 
    theta, phi: source direction in detector frame
    psi: polarization angle
    (in radians)
    '''
    A = np.sin(gamma)
    B = 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi)
    C = -np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
    return A*(B+C)

def fcross(gamma, theta, phi, psi):
    '''
    gamma: angle between interferometer arms
    theta, phi: source direction in detector frame
    psi: polarization angle
    (in radians)
    '''
    A = np.sin(gamma)
    B = 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi)
    C = np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
    return A*(B+C)

# %%  Antenna pattern function in ecliptic frame 
# References:
# Liang, arXiv:1901.09624v3
# Niu, arXiv:1910.10592
# Definition of antenna pattern function is $D_{ij}e^{ij}$, where Dij is detector tensor and eij is polarization tensor. In Liang's paper, these two tensor can be expressed in ecliptic frame with theta_e, phi_e, psi_e

def orbit_lisa(t,n):
    '''
    ! Don't use it !
    Calculate LISA's orbit r = ( x(t),y(t),z(t) ) in ecliptic frame
    t: time(s)
    n: =1,2,3, representing 3 detectors

    Reference: A4-A6 in Liang, 2019
    '''
    R = 1.4959787e11  # 1AU, unit:m
    L = 2.5e9  # arm length of LISA
    sigma = L/2/R
    epi = np.arctan(sigma / (1+sigma/1.732))
    eL = (1 + 2*sigma/1.732 + 4*sigma**2/3)**0.5 -1
    theta_Ln = 2*np.pi*(n-1)/3
    return 0


def arm_direction_lisa(i, t):
    '''
    Calculate the unit vector along the i-th arm of LISA in ecliptic frame. 

    Cutler    Liang
     l_1        u
     l_2        v

    i: 1,2,3
    t: timeseries

    Reference: 
    Cutler, arXiv:gr-qc/9703068v1
    Liang, arXiv:1901.09624v3
    '''
    T = 31536000.0  # seconds in a year
    alpha_i = 2*np.pi*t/T - np.pi/12 - (i-1)*np.pi/3
    phi = 2*np.pi*t/T

    ex = np.cos(phi)*np.sin(alpha_i)/2 - np.sin(phi)*np.cos(alpha_i)
    ey = np.sin(phi)*np.sin(alpha_i)/2 + np.cos(phi)*np.cos(alpha_i)
    ez = np.sqrt(3)*np.sin(alpha_i)/2
    tmp = np.array([ex,ey,ez])
    return np.einsum('ji',tmp)  # transpose


def tf_spa(f, tc, m1, m2):
    '''
    unit: f-Hz, tc-s, m1m2-solar mass

    See (A12) in Niu, arXiv:1910.10592
    '''
    m1 = m1 * 2e30
    m2 = m2 * 2e30
    m = m1 + m2
    eta = m1 * m2 / m**2
    Mc = eta**0.6 * m
    gamma = 0.5772
    G = 6.67e-11
    c = 299792458.0
    
    f[0] = f[1] / 1e4 # prevent infinity t

    balabala = 5 / 256 / (G * Mc / c ** 2)**(5 / 3) * (2 * np.pi * f)**(-8 / 3) * c ** (5 / 3)
    erpifgm = 2 * np.pi * f * G * m / c ** 3 
    
    tau0 = erpifgm
    # tau1 = 0
    tau2 = (743/252 + 11/3*eta)*erpifgm**(2/3)
    tau3 = (-32/5*np.pi)*erpifgm**(3/3)
    tau4 = (3058673/508032 + 5429/504*eta + 617/72*eta**2)*erpifgm**(4/3)
    tau5 = (-(7729/252-13/3*eta)*np.pi)*erpifgm**(5/3)
    #tau6 = (-10052469856691/23471078400 + 128/3*np.pi**2 + 6848*gamma/105 +
    #        (3147553127/3048192-451/12*np.pi**2)*eta - 15211/1728*eta**2 +
    #         25565/1296*eta**3 + 3424/105*np.log(16*erpifgm**(2/3))) * erpifgm**(6/3)
    # tau7 = balabala*((-15419335/127008 - 75703/756*eta + 14809/378*eta**2)*np.pi)*erpifgm**(7/3)
    return tc - (tau0 + tau2 + tau3 + tau4 + tau5) * balabala


def transfer_function(f, uw, L):
    '''
    Calculate the transfer funtion for given frequency array, u \cdot w, and detector arm length. 
    Returns a 1D numpy array (of frequency)

    Reference: 
    Liang, arXiv:1901.09624v3
    '''
    c = 299792458.0
    f_transfer = c / (2*np.pi*L)
    return 0.5 * (np.sinc(0.5*f/f_transfer*(1-uw)) * np.exp(-0.5j * f/f_transfer*(3+uw)) +
                  np.sinc(0.5*f/f_transfer*(1+uw)) * np.exp(-0.5j * f/f_transfer*(1+uw)))


def lisa_detector_tensor(name, f_array, theta, phi, tc, m1, m2, L):
    '''
    Returns a Nx3x3 numpy array:
      Detector tensor of LISA in ecliptic frame (of each frequency).
      Definition of detector tensor is D_ij = 1/2*(uiuj-vivj)

    Reference: 
    Liang, arXiv:1901.09624v3
    '''
    t_array = tf_spa(f_array, tc, m1, m2)
    if name == 'lisa1':
        u = arm_direction_lisa(1, t_array)
        v = arm_direction_lisa(2, t_array)
    elif name == 'lisa2':
        u = arm_direction_lisa(1, t_array)
        v = arm_direction_lisa(3, t_array)
    else:
        raise Exception('LISA supposed')
    
    w = np.array([-np.sin(theta) * np.cos(phi),
                  -np.sin(theta) * np.sin(phi),
                  -np.cos(theta)])

    return 0.5 * (np.einsum('ai,aj,a->aij', u, u, transfer_function(f_array, np.einsum('ai,i->a', u, w), L)) - 
                  np.einsum('ai,aj,a->aij', v, v, transfer_function(f_array, np.einsum('ai,i->a', v, w), L)))


def polarization_tensor_ecliptic(theta, phi, psi, polarization):
    '''
    Return the polarization tensor e_ij of GW in ecliptic frame
    polarizarion should be 'p' or 'c'

    Reference: 
    Liang, arXiv:1901.09624v3
    '''
    m = np.array([\
    np.cos(theta)*np.cos(phi)*np.cos(psi)+np.sin(phi)*np.sin(psi),\
    np.cos(theta)*np.sin(phi)*np.cos(psi)-np.cos(phi)*np.sin(psi),\
    -np.sin(theta)*np.cos(psi)])

    n = np.array([\
    -np.cos(theta)*np.cos(phi)*np.sin(psi)+np.sin(phi)*np.cos(psi),\
    -np.cos(theta)*np.sin(phi)*np.sin(psi)-np.cos(phi)*np.cos(psi),\
    np.sin(theta)*np.sin(psi)])

    if polarization == 'plus':
        return np.einsum('i,j', m, m) - np.einsum('i,j', n, n)
    elif polarization == 'cross':
        return np.einsum('i,j', m, n) + np.einsum('i,j', n, m)
    else:
        return 0


def get_lisa_fresponse(name, hp, hc, theta, phi, psi, f_array, tc, m1, m2, L):
    '''
    Get LISA response in frequency domain

    f: frequency series, Hz
    hp, hc: waveform in source frame (f domain)
    theta, phi, psi: source location in ecliptic frame
    '''
    D = lisa_detector_tensor(name, f_array, theta, phi, tc, m1, m2, L)
    ep = polarization_tensor_ecliptic(theta, phi, psi, 'plus')
    ec = polarization_tensor_ecliptic(theta, phi, psi, 'cross')

    fp = np.einsum('aij,ij->a', D, ep)
    fc = np.einsum('aij,ij->a', D, ec)
    return fp * hp + fc * hc


#%%
'''
test1=np.linspace(13,63,1e7)
test2=np.linspace(16,101,1e7)
testf=np.linspace(1,100,1e7)
%timeit get_lisa1_fresponse(hp=test1,hc=test2,theta=2,phi=3,psi=4,f=testf,tc=1000000,m1=2e6,m2=3e6)
'''
