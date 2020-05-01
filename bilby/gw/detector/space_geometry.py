import numpy as np

from .. import utils as gwutils

from typing import Callable, Union


class SpaceInterferometerGeometry(object):
    def __init__(self, length,
                 orbit: Callable[[np.ndarray], np.ndarray],
                 arm_direction: Callable[[int, np.ndarray], np.ndarray],
                 channel):
        '''
        Initialize a SpaceInterferometerGeometry instance.
        Parameters
        ----------

        length: float
            Length of the interferometer in km.
        orbit: 
            Function recieves a 1D numpy array(time array), returning position of interferometer center in ecliptic frame.
        arm_direction:
            Function recieves a int i and time array, returning unit vector along i-th arm in ecliptic frame.
        channel: str, 'a' or 'e'
            Specifying which channel does this interferometer object correspond to.
        '''
        self.length = length
        self.orbit = orbit
        self.unit_vector_along_arm = arm_direction
        
        self.channel = channel
        if channel == 'a':
            self.detector_tensor = self.detector_tensor_a
        elif channel == 'e':
            self.detector_tensor = self.detector_tensor_e

    def __eq__(self, other):
        for attribute in ['length', 'channel']:
            if not getattr(self, attribute) == getattr(other, attribute):
                return False
        return True

    def __repr__(self):
        return self.__class__.__name__ + '(length={}m)'.format(float(self.length))

    def detector_tensor_a(self, t):
        n1 = self.unit_vector_along_arm(1, t)
        n2 = self.unit_vector_along_arm(2, t)
        n3 = self.unit_vector_along_arm(3, t)
        return 1 / 6 * (np.einsum('ai,aj->aij', n1, n1) - 2 * np.einsum('ai,aj->aij', n2, n2) + np.einsum('ai,aj->aij', n3, n3))

    def detector_tensor_e(self, t):
        n1 = self.unit_vector_along_arm(1, t)
        n3 = self.unit_vector_along_arm(3, t)
        return np.sqrt(3) / 6 * (np.einsum('ai,aj->aij', n1, n1) - np.einsum('ai,aj->aij', n3, n3))


def earth_orbit(phase=0.):
    '''
    Returns a orbit for interferometer which moves in Earth orbit and ahead of the Earth by 'phase' degree.
    e.g., for LISA, phase=20; for Taiji, phase=-20
    '''
    def orbit(t):
        R = 1.4959787e11  # 1AU
        e = 0.0167        # eccentricity of the geocenter orbit around the Sun
        T = 31557600.0      # 1 sidereal year
        
        alpha = 2 * np.pi * t / T + phase * np.pi / 180

        # center of mass
        x = R * np.cos(alpha) + 0.5 * R * e * (np.cos(2 * alpha) - 3) - 1.5 * R * e**2 * np.cos(alpha) * (np.sin(alpha)** 2)
        y = R * np.sin(alpha) + 0.5 * R * e * np.sin(2 * alpha) + 0.25 * R * e**2 * (3 * np.cos(2 * alpha) - 1) * np.sin(alpha)
        z = np.zeros(t.shape)
        return np.array([x, y, z]).transpose()
    return orbit


def earth_orbit_circular(phase=0.):
    '''
    Returns a orbit for interferometer which moves in a circular orbit with R=1AU and ahead of the Earth by 'phase' degree.
    e.g., for LISA, phase=20; for Taiji, phase=-20
    '''
    def orbit(t):
        R = 1.4959787e11  # 1AU
        T = 31557600.0    # 1 sidereal year
        
        alpha = 2 * np.pi * t / T + phase * np.pi / 180

        return np.array([R * np.cos(alpha),
                         R * np.sin(alpha),
                         np.zeros(t.shape)]).transpose()
    
    return orbit


def LISAlike_arm_direction(phase=0.):
    '''
    Returns arm_direction function of a LISA-like interferometer.

    Reference: 
    Cutler, arXiv:gr-qc/9703068v1
    Liang, arXiv:1901.09624v3
    '''
    def arm_direction(i, t):
        T = 31557600.0    # 1 sidereal year
        alpha_i = 2 * np.pi * t / T - np.pi / 12 - (i - 1) * np.pi / 3 + phase * np.pi / 180
        phi = 2 * np.pi * t / T

        ex = np.cos(phi) * np.sin(alpha_i) / 2 - np.sin(phi) * np.cos(alpha_i)
        ey = np.sin(phi) * np.sin(alpha_i) / 2 + np.cos(phi) * np.cos(alpha_i)
        ez = np.sqrt(3) * np.sin(alpha_i) / 2
        return np.array([ex, ey, ez]).transpose()
    return arm_direction


def TianQinlike_arm_direction(theta_s, phi_s, fsc):
    '''
    Returns arm_direction function of a TianQin-like interferometer.
    Namely, the normal vector of detector plane points at a fixed reference source (theta_s,phi_s),
    and spacecrafts are moving in a circular orbit around the Earth with frequency fsc. 
    For TianQin, theta_s=-4.7, phi_s=120.5, fsc=1/315360 (1/3.65 days)

    Reference: 
    arXiv:1803.03368
    '''
    def arm_direction(i, t):
        thetas = theta_s * np.pi / 180
        phis = phi_s * np.pi / 180
        alpha_i = 2 * np.pi * fsc * t + 2 * np.pi / 3 * i - np.pi / 3

        x = np.cos(phis) * np.sin(thetas) * np.cos(alpha_i) - np.sin(alpha_i) * np.sin(phis)
        y = np.sin(phis) * np.sin(thetas) * np.cos(alpha_i) + np.sin(alpha_i) * np.cos(phis)
        z = -np.cos(alpha_i) * np.cos(theta_s)
        return np.array([x, y, z]).transpose()
    return arm_direction