import numpy as np

from .. import utils as gwutils
from ..utils import PropertyAccessor
from .calibration import Recalibrate
from .space_geometry import SpaceInterferometerGeometry
from .strain_data import InterferometerStrainData
from .psd import PowerSpectralDensity
from ..waveform_generator import WaveformGenerator
from .interferometer import Interferometer
from .networks import InterferometerList

from typing import Callable, List

class SpaceInterferometer(Interferometer):
    """Class for Space Interferometer """
    length = PropertyAccessor('geometry', 'length')
    latitude = None
    latitude_radians = None
    longitude = None
    longitude_radians = None
    elevation = None
    x = None
    y = None
    xarm_azimuth = None
    yarm_azimuth = None
    xarm_tilt = None
    yarm_tilt = None
    vertex = None
    detector_tensor = PropertyAccessor('geometry', 'detector_tensor')

    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency,
                 length, orbit, arm_direction, channel,
                 generator: Callable[[List[int]], WaveformGenerator], mode_array, calibration_model=Recalibrate()):
        '''
        generator: 
            function recieve a mode(list), returning a corresponding WaveformGenerator object.
        '''
        self.geometry = SpaceInterferometerGeometry(length, orbit, arm_direction, channel)

        self.name = name
        self.power_spectral_density = power_spectral_density
        self.calibration_model = calibration_model
        self.strain_data = InterferometerStrainData(
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency)
        self.mode_array = mode_array
        self.generator_from_mode = generator
        self.meta_data = dict()

    def __eq__(self, other):
        if self.name == other.name and \
                self.geometry == other.geometry and \
                self.power_spectral_density.__eq__(other.power_spectral_density) and \
                self.calibration_model == other.calibration_model and \
                self.strain_data == other.strain_data:
            return True
        return False
    
    def __repr__(self):
        return self.__class__.__name__ + '(name=\'{}\')'.format(self.name)

    def single_mode_response(self, waveform_polarizations, theta, phi, psi, t):
        D = self.geometry.detector_tensor(t)
        signal = {}
        for mode in ['plus', 'cross']:
            polarization_tensor = gwutils.get_polarization_tensor_ecliptic(theta, phi, psi, mode)
            F = np.einsum('aij,ij->a', D, polarization_tensor)
            signal[mode] = waveform_polarizations[mode] * F
        return sum(signal.values())

    def get_detector_response(self, waveform_polarizations, parameters):
        """ Get the detector response for a particular waveform

        Parameters
        -------
        waveform_polarizations: dict
            polarizations of the waveform
            Not used, leave it just to be consistent with Interferometer class. 
        parameters: dict
            parameters describing position and time of arrival of the signal

        Returns
        -------
        array_like: A 3x3 array representation of the detector response (signal observed in the interferometer)
        """

        m1 = parameters['mass_1']
        m2 = parameters['mass_2']
        theta = parameters['theta']
        phi = parameters['phi']
        psi = parameters['psi']
        tc = parameters['geocent_time']

        signal_mode = []
        for mode in self.mode_array:
            t = gwutils.tf_spa_from_mode(self.frequency_array, tc, m1, m2, mode)
            waveform_polarizations = self.generator_from_mode(mode).frequency_domain_strain(parameters)

            signal = self.single_mode_response(waveform_polarizations, theta, phi, psi, t)
            dt = self.time_delay_from_solarcenter(theta, phi, t)[self.strain_data.frequency_mask]

            signal *= self.strain_data.frequency_mask

            dt_geocent = parameters['geocent_time'] - self.strain_data.start_time  # not really "geo"cent
            dt += dt_geocent
            signal[self.strain_data.frequency_mask] = signal[self.strain_data.frequency_mask] * np.exp(
                -1j * 2 * np.pi * dt * self.strain_data.frequency_array[self.strain_data.frequency_mask])
            signal_mode.append(signal)
        return sum(signal_mode)

    def time_delay_from_solarcenter(self, theta, phi, t):
        '''
        reference: arXiv:1803.03368v1
        '''
        c = 299792458
        Omega = - np.array([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)])
        r0 = self.geometry.orbit(t) / c
        return np.einsum('j,ij->i', Omega, r0)

class SpaceInterferometerList(InterferometerList):
    def __init__(self, name, power_spectral_density, minimum_frequency, maximum_frequency,
                 length, orbit, arm_direction, generator, mode_array):
        super(SpaceInterferometerList, self).__init__([])
        for channel in ['a', 'e']:
            self.append(SpaceInterferometer('_'.join([name, channel]), power_spectral_density, minimum_frequency, maximum_frequency,
                length, orbit, arm_direction, channel, generator, mode_array))


def get_space_interferometer(name, psd_frequency_array, generator, mode_array):
    from .noise_detector_3g import sn_lisa, sn_tianqin, sn_taiji
    from .space_geometry import earth_orbit_circular, earth_orbit, LISAlike_arm_direction, TianQinlike_arm_direction
    ifo_config = {
        'lisa': {
            'length': 2.5e6,
            'psd': sn_lisa,
            'orbit': earth_orbit_circular(20),
            'arm_direction': LISAlike_arm_direction()
        },
        'tianqin': {
            'length': 1.7e5,
            'psd': sn_tianqin,
            'orbit': earth_orbit(0),
            'arm_direction': TianQinlike_arm_direction(-4.7, 120.5, 1/315360)
        },
        'taiji': {
            'length': 3e6,
            'psd': sn_taiji,
            'orbit': earth_orbit_circular(-20),
            'arm_direction': LISAlike_arm_direction(30)
        }
    }
    if name.lower() in ifo_config.keys():
        config = ifo_config[name.lower()]
        return SpaceInterferometerList(
            power_spectral_density=PowerSpectralDensity(
                frequency_array=psd_frequency_array, psd_array=config['psd'](psd_frequency_array)),
            minimum_frequency=min(psd_frequency_array), maximum_frequency=max(psd_frequency_array),
            name=name, length=config['length'], orbit=config['orbit'], arm_direction=config['arm_direction'],
            generator=generator, mode_array=mode_array)
    else:
        raise ValueError("can only provide these interferometers: " + ','.join(ifo_config.keys()))

