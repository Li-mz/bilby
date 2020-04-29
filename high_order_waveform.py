# %%
import bilby
import numpy as np
import matplotlib.pyplot as plt
from noise_detector_3g import sn_lisa, sn_tianqin, sn_taiji

import pycbc
import pycbc.noise
import pycbc.psd
from pycbc.types import FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.waveform import get_fd_waveform

# %%
approx = 'IMRPhenomXHM'

def GR_waveform(farray, mass_1, mass_2,
                phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, mode_array, **kwargs):
    flow = 1e-4
    fhigh = farray[-1]
    deltaf = farray[1] - farray[0]

    waveform_polarizations={}
    hp, hc = \
    get_fd_waveform(approximant=approx,
        mass1=mass_1, mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota, coa_phase=phase,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        delta_f=deltaf, f_lower=flow, f_final=fhigh,
        mode_array=mode_array)
    waveform_polarizations['plus'] = hp.numpy()
    waveform_polarizations['cross'] = hc.numpy()
    return waveform_polarizations


def PV_waveform(farray, mass_1, mass_2,
                phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, A, mode_array, **kwargs):
    flow = 1e-4
    fhigh = farray[-1]
    deltaf = farray[1] - farray[0]

    waveform_polarizations = {}
    hp, hc = \
    get_fd_waveform(approximant=approx,
        mass1=mass_1, mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota, coa_phase=phase,
        spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
        spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
        delta_f=deltaf, f_lower=flow, f_final=fhigh,
        mode_array=mode_array)

    dphi1 = A * (np.pi * farray)** 2
    waveform_polarizations['plus'] = (hp + hc * dphi1).numpy()
    waveform_polarizations['cross'] = (hc - hp * dphi1).numpy()
    return waveform_polarizations


def GR_waveform_from_mode(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, **kwargs):
        return GR_waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                           spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, mode_array, **kwargs)
    
    return waveform


def PV_waveform_from_mode(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, A, **kwargs):
        return PV_waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                           spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, A, mode_array, **kwargs)
    
    return waveform

# %%

injection_parameters = dict(mass_1 = 5e6, mass_2 = 3e6, phase=0, iota=1.3, theta=1, phi=3, psi=np.pi/3,
                            luminosity_distance=20e3, geocent_time=1126259642.4,
                            # spin1x=0.1, spin1y=0.12, spin1z=0.13, spin2x=0.1, spin2y=0.05, spin2z=0.07
                            spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0.)

duration = 2**18
sampling_frequency = 1/16.

np.random.seed(567)
outdir = 'TianQin_LISA'
label = 'PV'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

waveform_arguments = dict(minimum_frequency=1e-4)

# %%

def GR_generator_from_mode(mode_array):
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=GR_waveform_from_mode(mode_array),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)


def PV_generator_from_mode(mode_array):
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=PV_waveform_from_mode(mode_array),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)


def interferometer_from_mode(name, length, sn, mode_array):
    GR_generator = GR_generator_from_mode(mode_array)
    frequencies = np.linspace(1e-4, 1e-2, len(GR_generator.frequency_array))
    psd = sn(frequencies)

    mode_str = ''.join([''.join([str(i) for i in mode]) for mode in mode_array])

    interferometer_a = bilby.gw.detector.Interferometer(
        power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies, psd_array=psd),
        name=name + '_a_' + mode_str, length=length,
        minimum_frequency=min(frequencies), maximum_frequency=max(frequencies),
        latitude=-31.34, longitude=115.91, elevation=0., xarm_azimuth=2., yarm_azimuth=125.)

    interferometer_e = bilby.gw.detector.Interferometer(
        power_spectral_density=bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies, psd_array=psd),
        name=name + '_e_' + mode_str + mode_str, length=length,
        minimum_frequency=min(frequencies), maximum_frequency=max(frequencies),
        latitude=-31.34, longitude=115.91, elevation=0., xarm_azimuth=0, yarm_azimuth=60)
    return interferometer_a, interferometer_e

# %%
mode_array = [[2, 2], [2, 1], [3, 3], [4, 4], [5, 5]]
lisa_a, lisa_e = interferometer_from_mode('lisa', 2.5e6, sn_lisa, mode_array)
taiji_a, taiji_e = interferometer_from_mode('taiji', 3e6, sn_taiji, mode_array)
tianqin_a, tianqin_e = interferometer_from_mode('tianqin', 1.7e5, sn_tianqin, mode_array)

ifos=bilby.gw.detector.InterferometerList([])
ifos.append(lisa_a)
ifos.append(lisa_e)
ifos.append(taiji_a)
ifos.append(taiji_e)
ifos.append(tianqin_a)
ifos.append(tianqin_e)

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - duration + 2096 )
ifos.inject_signal(waveform_generator=GR_generator_from_mode([[2, 2]]),
                   parameters=injection_parameters)
# ifos.plot_data(outdir=outdir)
# %%
priors={}
for key, value in injection_parameters.items():
    priors[key] = value

priors['mass_1'] = bilby.core.prior.Uniform(minimum=1e5, maximum=1e7, name='mass_1')
priors['mass_2'] = bilby.core.prior.Uniform(minimum=1e5, maximum=1e7, name='mass_2')

priors['phase'] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2*np.pi, boundary='periodic')
priors['iota'] = bilby.core.prior.Sine(name='iota')
priors['theta'] = bilby.core.prior.Uniform(name='theta', minimum=0, maximum=np.pi, boundary='periodic')
priors['phi'] = bilby.core.prior.Uniform(name='phi', minimum=0, maximum=2*np.pi, boundary='periodic')
priors['psi'] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')

priors['luminosity_distance'] = bilby.core.prior.Uniform(minimum = 1e3, maximum = 1e5, name='luminosity_distance')
priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 10,
    maximum=injection_parameters['geocent_time'] + 10,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
'''
for key in ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y','spin2z']:
    priors[key] = bilby.core.prior.Uniform(minimum = -0.5, maximum = 0.5, name=key)    
'''
priors['A'] = bilby.core.prior.Uniform(minimum = -1e3, maximum = 1e3, name='A')

#%%
# 这里随便选个generator，对不同阶定义的探测器波形都不一样，因此算响应时需要自行计算波形
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers = ifos, waveform_generator = PV_generator_from_mode([[2, 2]]), priors = priors)

# %%
sampler = 'pymultinest'

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler=sampler, npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label, resume=False)

# %%
no_sample_parameters = ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z']
non_GR_parameters = {'A': 0}
plot_parameters = injection_parameters.copy()
for para in no_sample_parameters:
    plot_parameters.pop(para)
plot_parameters.update(non_GR_parameters)

result.convert_result_mass()
result.plot_corner(parameters = plot_parameters, quantities=[0.05, 0.95])
