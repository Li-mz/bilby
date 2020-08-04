# %%
import bilby
import numpy as np
import matplotlib.pyplot as plt

import pycbc
import pycbc.noise
import pycbc.psd
from pycbc.types import FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.waveform import get_fd_waveform

# %%
approx = 'IMRPhenomXHM'


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

    dphi1 = A * (np.pi * farray) ** 2
    waveform_polarizations['plus'] = (hp + hc * dphi1).numpy()
    waveform_polarizations['cross'] = (hc - hp * dphi1).numpy()
    return waveform_polarizations


def PVam_waveform(farray, mass_1, mass_2,
                  phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                  spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, B, mode_array, **kwargs):
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

    dh = B * (np.pi * farray)
    waveform_polarizations['plus'] = (hp - hc * dh * 1j).numpy()
    waveform_polarizations['cross'] = (hc + hp * dh * 1j).numpy()
    return waveform_polarizations


def PV_waveform_from_mode(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, A, **kwargs):
        return PV_waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                           spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, A, mode_array, **kwargs)

    return waveform


def PVam_waveform_from_mode(mode_array):
    def waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                 spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, B, **kwargs):
        return PVam_waveform(farray, mass_1, mass_2, phase, iota, theta, phi, psi, luminosity_distance, geocent_time,
                             spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, B, mode_array, **kwargs)

    return waveform

# %%
injection_parameters = dict(mass_1=5e6, mass_2=3e6, phase=1., iota=1.3, theta=1, phi=3, psi=np.pi / 3,
                            luminosity_distance=20e3, geocent_time=1126259642.4,
                            spin1x=0., spin1y=0., spin1z=0., spin2x=0., spin2y=0., spin2z=0., A=0)

duration = 2**18
sampling_frequency = 1 / 16

np.random.seed(0)
outdir = 'LISA_Taiji_TianQin_PV'
label = 'PV'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

waveform_arguments = dict(minimum_frequency=1e-4)

# %%
def PV_generator_from_mode(mode):
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=PV_waveform_from_mode([mode]),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        init_log=False)


def PVam_generator_from_mode(mode):
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=PVam_waveform_from_mode([mode]),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        init_log=False)


# %%
mode_array = [[2, 2], [2, 1], [3, 3], [4, 4], [5, 5]]
frequencies = np.linspace(1e-4, 1e-2, len(PV_generator_from_mode([2, 2]).frequency_array))
lisa = bilby.gw.detector.get_space_interferometer('LISA', frequencies, PV_generator_from_mode, mode_array)
taiji = bilby.gw.detector.get_space_interferometer('Taiji', frequencies, PV_generator_from_mode, mode_array)
tianqin = bilby.gw.detector.get_space_interferometer('Tianqin', frequencies, PV_generator_from_mode, mode_array)

ifos = lisa
ifos.extend(taiji)
ifos.extend(tianqin)

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - duration + 2096)
ifos.inject_signal(waveform_generator=PV_generator_from_mode([2, 2]),
                   parameters=injection_parameters)
# ifos.plot_data(outdir=outdir)
# %%
priors = {}
for key, value in injection_parameters.items():
    priors[key] = value

priors['mass_1'] = bilby.core.prior.Uniform(minimum=1e5, maximum=1e7, name='mass_1')
priors['mass_2'] = bilby.core.prior.Uniform(minimum=1e5, maximum=1e7, name='mass_2')

priors['phase'] = bilby.core.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors['iota'] = bilby.core.prior.Sine(name='iota')
priors['theta'] = bilby.core.prior.Uniform(
    name='theta', minimum=0, maximum=np.pi, boundary='periodic', latex_label=r'$\theta_e$')
priors['phi'] = bilby.core.prior.Uniform(name='phi', minimum=0, maximum=2 * np.pi,
                                         boundary='periodic', latex_label=r'$\phi_e$')
priors['psi'] = bilby.core.prior.Uniform(name='psi', minimum=0, maximum=np.pi,
                                         boundary='periodic', latex_label=r'$\psi$')

priors['luminosity_distance'] = bilby.core.prior.Uniform(minimum=1e3, maximum=1e5, name='luminosity_distance')

priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 10,
    maximum=injection_parameters['geocent_time'] + 10,
    name='geocent_time', latex_label='$t_c$', unit='$s$')
'''
for key in ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y','spin2z']:
    priors[key] = bilby.core.prior.Uniform(minimum = -0.5, maximum = 0.5, name=key)    
'''
priors['A'] = bilby.core.prior.Uniform(minimum=-1e3, maximum=1e3, name='A')

# %%
# waveform_generator here is not applied in calculating likelihood, because we only use parameters injected to calculate response.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=PV_generator_from_mode([2, 2]), priors=priors, reference_frame='ecliptic')

# %%
sampler = 'pymultinest'

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler=sampler, npoints=1000,
    injection_parameters=injection_parameters, outdir=outdir, label=label)

result.convert_result_mass()
result.plot_corner(quantiles=[0.05, 0.95])
