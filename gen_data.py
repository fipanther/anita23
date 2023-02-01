# this script will generate data for 
# the ANITA 2023 summer school. 
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pycbc.noise
import pycbc.psd
import pycbc.filter
import pycbc.waveform
import  pycbc.detector
from pycbc.noise.reproduceable import colored_noise
from pycbc.types import TimeSeries, FrequencySeries

#------- Some setup -------#
# this will mean it runs for anyone
CWD = os.getcwd()

# The PSDs - this is from O3a
PSD_DIR = os.path.join(CWD, 'input_data')
PSD_fnames = os.listdir(PSD_DIR)

print('Available PSDs are {}'.format(PSD_fnames))

#------- Functions that we need to speed this up -------#

# just generate Gaussian noise from the O3 PSD at 4096 Hz: 
# we are only going to inject BBH signals, so we don't need a very high sample
# rate

def generate_noise_timeseries(PSD_DIR, PSD_FNAME, gps_start, duration, random_seed = 122, f_low = 10.0, 
															delta_f = 0.25, fmax = 4096, 
															is_asd_file=True, plot_psd = False,
															plot_timeseries = False):

	"""
	Generates a PyCBC timeseries of colored Gaussian noise for a given PSD
	PSD_DIR (str): directory where your PSDs are
	my_psds (arr): filenames of the PSD you want to load
	gps_start (float): the gps start time of the data you'll generate

	optional:
	random_seed (int): give the noise a random seed. Make sure you use the same seed if you want 
						this to be reproducible
	f_low (float): low frequency cut off
	delta_f (float): PSD interpolation interval
	fmax (float): maximum frequency to go up to
	is_asd_file: whether or not this file is an ASD or PSD
	plot_psd: make a plot to check stuff
	plot_timeseries: make a plot of the output timeseries
	"""
	flen = int(fmax/delta_f) + 1
	# set some constants that we'll need that will stay the same across everything
	sample_rate = 4096 # sample rate in Hz
	delta_t = 1.0 / sample_rate # inverse sample rate

	psd = pycbc.psd.from_txt(os.path.join(PSD_DIR, PSD_FNAME), flen, delta_f,
                         f_low, is_asd_file=is_asd_file)

	#---------- Can plot them just to be sure it all loaded OK --------#
	if plot_psd == True:
		plt.figure(figsize = (7,6))
		plt.plot(psd_L1.sample_frequencies, psd_L1, label = 'L1 (Livingston)')
		plt.plot(psd_H1.sample_frequencies, psd_H1, label = 'H1 (Hanford)')
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim([10, 5000])
		plt.title('Display loaded PSDs')
		plt.legend(loc = 'best')
		plt.show()

	# generate the noise timeseries using PyCBC
	tsamples = int(duration / delta_t)

	Timeseries = colored_noise(psd, gps_start,  gps_start + duration, seed = random_seed, 
												sample_rate = sample_rate, 
												low_frequency_cutoff = f_low)

	if plot_timeseries == True:

		#-------- plot it to check it out if you want --------#
		fig, axs = plt.subplots(nrows = 2, ncols = 1,
							    figsize = (10, 5),
							    sharex = True)
		ax1, ax2 = axs
		ax1.plot(L1_example_timeseries.sample_times, L1_example_timeseries, 'tab:blue', label = 'L1 (Livingston)')
		ax2.plot(H1_example_timeseries.sample_times, H1_example_timeseries, 'tab:orange', label = 'H1 (Hanford)')

		ax1.legend(loc='best')
		ax2.legend(loc='best')

		ax1.set_ylabel('Strain')
		ax2.set_ylabel('Strain')
		ax2.set_xlabel('time (s)')
		plt.show()

	return Timeseries

#-------- generate the injections ---------#

def inject_waveform_to_timeseries(noise_timeseries, this_detector, ra, dec, pol, inc, distance, 
									mass1, mass2, gps_start, injection_time, 
									spin1z=1., spin2z = 1., f_low = 30.0, apx = "IMRPhenomD", 
									make_plot = False):
	"""
	Inject a waveform into your timeseries
	
	noise_timeseries: PyCBC timeseries of noise to inject into
	this_detector: which detector are you using?
	# these are parameters that are defined by PyCBC's waveform/detector module
	ra (float (0, 2pi)): right ascension of injection
	dec (float, (0, pi)): declination of injection
	pol (float): polarization angle
	inc (float): inclination angle
	distance (float): distance to source in Mpc
	mass1, mass2 (float): masses 
	gps_start (float): beginning of the noise timeseries
	injection_time (float): merger time in seconds relative to gps_start

	optional:
	spin1/2z: spin of objects
	f_low: low frequency cut off
	approximant: if you want to change the approximant, change this here. 
	"""
	# set some constants that we'll need that will stay the same across everything
	sample_rate = 4096 # sample rate in Hz
	delta_t = 1.0 / sample_rate # inverse sample rate
	
	# set up the detectors ready to project the waveforms:
	current_detector = pycbc.detector.Detector(this_detector)

	# generate the antenna beam pattern functions for this particular sky localization
	# We get back the fp and fc antenna pattern weights.
	fp, fc = current_detector.antenna_pattern(ra, dec, pol, injection_time)


	## Generate a waveform
	hp, hc = pycbc.waveform.get_td_waveform(approximant=apx, mass1=mass1, mass2=mass2,
	                         f_lower=f_low, delta_t=delta_t, inclination=inc,
	                         distance=distance)

	# shift the merger time
	hp.start_time+=gps_start+injection_time
	hc.start_time+=gps_start+injection_time

	# calculate the waveform projected into each detector
	ht = current_detector.project_wave(hp, hc,  ra, dec, pol)#fp * hp + fc * hc

	strain_with_injection = noise_timeseries.inject(ht, copy=True)

	return strain_with_injection

#------ then we can save everything out into our data directory-----#

def save_data_and_generate_log_file(OUT_DIR, out_name, out_logfile, strain_with_injection, PSD_DIR, PSD_FNAME, 
															gps_start, duration, 
															ra, dec, pol, inc, distance, 
															mass1, mass2, injection_time, multi_inj = False):
	"""
	This will save everything and also create a log file so you know what inputs you made
	"""

	# save the timeseries:
	out_fname = os.path.join(OUT_DIR, out_name) + '.txt'
	print(out_fname)
	# this file can then be re-loaded with PyCBC in the future
	strain_with_injection.save(out_fname)

	if multi_inj == False:
		out_labels = np.asarray(['filename:', 'psd:', 'gps_segment_start:', 'duration:',
						'ra:', 'dec:', 'polarization:', 'inclination:', 'distance:', 
						'mass1:', 'mass2:', 'injection_time:'])
		out_vals = [out_fname, os.path.join(PSD_DIR, PSD_FNAME), gps_start, duration, ra, dec, pol, inc, distance, 
					mass1, mass2, injection_time]


		logfile_fname = os.path.join(OUT_DIR, out_logfile)

		out_df = pd.DataFrame(np.column_stack([out_labels, out_vals]))
		if not os.path.isfile(logfile_fname+'.csv'):
			out_df.to_csv(logfile_fname+'.csv', index=False, index_label=None, header = False)
	return print('saved all files!')



OUT_DIR = os.path.join(os.getcwd(), 'DATA')
if not os.path.isdir(OUT_DIR):
	os.makedirs(OUT_DIR)
	print('created folder for data {}'.format(OUT_DIR))

else: 
	print('Folder {} already exists!'.format(OUT_DIR))


#----- usage to generate the example: -----#

# example_gps = 1243350018
# example_seed = 122
# duration = 128
# ra = 1.7
# dec = 1.7
# pol = 0.4
# inc = 1
# distance = 900
# mass1 = 30
# mass2 = 30
# example_injection_time = 40
# example_fname = 'example_timeseries_'
# example_fname_log = 'out_logfile_example'


# L1_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[1], example_gps, duration, 
# 															random_seed = example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# H1_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[0], example_gps, duration, 
# 															random_seed = example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# L1_example_with_injection = inject_waveform_to_timeseries(L1_example_timeseries, "L1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, example_gps, example_injection_time)
# H1_example_with_injection = inject_waveform_to_timeseries(H1_example_timeseries, "H1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, example_gps, example_injection_time)

# for this_timeseries, this_detector, this_psd in zip([L1_example_with_injection, H1_example_with_injection], 
# 													['L1', 'H1'], [PSD_fnames[1], PSD_fnames[0]]):

# 	save_data_and_generate_log_file(OUT_DIR, example_fname+this_detector, example_fname_log, 
# 									this_timeseries, PSD_DIR, this_psd, 
# 									example_gps, duration, 
# 									ra, dec, pol, inc, distance, 
# 									mass1, mass2, example_injection_time)



#----- Generate high SNR test example: -----#

# highSNR_example_gps = 1243350018
# highSNR_example_seed = 124
# duration = 128
# ra = 0.2
# dec = 1.5
# pol = 0.1
# inc = 0.
# distance = 300
# mass1 = 25
# mass2 = 31
# highSNR_example_injection_time = 90
# highSNR_example_fname = 'highSNR_example_timeseries_'
# highSNR_example_fname_log = 'highSNR_out_logfile_example'


# L1_highSNR_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[1], highSNR_example_gps, duration, 
# 															random_seed = highSNR_example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# H1_highSNR_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[0], highSNR_example_gps, duration, 
# 															random_seed = highSNR_example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# L1_highSNR_with_injection = inject_waveform_to_timeseries(L1_highSNR_example_timeseries, "L1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, highSNR_example_gps, highSNR_example_injection_time)
# H1_highSNR_with_injection = inject_waveform_to_timeseries(H1_highSNR_example_timeseries, "H1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, highSNR_example_gps, highSNR_example_injection_time)

# for this_timeseries, this_detector, this_psd in zip([L1_highSNR_with_injection, H1_highSNR_with_injection], 
# 													['L1', 'H1'], [PSD_fnames[1], PSD_fnames[0]]):

# 	save_data_and_generate_log_file(OUT_DIR, highSNR_example_fname+this_detector, highSNR_example_fname_log, 
# 									this_timeseries, PSD_DIR, this_psd, 
# 									highSNR_example_gps, duration, 
# 									ra, dec, pol, inc, distance, 
# 									mass1, mass2, highSNR_example_injection_time)


#----- Generate low SNR test example: -----#

# lowSNR_example_gps = 1243350018
# lowSNR_example_seed = 124
# duration = 128
# ra = 0.2
# dec = 1.5
# pol = 0.1
# inc = 0.
# distance = 2400
# mass1 = 75
# mass2 = 40
# lowSNR_example_injection_time = 45
# lowSNR_example_fname = 'lowSNR_example_timeseries_'
# lowSNR_example_fname_log = 'lowSNR_out_logfile_example'


# L1_lowSNR_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[1], lowSNR_example_gps, duration, 
# 															random_seed = lowSNR_example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# H1_lowSNR_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[0], lowSNR_example_gps, duration, 
# 															random_seed = lowSNR_example_seed, f_low = 10.0, 
# 															delta_f = 0.25, fmax = 4096, 
# 															is_asd_file=True, plot_psd = False,
# 															plot_timeseries = False)

# L1_lowSNR_with_injection = inject_waveform_to_timeseries(L1_lowSNR_example_timeseries, "L1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, lowSNR_example_gps, lowSNR_example_injection_time)
# H1_lowSNR_with_injection = inject_waveform_to_timeseries(H1_lowSNR_example_timeseries, "H1",
# 													 ra, dec, pol, inc, distance, 
# 													mass1, mass2, lowSNR_example_gps, lowSNR_example_injection_time)

# for this_timeseries, this_detector, this_psd in zip([L1_lowSNR_with_injection, H1_lowSNR_with_injection], 
# 													['L1', 'H1'], [PSD_fnames[1], PSD_fnames[0]]):

# 	save_data_and_generate_log_file(OUT_DIR, lowSNR_example_fname+this_detector, lowSNR_example_fname_log, 
# 									this_timeseries, PSD_DIR, this_psd, 
# 									lowSNR_example_gps, duration, 
# 									ra, dec, pol, inc, distance, 
# 									mass1, mass2, lowSNR_example_injection_time )


#----- Generate the challenge: -----#

challenge_gps = 1243350018
challenge_seed = 129
duration = 256

# # generate injection times
# how many injections to generate
n_injections = 20

# give at least 10 second buffer at start and end
valid_times = [10, 256-10]

# how long do you have to inject over if they were equally spaced?
inj_segment_len = (valid_times[1] - valid_times[0]) / n_injections

# calculate the initial injection time and then do everything iteratively
injection_times = []
init_injection_time = np.random.uniform(valid_times[0], valid_times[0] + inj_segment_len)
injection_times.append(init_injection_time)
for i in range(n_injections - 1):
	# add 2 second buffer at the beginning of each
	this_injection_time = np.random.uniform(injection_times[i] + 3, injection_times[i] + inj_segment_len + 3)
	injection_times.append(this_injection_time)

# check that the last injection time is still within the segment length:
if max(injection_times) > valid_times[1]:
	print('WARNING! Last injection after end of valid segment!')



# # set up some random injection parameters
# # make them uniform
ra = np.random.uniform(0, np.pi * 2, n_injections)
dec = np.random.uniform(- np.pi / 2, np.pi / 2, n_injections)
pol = np.random.uniform(0, np.pi * 2, n_injections)
inc = np.random.uniform(0, np.pi * 2, n_injections)
distance = np.random.uniform(300, 3000, n_injections)
mass1 = np.random.uniform(20, 100, n_injections)
mass2 = np.random.uniform(20, 100, n_injections)
spin1z = np.random.uniform(0, 1, n_injections)
spin2z = np.random.uniform(0, 1, n_injections)

apx = "IMRPhenomD"

# load the PSDs as it is needed
L1_psd = pycbc.psd.from_txt(os.path.join(PSD_DIR, PSD_fnames[1]), int(4096/0.25) + 1, 0.25,
                         20, is_asd_file=True)

H1_psd = pycbc.psd.from_txt(os.path.join(PSD_DIR, PSD_fnames[1]), int(4096/0.25) + 1, 0.25,
                         20, is_asd_file=True)



# calculate the SNR and duration of each signal
opt_snr_L1 = []
siglens = []
for i in range(n_injections):
	# approximate
	hp, _ = pycbc.waveform.get_td_waveform(approximant=apx, mass1=mass1[i], mass2=mass2[i],
	                         f_lower=30, delta_t=1/4096, inclination=inc[i],
	                         distance=distance[i])

	template = hp.cyclic_time_shift(hp.start_time)

	L1_psd = pycbc.psd.interpolate(L1_psd, hp.delta_f)
	H1_psd = pycbc.psd.interpolate(H1_psd, hp.delta_f)

	L1_SNR = max(np.abs(pycbc.filter.matched_filter(template, hp, psd = L1_psd, low_frequency_cutoff = 30).crop(0.1, 0.1)))
	dur = pycbc.waveform.compress.rough_time_estimate(mass1[i], mass1[i], 30)

	opt_snr_L1.append(L1_SNR)
	siglens.append(dur)


challenge_fname = 'challenge_timeseries_'
fname_log = 'challenge_logfile'


L1_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[1], challenge_gps, duration, 
															random_seed = challenge_seed, f_low = 10.0, 
															delta_f = 0.25, fmax = 4096, 
															is_asd_file=True, plot_psd = False,
															plot_timeseries = False)

H1_example_timeseries = generate_noise_timeseries(PSD_DIR, PSD_fnames[0], challenge_gps, duration, 
															random_seed = challenge_seed, f_low = 10.0, 
															delta_f = 0.25, fmax = 4096, 
															is_asd_file=True, plot_psd = False,
															plot_timeseries = False)

for i in range(n_injections):

	L1_example_with_injection = inject_waveform_to_timeseries(L1_example_timeseries, "L1",
														 ra[i], dec[i], pol[i], inc[i], distance[i], 
														mass1[i], mass2[i], challenge_gps, injection_times[i], 
														spin1z = spin1z[i], spin2z = spin2z[i])
	H1_example_with_injection = inject_waveform_to_timeseries(H1_example_timeseries, "H1",
														 ra[i], dec[i], pol[i], inc[i], distance[i], 
														mass1[i], mass2[i], challenge_gps, injection_times[i], 
														spin1z = spin1z[i], spin2z = spin2z[i])

for this_timeseries, this_detector, this_psd in zip([L1_example_with_injection, H1_example_with_injection], 
													['L1', 'H1'], [PSD_fnames[1], PSD_fnames[0]]):

	save_data_and_generate_log_file(OUT_DIR, challenge_fname+this_detector, None, 
									this_timeseries, None, None, 
									 None,  None, 
									 None,  None,  None,  None,  None, 
									 None,  None,  None, multi_inj = True)

	columns_out = ['injection_time', 'mass1', 'mass2', 'distance', 
					'optsnr_L1', 'duration', 'ra', 'dec', 'inc', 'pol', 'spin1z', 'spin2z']

	data_out = np.column_stack([injection_times, mass1, mass2, distance,
								opt_snr_L1, siglens, ra, dec, inc, pol, spin1z, spin2z])

	logfile_fname = os.path.join(OUT_DIR, fname_log)

	out_df = pd.DataFrame(data_out, columns = columns_out)
	out_df.to_csv(logfile_fname+'.csv')



