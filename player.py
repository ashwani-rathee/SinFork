#Main FIle of MP3Player
# Author: Ashwani Rathee
# Learned a lot

#Importing Libraries
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import pygame
from PIL import ImageTk, Image
import os
import librosa
import pandas as pd
import numpy as np
from matplotlib import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import librosa.display
import webbrowser
import essentia
import essentia.standard
import essentia.streaming
from pylab import plot, show, figure, imshow
import scipy
import sklearn

from essentia.standard import *
plt.rcParams['figure.figsize'] = (15, 6)

def get_filenames():
    path = r"/home/ashwani/hamr-project-final/assets/audio1"
    return os.listdir(path)

#print(dir(essentia.standard)) #to see features available in essentia

root=Tk()
root.title('Sinfork')             # Name of the player
root.geometry("890x540")       # Size of the player
root.resizable(0, 0)

#create menu
my_menu=Menu(root)
root.config(menu=my_menu)

#add song function
def add_song():
	song = filedialog.askopenfilename(initialdir="assets/audio/",title="Choose A song",filetypes=(("mp3 Files","*.mp3"),("wav files","*.wav"),("m4a files","*.m4a"),("ogg files","*.ogg"),))
	song = song.replace("/home/ashwani/hamr-project-final/assets/audio1/","")
	song_box.insert(END,song)

#Add song menu
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="File",menu=add_song_menu)
add_song_menu.add_command(label="Add to List",command=add_song)
add_song_menu.add_command(label="Exit",command=root.quit)


#
def github_link():
	webbrowser.open_new("https://github.com/ashwani-rathee/SinFork")

def contact():
	pass
	
help_menu = Menu(my_menu)
my_menu.add_cascade(label="Help",menu=help_menu)
help_menu.add_command(label="Github",command=github_link)
help_menu.add_command(label="Contact",command=contact)
photo = PhotoImage(file = "assets/icons/icon.png")
root.iconphoto(False, photo)
#Initialize pygame.Mixer
#root.configure(background='gray')

# #Audio Tsne
# T_Sne=Menu(my_menu)
# my_menu.add_cascade(label="T_Sne",menu=T_Sne)
# T_Sne.add_command(label="Select the folder",command=)

#
pygame.mixer.init()



def play():
	song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	pygame.mixer.music.load(song)
	pygame.mixer.music.play(loops=0)

def waveplotplot():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr =librosa.load(song)
	plt.figure()
	librosa.display.waveplot(y=x, sr=sr)
	plt.show()
	return
def onsetplot():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	y,sr =librosa.load(song)
	o_env = librosa.onset.onset_strength(y, sr=sr)
	times = librosa.times_like(o_env, sr=sr)
	onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
	D = np.abs(librosa.stft(y))
	fig, ax = plt.subplots(nrows=2, sharex=True)
	librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),x_axis='time', y_axis='log', ax=ax[0])
	ax[0].set(title='Power spectrogram')
	ax[0].label_outer()
	ax[1].plot(times, o_env, label='Onset strength')
	ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
	ax[1].legend()
	plt.show()

def stop():
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)

def onset_detect():
	x,sr=librosa.load(song_box.get())


def essentia_example():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	loader=essentia.standard.MonoLoader(filename=song)
	audio=loader()
	plot(audio[1*44100:2*44100])
	plt.title("This is how the 2nd second of this audio looks like:")
	plt.show()

def essentia_example2():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	loader=essentia.standard.MonoLoader(filename=song)
	audio=loader()
	w = Windowing(type = 'hann')
	spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
	mfcc = MFCC()
	frame = audio[6*44100 : 6*44100 + 1024]
	spec = spectrum(w(frame))
	mfcc_bands, mfcc_coeffs = mfcc(spec)

	plot(spec)
	plt.title("The spectrum of a frame:")
	show()

	plot(mfcc_bands)
	plt.title("Mel band spectral energies of a frame:")
	show()

	plot(mfcc_coeffs)
	plt.title("First 13 MFCCs of a frame:")
	show()

#RMSE
def rmse():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	y,sr =librosa.load(song)
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rms(S=S)
	fig, ax = plt.subplots(nrows=2, sharex=True)
	times = librosa.times_like(rms)
	ax[0].semilogy(times, rms[0], label='RMS Energy')
	ax[0].set(xticks=[])
	ax[0].legend()
	ax[0].label_outer()
	librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax[1])
	ax[1].set(title='log Power spectrogram')
	S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
	librosa.feature.rms(S=S)
	plt.show()
##
#Beat_Detectiom
def beat_detection():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	loader=essentia.standard.MonoLoader(filename=song)
	# Compute beat positions and BPM
	rhythm_extractor = RhythmExtractor2013(method="multifeature")
	bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
	marker = AudioOnsetsMarker(onsets=beats, type='beep')
	marked_audio = marker(audio)
	MonoWriter(filename='audio/dubstep_beats.flac')(marked_audio)
	plot(audio)
	for beat in beats:
		plt.axvline(x=beat*44100, color='red')
	plt.title("Audio waveform and the estimated beat positions")
	plt.show()
##
def predominant_melody():
	hopSize = 128
	frameSize = 2048
	sampleRate = 44100
	guessUnvoiced = True # read the algorithm's reference for more details
	run_predominant_melody = PitchMelodia(guessUnvoiced=guessUnvoiced,frameSize=frameSize,hopSize=hopSize);
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	audio = MonoLoader(filename = song, sampleRate=22050)()
	audio = EqualLoudness()(audio)
	pitch, confidence = run_predominant_melody(audio)
	n_frames = len(pitch)
	print("number of frames: %d" % n_frames)
	fig = plt.figure()
	plot(range(n_frames), pitch, 'b')
	n_ticks = 10
	xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
	xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
	xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
	plt.xticks(xtick_locs, xtick_lbls)
	ax = fig.add_subplot(111)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Pitch (Hz)')
	#suptitle("Predominant melody pitch")
	plt.title("Predominant melody pitch")

def predominant_melody2():
	hopSize = 128
	frameSize = 2048
	sampleRate = 44100
	guessUnvoiced = True # read the algorithm's reference for more details
	run_predominant_melody = PitchMelodia(guessUnvoiced=guessUnvoiced,frameSize=frameSize,hopSize=hopSize);
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	audio = MonoLoader(filename = song, sampleRate=22050)()
	audio = EqualLoudness()(audio)
	pitch, confidence = run_predominant_melody(audio)
	n_frames = len(pitch)
	print("number of frames: %d" % n_frames)
	fig = plt.figure()
	plot(range(n_frames), confidence, 'b')
	n_ticks = 10
	xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
	xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
	xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
	plt.xticks(xtick_locs, xtick_lbls)
	ax = fig.add_subplot(111)
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Confidence')
	#suptitle("Predominant melody pitch confidence")
	plt.title("Predominant melody pitch confidence")
	plt.show()	

#discontinuity detector
def discontinuity_detector():
	pass
	# fs = 44100.
	# song=song =song_box.get(ACTIVE)
	# song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	# audio = MonoLoader(filename = song, sampleRate=22050)()
	# originalLen = len(audio)
	# startJumps = np.array([originalLen / 4, originalLen / 2])
	# groundTruth = startJumps / float(fs)

	# for startJump in startJumps:
 #    # make sure that the artificial jump produces a prominent discontinuity
 #    	if audio[startJump] > 0:
 #        	end = next(idx for idx, i in enumerate(audio[startJump:]) if i < -.3)
 #    	else:
 #        	end = next(idx for idx, i in enumerate(audio[startJump:]) if i > .3)

 #    endJump = startJump + end
 #    audio = esarr(np.hstack([audio[:startJump], audio[endJump:]]))


	# for point in groundTruth:
 #    	l1 = plt.axvline(point, color='g', alpha=.5)

	# times = np.linspace(0, len(audio) / fs, len(audio))
	# plt.plot(times, audio)
	# plt.title('Signal with artificial clicks of different amplitudes')
	# l1.set_label('Click locations')
	# plt.legend()






####ending of detector

#MFCC
def mfcc_coeffs():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)	
	S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128,fmax=8000)
	mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
	fig, ax = plt.subplots()
	img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
	fig.colorbar(img, ax=ax)
	ax.set(title='MFCC')
	m_slaney = librosa.feature.mfcc(y=x, sr=sr, dct_type=2)
	m_htk = librosa.feature.mfcc(y=x, sr=sr, dct_type=3)
	fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
	img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
	ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
	fig.colorbar(img, ax=[ax[0]])
	img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
	ax[1].set(title='HTK-style (dct_type=3)')
	fig.colorbar(img2, ax=[ax[1]])
	plt.show()
##

#BPM HISTORGRAM
def beat_histo():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)	
	rhythm_extractor = RhythmExtractor2013(method="multifeature")
	bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
	peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = BpmHistogramDescriptors()(beats_intervals)
	fig, ax = plt.subplots()
	ax.bar(range(len(histogram)), histogram, width=1)
	ax.set_xlabel('BPM')
	ax.set_ylabel('Frequency')
	plt.title("BPM histogram")
	ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
	ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
	plt.show()
##

##
##Fourier Transform function
def fourier_transform():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	X=scipy.fft(x)
	X_mag=np.absolute(X)	
	f=np.linspace(0,sr,len(X_mag))#frequency variable
	plt.figure(figsize=(13, 5))
	plt.plot(f, X_mag) # magnitude spectrum
	plt.xlabel('Frequency (Hz)')
	plt.title("fourier_transform")
	plt.show()
###

#STFT TRANSFORM
def stftfourier_transform():
	song=song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	X=scipy.stft(x)
	X_mag=np.absolute(X)	
	f=np.linspace(0,sr,len(X_mag))#frequency variable
	plt.figure(figsize=(13, 5))
	plt.plot(f, X_mag) # magnitude spectrum
	plt.xlabel('Frequency (Hz)')
	plt.title("stftfourier_transform")
	plt.show()
##

#Spectrogram
def spectrogram():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)	
	hop_length = 512
	n_fft = 2048
	X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
	S = librosa.amplitude_to_db(abs(X))
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title("spectrogram")
	plt.show()
##

#Mel-Spectrogram
def mel_spectrogram():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)	
	hop_length = 256
	n_fft = 2048
	X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
	S = librosa.feature.melspectrogram(x, sr=sr, n_fft=4096, hop_length=hop_length)	
	logS = librosa.power_to_db(abs(S))
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(logS, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
	plt.colorbar(format='%+2.0f dB')
	plt.title("mel_spectrogram")
	plt.show()	
##

#Constant-Q-Transform
def cqt_spectrogram():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	fmin = librosa.midi_to_hz(36)
	C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72)
	logC = librosa.amplitude_to_db(abs(C))
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
	plt.colorbar(format='%+2.0f dB')
	plt.title("cqt_spectrogram")
	plt.show()
##

#Chromatogram_STFT
def chromatogram_stft():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	hop_length = 256
	chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
	plt.title("chromatogram_stft")
	plt.show()
##

#Chromatogram_CQT
def chromatogram_cqt():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	hop_length = 256
	chromagram = librosa.feature.chroma_cqt(x, sr=sr, hop_length=hop_length)
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
	plt.title("chromatogram_cqt")
	plt.show()
##

#Chromatogram_CENS
def chromatogram_cens():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)	
	fmin = librosa.midi_to_hz(36)
	C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72)
	logC = librosa.amplitude_to_db(abs(C))
	plt.figure(figsize=(15, 5))
	librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note', fmin=fmin, cmap='coolwarm')
	plt.colorbar(format='%+2.0f dB')
	plt.title("chromatogram_cens")
	plt.show()
##

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Spectral_centroid
def spectral_centroid():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]		
	frames = range(len(spectral_centroids))
	t = librosa.frames_to_time(frames)
	librosa.display.waveplot(x, sr=sr, alpha=0.4)
	plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes
	plt.title("spectral_centroid")
	plt.show()
## 

#Spectral Bandwidth
def spectral_bandwidth():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]		
	frames = range(len(spectral_centroids))
	t = librosa.frames_to_time(frames)
	spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
	spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
	spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
	librosa.display.waveplot(x, sr=sr, alpha=0.4)
	plt.plot(t, normalize(spectral_bandwidth_2), color='r')
	plt.plot(t, normalize(spectral_bandwidth_3), color='g')
	plt.plot(t, normalize(spectral_bandwidth_4), color='y')
	plt.legend(('p = 2', 'p = 3', 'p = 4'))		
	plt.title("spectral_bandwidth")
	plt.show()
##

#Spectral Contrast
def spectral_contrast():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	# frames = range(len(spectral_centroids))
	# t = librosa.frames_to_time(frames)
	spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
	plt.imshow(normalize(spectral_contrast, axis=1), aspect='auto', origin='lower', cmap='coolwarm')
	plt.title("spectral_contrast")
	plt.show()
##

#Spectral Rolloff
def spectral_rolloff():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr=librosa.load(song)
	spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]		
	frames = range(len(spectral_centroids))
	t = librosa.frames_to_time(frames)
	spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
	librosa.display.waveplot(x, sr=sr, alpha=0.4)
	plt.plot(t, normalize(spectral_rolloff), color='r')
	plt.title("Spectral_rolloff plot")
	plt.show()
##

Label(root,text="Plots: ").grid(row=0,column=0)
Label(root,text="Misc.: ").grid(row=1,column=0)
Label(root,text="Transforms: ").grid(row=2,column=0)
Label(root,text="Spectrogram: ").grid(row=3,column=0)
Label(root,text="Chromatogram: ").grid(row=4,column=0)
Label(root,text="Spectral Features: ").grid(row=5,column=0)
#Label(root,text="Rhythm Features: ").grid(row=6,column=0)
##waveplot
waveplot_button = Button(master = root,  command = waveplotplot, height = 1,  width = 15, text = "Waveplot") 
waveplot_button.grid(row=0,column=1)

#onset_detection
onset_button = Button(master = root, command = onsetplot, height = 1,   width = 15, text = "Onsets") 
onset_button.grid(row=0,column=2)

#essentia-example
essentia_example=Button(master = root, command = essentia_example,  height = 1,   width = 15, text = "EExample") 
essentia_example.grid(row=0,column=3)

#example2
essentia_example2=Button(master = root,  command = essentia_example2, height = 1, width = 15, text = "EExample2") 
essentia_example2.grid(row=0,column=5)

#ENERGY
##
#Beat Detection
beat_detection=Button(master = root,  command = beat_detection, height = 1, width = 15, text = "Beat_Detection") 
beat_detection.grid(row=1,column=4)
##
#BPM Histogran
#Beat Detection
beat_histo=Button(master = root,  command = beat_histo, height = 1, width = 15, text = "BPM Histogram") 
beat_histo.grid(row=1,column=5)
##
#RMSE
rmse=Button(master = root,  command = rmse, height = 1, width = 15, text = "RMSE") 
rmse.grid(row=0,column=4)
##

#Predominant melody
Predominant_melody=Button(master = root,  command = predominant_melody, 
height = 1,  
                     width = 15, 
                     text = " Pred_melody") 
Predominant_melody.grid(row=1,column=1)

#PredominantPitch Melody COnfidence
Predominant_melody2=Button(master = root,  command = predominant_melody2, height = 1,  width = 15, text = " Pred_melody2") 
Predominant_melody2.grid(row=1,column=2)

#Disontinuty Detector
disontinuty_detector=Button(master = root,  command = discontinuity_detector, height = 1,  width = 15,  text = "DiscontinuityDetect") 
disontinuty_detector.grid(row=1,column=3)
##

#MFCC
mfcc_coeffs=Button(master = root,  command = mfcc_coeffs, height = 1,  width = 15,  text = "MFCC") 
mfcc_coeffs.grid(row=1,column=3)
##

#Fourier Transform
fourier_transform=Button(master = root,command = fourier_transform, height = 1,width = 15,text = "Fourier_Transform") 
fourier_transform.grid(row=2,column=1)
##

# #STFT
# stftfourier_transform=Button(master = root,command = stftfourier_transform, height = 1,width = 15,text = "STFT_Transform") 
# stftfourier_transform.grid(row=2,column=2)
# ##

#Spectrogram
spectrogram=Button(master = root,text = "Spectrogram",command = spectrogram, height = 1,width = 15) 
spectrogram.grid(row=3,column=1)
##

#Mel_Spectrogram
mel_spectrogram=Button(master = root,text = "Mel_Spectrogram",command = mel_spectrogram, height = 1,width = 15) 
mel_spectrogram.grid(row=3,column=2)
##

#Constant Q-Transform
cqt_spectrogram=Button(master = root,text = "CQT_Spectrogram",command = cqt_spectrogram, height = 1,width = 15) 
cqt_spectrogram.grid(row=3,column=3)
##

#Chroma STFT
chroma_stft=Button(master = root,text = "Chromatogram_cqt",command = chromatogram_cqt, height = 1,width = 15) 
chroma_stft.grid(row=4,column=1)
##

#Chroma CQT
chroma_cqt=Button(master = root,text = "Chromatogram_stft",command = chromatogram_stft, height = 1,width = 15) 
chroma_cqt.grid(row=4,column=2)
##
#CHromatogram_CENS
chromatogram_cens=Button(master = root,text = "Chromatogram_CENS",command = chromatogram_cens, height = 1,width = 15) 
chromatogram_cens.grid(row=4,column=3)
##

##Sepctral Features
#Spectral Centroid
spectral_centroid=Button(master = root,text = "Spectral_Centroid",command = spectral_centroid, height = 1,width = 15) 
spectral_centroid.grid(row=5,column=1)
##

#Spectral Bandwidth
spectral_bandwidth=Button(master = root,text = "Spectral_Bandwidth",command = spectral_bandwidth, height = 1,width = 15) 
spectral_bandwidth.grid(row=5,column=2)
##

#spectral contrast
spectral_contrast=Button(master = root,text = "Spectral_Contrast",command = spectral_contrast, height = 1,width = 15) 
spectral_contrast.grid(row=5,column=3)
##

#Spectral Rolloff
spectral_rolloff=Button(master = root,text = "Spectral_rolloff",command = spectral_rolloff, height = 1,width = 15) 
spectral_rolloff.grid(row=5,column=4)
##

##

#create PLayist Box
song_box =Listbox(root,bg="white",fg="black",selectbackground="gray",font="ubuntu",height=15,width=15)
song_box.grid(column=0, row=10, sticky=(N,W,E,S))
for filename in get_filenames():
    song_box.insert(END, filename)




###DETAILS ABOUT THE SONG
def get_data():
	song=song_box.get(ACTIVE)
	x,sr=librosa.load(f"/home/ashwani/hamr-project-final/assets/audio1/{song}")
	data="Details about this song:\n"+"Length of the Audio Array:\n"+str(len(x))+"\n"+"Sample Rate:\n"+str(sr)+"\n"+"Librosa Version:\n"+str(librosa.__version__)+"Audio's Duration:\n"+str(librosa.get_duration(x, sr))
	return data

# details=Text(root,height=10,width=30)
# l=Label(root,text="Details about this song")
# l.config(font=("ubuntu",14))
# data=get_data()
# details.insert(END,data)
# details.grid(row=5,column=1)

#create player control buttons
play_btn_img = PhotoImage(file='assets/icons/music.png')
stop_btn_img = PhotoImage(file='assets/icons/pause.png')

#create player control frame
controls_frame=Frame(root)
controls_frame.grid(row=11,column=0,padx=10,pady=10)

#create player control buttons
play_button = Button(controls_frame,image=play_btn_img,borderwidth=0,command=play)
stop_button = Button(controls_frame,image=stop_btn_img,borderwidth=0,command=stop)

play_button.grid(row=1,column=0,padx=10)
stop_button.grid(row=1,column=1,padx=10)

root.mainloop()