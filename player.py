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
from matplotlib import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import librosa.display

def get_filenames():
    path = r"/home/ashwani/hamr-project-final/assets/audio1"
    return os.listdir(path)

root=Tk()
root.title('Sinfork')             # Name of the player
root.geometry("960x540")       # Size of the player
root.resizable(0, 0)

photo = PhotoImage(file = "assets/icons/icon.png")
root.iconphoto(False, photo)
#Initialize pygame.Mixer
#root.configure(background='gray')

pygame.mixer.init()

#add song function
def add_song():
	song = filedialog.askopenfilename(initialdir="assets/audio/",title="Choose A song",filetypes=(("mp3 Files","*.mp3"),("wav files","*.wav"),("m4a files","*.m4a"),("ogg files","*.ogg"),))
	song = song.replace("/home/ashwani/hamr-project-final/assets/audio1/","")
	song_box.insert(END,song)

def play():
	song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	pygame.mixer.music.load(song)
	pygame.mixer.music.play(loops=0)

def spectrogramplot():
	song=song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio1/{song}'
	x,sr =librosa.load(song)
	plt.figure()
	librosa.display.waveplot(y=x, sr=sr)
	plt.show()
	return

def stop():
    pygame.mixer.music.stop()
    song_box.selection_clear(ACTIVE)

def onset_detect():
	x,sr=librosa.load(song_box.get())


## button that displays the plot 
spectrogram_button = Button(master = root,  
                     command = spectrogramplot, 
                     height = 2,  
                     width = 10, 
                     text = "Plot") 
spectrogram_button.grid(row=4,column=0,padx=10,pady=10)

#create PLayist Box
song_box =Listbox(root,bg="white",fg="black",selectbackground="gray")
song_box.grid(column=0, row=0, sticky=(N,W,E,S),padx=10,pady=10)
for filename in get_filenames():
    song_box.insert(END, filename)




###DETAILS ABOUT THE SONG
def get_data():
	song=song_box.get(ACTIVE)
	x,sr=librosa.load(f"/home/ashwani/hamr-project-final/assets/audio1/{song}")
	data="Details about this song:\n"+"Length of the Audio Array:\n"+str(len(x))+"\n"+"Sample Rate:\n"+str(sr)+"\n"+"Librosa Version:\n"+str(librosa.__version__)
	return data

details=Text(root,height=10,width=50)
l=Label(root,text="Details about this song")
l.config(font=("ubuntu",14))
data=get_data()
details.insert(END,data)
details.grid(row=0,column=1)




#create player control buttons
play_btn_img = PhotoImage(file='assets/icons/music.png')
stop_btn_img = PhotoImage(file='assets/icons/pause.png')

#create player control frame
controls_frame=Frame(root)
controls_frame.grid(row=1,column=0,padx=10)

#create player control buttons
play_button = Button(controls_frame,image=play_btn_img,borderwidth=0,command=play)
stop_button = Button(controls_frame,image=stop_btn_img,borderwidth=0,command=stop)

play_button.grid(row=1,column=0,padx=10)
stop_button.grid(row=1,column=1,padx=10)


#create menu
my_menu=Menu(root)
root.config(menu=my_menu)

#Add song menu
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="File",menu=add_song_menu)
add_song_menu.add_command(label="Add to List",command=add_song)

# #Audio Analysismenu
# analysis=Menu(root)
# root.config(menu=my_menu)
# add_song_menu = Menu(my_menu)
# my_menu.add_cascade(label="File",menu=add_song_menu)
# add_song_menu.add_command(label="Add to List",command=add_song)

root.mainloop()