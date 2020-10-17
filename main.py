#main file and starting point
import numpy as np
import pandas as pd
import librosa
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import time
import pygame
root=Tk()
#root=Tk(className='sonarawaaz')
root.title('HAMR-PROJECT')
#root.iconbitmap('@icon.ico')

pygame.mixer.init()
def play():
	#pygame.mixer.music.load('/home/ashwani/hamr-project-final/kush.mp3')
	#pygame.mixer.music.play(Loops=0)
	pygame.mixer.music.load('/home/ashwani/hamr-project-final/kush.mp3')  # Load a sound.
	pygame.mixer.music.play()
	time.sleep(3)
def stop():
	pygame.mixer.music.stop()

root.geometry("600x600")
# Label(root,text='Audio-1').grid(row=0)
# Label(root,text='Audio-2').grid(row=1)
# filename1=Entry(root)
# filename2=Entry(root)
# filename1.grid(row=0,column=1)
# filename2.grid(row=1,column=1)

my_button=Button(root,text="Play Song",font=("Helvetica",32),command=play)
my_button.pack(pady=20)

stop_button=Button(root,text="Stop",command=stop)
stop_button.pack(pady=20)

root.mainloop()