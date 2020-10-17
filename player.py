#mp3-player
from tkinter import *
from tkinter import filedialog

import pygame

root=Tk()
root.title('MP3 Player')
root.geometry("500x300")

#initialize Pygame.Mixer
pygame.mixer.init()

#add song function
def add_song():
	song = filedialog.askopenfilename(initialdir="assets/audio/",title="Choose A song",filetypes=(("mp3 Files","*.mp3"),("wav files","*.wav"),))
	song = song.replace("/home/ashwani/hamr-project-final/assets/audio/","")
	song_box.insert(END,song)

def play():
	song =song_box.get(ACTIVE)
	song = f'/home/ashwani/hamr-project-final/assets/audio/{song}'
	pygame.mixer.music.load(song)
	pygame.mixer.music.play(loops=0)


def stop():
    pygame.mixer.music.load(song)
    song_box.selection_clear(ACTIVE)

#create PLayist Box
song_box =Listbox(root,bg="white",fg="black",width=60,selectbackground="gray")
song_box.pack(pady=20)

#create player control buttons
back_btn_img = PhotoImage(file='assets/icons/backward.png')
forward_btn_img = PhotoImage(file='assets/icons/fast-forward.png')
play_btn_img = PhotoImage(file='assets/icons/music.png')
pause_btn_img = PhotoImage(file='assets/icons/pause.png')
stop_btn_img = PhotoImage(file='assets/icons/stop.png')

#create player control frame
controls_frame=Frame(root)
controls_frame.pack()

#create player control buttons
back_button = Button(controls_frame,image=back_btn_img,borderwidth=0)
forward_button = Button(controls_frame,image=forward_btn_img,borderwidth=0)
play_button = Button(controls_frame,image=play_btn_img,borderwidth=0,command=play)
pause_button = Button(controls_frame,image=pause_btn_img,borderwidth=0)
stop_button = Button(controls_frame,image=stop_btn_img,borderwidth=0,command=stop)

back_button.grid(row=0,column=0,padx=10)

play_button.grid(row=0,column=1,padx=10)
pause_button.grid(row=0,column=3,padx=10)
stop_button.grid(row=0,column=4,padx=10)

forward_button.grid(row=0,column=5,padx=10)

#create menu
my_menu=Menu(root)
root.config(menu=my_menu)

#Add song menu
add_song_menu = Menu(my_menu)
my_menu.add_cascade(label="File",menu=add_song_menu)
add_song_menu.add_command(label="Add to List",command=add_song)

root.mainloop()