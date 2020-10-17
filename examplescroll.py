from tkinter import *
from tkinter import ttk
import os

def get_filenames():
    path = r"/home/ashwani/hamr-project-final/assets/audio/"
    return os.listdir(path)


root = Tk()
l = Listbox(root, height=5)
l.grid(column=0, row=0, sticky=(N,W,E,S))
s = ttk.Scrollbar(root, orient=VERTICAL, command=l.yview)
s.grid(column=1, row=0, sticky=(N,S))
l['yscrollcommand'] = s.set
ttk.Sizegrip().grid(column=1, row=1, sticky=(S,E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
for filename in get_filenames():
    l.insert(END, filename)
root.mainloop()