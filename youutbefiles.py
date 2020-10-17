import webbrowser
from tkinter import *

window = Tk()
window.title("DdP Program")
window.geometry("300x100")


def onClick(x):
    webbrowser.open(x,new=1)


label = Label(text="some text")
label.pack()

labelOne = Label(text="A little bit more text")
labelOne.pack()


url = "http://docs.python.org/library/webbrowser.html"

click = Button(text="Clica Aqui!", command=lambda: onClick(url))
click.pack()

window.mainloop()
