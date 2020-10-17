from guizero import App, Combo,Text,CheckBox, ButtonGroup,PushButton,info,TextBox, Picture
import os
import time
import sys

from io import StringIO



     
def counter_loop():
   import increment_test
    
   old_stdout = sys.stdout
 
# This variable will store everything that is sent to the standard output
 
   result = StringIO()
 
   sys.stdout = result
   increment_test.increment()
    #sys.stdout = old_stdout
   result_string = result.getvalue()
   counter.value = result_string # read output
        
   button.disable()
        
   

        

app = App(title="get data", width=1000, height=1000,)
counter = Text(app, size = 20, font = "Times New Roman", color="black")
button = PushButton(app, command=counter_loop, text = "Display your name")



    





app.display()