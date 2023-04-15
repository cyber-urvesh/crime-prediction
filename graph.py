
import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
#import video_capture as value
#import lecture_details as detail_data
#import video_second as video1

#import lecture_video  as video

global fn
fn = ""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="brown")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Crime Prediction System")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('BG1.jpg')
image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
#
label_l1 = tk.Label(root, text="Crime Prediction System",font=("Times New Roman", 35, 'bold'),
                    background="#152238", fg="white", width=20, height=1)
label_l1.place(x=400, y=20)

#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def theft():
    from subprocess import call
    call(["python","theft1.py"])
    
  
def murder():
    from subprocess import call
    call(["python","murder1.py"])
    
def kidnap():
    from subprocess import call
    call(["python","kidnap.py"])
    
def rape():
    from subprocess import call
    call(["python","rape1.py"])
    
def window():
  root.destroy()

button1 = tk.Button(root, text="Theft", command=theft, width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button1.place(x=100, y=160)

button2 = tk.Button(root, text="Murder",command=murder,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button2.place(x=100, y=240)

button3 = tk.Button(root, text="Kidnap",command=kidnap,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button3.place(x=100, y=330)

button4 = tk.Button(root, text="Rape",command=rape,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button4.place(x=100, y=420)

button4 = tk.Button(root, text="Exit",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button4.place(x=100, y=510)
root.mainloop()