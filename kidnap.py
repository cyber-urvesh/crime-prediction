import tkinter as tk
# from tkinter import *
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re
import random
import os
import cv2


window = tk.Tk()
window.geometry("400x400")
window.title("kidnap")
window.configure(background="grey")
image2 = Image.open('kidnapping.png')
image2 = image2.resize((400, 400), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(window, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) 
window.mainloop()