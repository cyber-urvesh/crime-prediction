import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from tkinter import *

from subprocess import call
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')
  
    
root = tk.Tk()
root.title("Crime Prediction System")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
image2 =Image.open(r'BG.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)


label_l1 = tk.Label(root, text="Crime Prediction System",font=("Times New Roman", 35, 'bold'),
                    background="#152238", fg="white", width=25)
label_l1.place(x=300, y=10)






#--------------------------------------------------------------
def theft_Display():
    columns = [ 'ID','Area_Name', 'Year', 'Sub_Group_Name','Auto_Theft_Stolen']
    print(columns)

    data1 = pd.read_csv(r"updatedtheft.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1
    ID = data1.iloc[:, 0]
    Area_Name = data1.iloc[:, 1]
    Year = data1.iloc[:, 2]
    Sub_Group_Name = data1.iloc[:, 3]
    Auto_Theft_Stolen = data1.iloc[:, 4]


    display = tk.LabelFrame(root, width=600, height=400, )
    display.place(x=300, y=150)
    display['borderwidth'] = 15

    tree = ttk.Treeview(display, columns=(
    'ID','Area_Name', 'Year', 'Sub_Group_Name','Auto_Theft_Stolen'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3","4","5")
    tree.column("1", width=100)
    tree.column("2", width=100)
    tree.column("3", width=200)
    tree.column("4", width=100)
    tree.column("5", width=100)
    
    tree.heading("1", text="ID")
    tree.heading("2", text="Area_Name")
    tree.heading("3", text="Year")
    tree.heading("4", text="Sub_Group_Name")
    tree.heading("5", text="Auto_Theft_Stolen")
    

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 1865):
        tree.insert("", 'end', values=(
            ID[i], Area_Name[i], Year[i],Sub_Group_Name[i],Auto_Theft_Stolen[i]))
        i = i + 1
        print(i)

def murder_Display():
    columns = ['TID', 'Tweets', 'Label']
    print(columns)

    data1 = pd.read_csv(r"Murder_victim_age_sex.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    Area_Name = data1.iloc[:, 0]
    Year = data1.iloc[:, 1]
    Sub_Group_Name = data1.iloc[:, 2]
    Victims_Total = data1.iloc[:, 3]


    display = tk.LabelFrame(root, width=600, height=400, )
    display.place(x=300, y=150)
    display['borderwidth'] = 15

    tree = ttk.Treeview(display, columns=(
    'Area_Name', 'Year','Sub_Group_Name','Victims_Total'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3","4")
    tree.column("1", width=200)
    tree.column("2", width=100)
    tree.column("3", width=200)
    tree.column("4", width=100)
   
    

    tree.heading("1", text="Area_Name")
    tree.heading("2", text="Year")
    tree.heading("3", text="Sub_Group_Name")
    tree.heading("4", text="Victims_Total")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 1019):
        tree.insert("", 'end', values=(
            Area_Name[i], Year[i], Sub_Group_Name[i],Victims_Total[i]))
        i = i + 1
        print(i)

def kidn_Display():
    columns = ['TID', 'Tweets', 'Label']
    print(columns)

    data1 = pd.read_csv(r"kidnapping.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    Area_Name = data1.iloc[:, 0]
    Year = data1.iloc[:, 1]
    Sub_Group_Name = data1.iloc[:, 2]
    K_A_Cases_Reported = data1.iloc[:, 3]


    display = tk.LabelFrame(root, width=600, height=400, )
    display.place(x=300, y=150)
    display['borderwidth'] = 15

    tree = ttk.Treeview(display, columns=(
    'Area_Name', 'Year','Sub_Group_Name','K_A_Cases_Reported'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3","4")
    tree.column("1", width=200)
    tree.column("2", width=100)
    tree.column("3", width=200)
    tree.column("4", width=100)

    

    tree.heading("1", text="Area_Name")
    tree.heading("2", text="Year")
    tree.heading("3", text="Sub_Group_Name")
    tree.heading("4", text="K_A_Cases_Reported")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 3570):
        tree.insert("", 'end', values=(
            Area_Name[i], Year[i], Sub_Group_Name[i],K_A_Cases_Reported[i]))
        i = i + 1
        print(i)

def women_Display():
    columns = ['Area_Name', 'Year','Sub_Group_Name','Rape_Cases_Reported']
    print(columns)

    data1 = pd.read_csv(r'rape.csv', encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    Area_Name = data1.iloc[:, 0]
    Year = data1.iloc[:, 1]
    Sub_Group_Name = data1.iloc[:, 2]
    Rape_Cases_Reported = data1.iloc[:, 3]


    display = tk.LabelFrame(root, width=600, height=400, )
    display.place(x=300, y=150)
    display['borderwidth'] = 15

    tree = ttk.Treeview(display, columns=(
    'Area_Name', 'Year','Sub_Group_Name','Rape_Cases_Reported'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3","4")
    tree.column("1", width=200)
    tree.column("2", width=100)
    tree.column("3", width=200)
    tree.column("4", width=100)
    

    tree.heading("1", text="Area_Name")
    tree.heading("2", text="Year")
    tree.heading("3", text="Sub_Group_Name")
    tree.heading("4", text="Rape_Cases_Reported")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 1865):
        tree.insert("", 'end', values=(
            Area_Name[i], Year[i],Sub_Group_Name[i],Rape_Cases_Reported[i]))
        i = i + 1
        print(i)
# -------------------------------------------------------------------------------------------------------------       
def Data_Preprocessing_theft():
    data = pd.read_csv('last.csv')
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Auto_Theft_Stolen','ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['ID']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=5)
        
def Model_Training_theft():
    data = pd.read_csv("last.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
   

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Auto_Theft_Stolen','ID'], axis=1)
    
    data = data.dropna()

    print(type(x))
    y = data['Auto_Theft_Stolen']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear', C = 1.0 , gamma = 'scale')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as crime_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    from joblib import dump
    dump (svcclassifier,"crime_prediction_MODEL.joblib")
    print("Model saved as crime_prediction_MODEL.joblib")
     
def call_file_theft():
    #import Check_Heart
    #Check_Heart.Train()
    Area_Name = tk.StringVar()
    Group_Name = tk.StringVar()
    Sub_Group_Name = tk.StringVar()
    Year = tk.IntVar()
    
    def Detect():
        e1=Area_Name.get()
        print(e1)
        e2=Year.get()
        print(e2)
        #e3=Group_Name.get()
        #print(e3)
        e4=Sub_Group_Name.get()
        print(e4)
       
        #########################################################################################
        
        from joblib import dump , load
        a1=load('crime_prediction_MODEL.joblib')
        v= a1.predict([[e1, e2,e4]])
        #v=[v]
        
        #predicted_price=listToString(v) 
        yes = tk.Label(root,text="Predicted Auto_Theft_Stolen is  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=10,y=600)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.grid(row=0, column=0, sticky='nw')
    frame_alpr.place(x=400, y=80)
    l1=tk.Label(root,text="Area_Name",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=500,y=100)
    #Area_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Area_Name)
    #Area_Name.place(x=200,y=1)
    R5 = Radiobutton(root, text="Andaman & Nicobar Islands", variable=Area_Name, value=1).place(x=800,y=100)
    R6 = Radiobutton(root, text="Andhra Pradesh", variable=Area_Name, value=2).place(x=800,y=150)
    R7 = Radiobutton(root, text="Arunachal Pradesh", variable=Area_Name, value=3).place(x=800,y=200)
    R8 = Radiobutton(root, text="Assam", variable=Area_Name, value=4).place(x=1000,y=100)
    

    l2=tk.Label(root,text="Year",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=500,y=250)
    Year=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Year)
    Year.place(x=800,y=250)
    

    
    l4=tk.Label(root,text="Sub_Group_Name",background="purple",font=('times', 20, ' bold '))
    l4.place(x=500,y=300)
   # Sub_Group_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Sub_Group_Name)
    #Sub_Group_Name.place(x=200,y=160)
    R1 = Radiobutton(root, text="1. Motor Cycles/ Scooters", variable=Sub_Group_Name, value=1).place(x=800,y=300)
    R2 = Radiobutton(root, text="2. Motor Car/Taxi/Jeep", variable=Sub_Group_Name, value=2).place(x=800,y=350)
    R3 = Radiobutton(root, text="3. Buses", variable=Sub_Group_Name, value=3).place(x=800,y=400)
    R4 = Radiobutton(root, text="4. Goods carrying vehicles (Trucks/Tempo etc)", variable=Sub_Group_Name, value=4).place(x=800,y=450)
    

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=700,y=600)
 
        
 
    
def Data_Preprocessing_murder():
    data = pd.read_csv('Murder_victim_age_sex.csv')
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Victims_Total','ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['ID']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=5)
        
def Model_Training_murder():
    data = pd.read_csv("Murder_victim_age_sex.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
   

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Victims_Total','ID'], axis=1)
    
    data = data.dropna()

    print(type(x))
    y = data['Victims_Total']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear', C = 1.0 , gamma = 'scale')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as murder_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    from joblib import dump
    dump (svcclassifier,"murder_prediction_MODEL.joblib")
    print("Model saved as murder_prediction_MODEL.joblib")

def call_file_murder():
    #import Check_Heart
    #Check_Heart.Train()
    Area_Name = tk.StringVar()
    Group_Name = tk.StringVar()
    Sub_Group_Name = tk.StringVar()
    Year = tk.IntVar()
    
    def Detect():
        e1=Area_Name.get()
        print(e1)
        e2=Year.get()
        print(e2)
        #e3=Group_Name.get()
        #print(e3)
        e4=Sub_Group_Name.get()
        print(e4)
       
        #########################################################################################
        
        from joblib import dump , load
        a1=load('murder_prediction_MODEL.joblib')
        v= a1.predict([[e1, e2,e4]])
        #v=[v]
        
        #predicted_price=listToString(v) 
        yes = tk.Label(root,text="Predicted Total Murder Cases is  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=10,y=600)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.grid(row=0, column=0, sticky='nw')
    frame_alpr.place(x=400, y=80)
    l1=tk.Label(root,text="Area_Name",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=500,y=100)
    #Area_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Area_Name)
    #Area_Name.place(x=200,y=1)
    R5 = Radiobutton(root, text="Andaman & Nicobar Islands", variable=Area_Name, value=1).place(x=800,y=100)
    R6 = Radiobutton(root, text="Andhra Pradesh", variable=Area_Name, value=2).place(x=800,y=150)
    R7 = Radiobutton(root, text="Arunachal Pradesh", variable=Area_Name, value=3).place(x=800,y=200)
    R8 = Radiobutton(root, text="Assam", variable=Area_Name, value=4).place(x=1000,y=100)
    

    l2=tk.Label(root,text="Year",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=500,y=250)
    Year=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Year)
    Year.place(x=800,y=250)
    

    
    l4=tk.Label(root,text="Sub_Group_Name",background="purple",font=('times', 20, ' bold '))
    l4.place(x=500,y=300)
   # Sub_Group_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Sub_Group_Name)
    #Sub_Group_Name.place(x=200,y=160)
    R1 = Radiobutton(root, text="1. Female Murders", variable=Sub_Group_Name, value=1).place(x=800,y=300)
    R2 = Radiobutton(root, text="2. Male Murders", variable=Sub_Group_Name, value=2).place(x=800,y=350)
    R3 = Radiobutton(root, text="3. Total Murders", variable=Sub_Group_Name, value=3).place(x=800,y=400)
   

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=700,y=600)
        

def Data_Preprocessing_kids():
    data = pd.read_csv('kidnapping.csv')
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['K_A_Cases_Reported','ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['ID']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=5)
        
def Model_Training_kids():
    data = pd.read_csv("kidnapping.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
   

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['K_A_Cases_Reported','ID'], axis=1)
    
    data = data.dropna()

    print(type(x))
    y = data['K_A_Cases_Reported']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear', C = 1.0 , gamma = 'scale')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as kidnapping_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    from joblib import dump
    dump (svcclassifier,"kidnapping_prediction_MODEL.joblib")
    print("Model saved as kidnapping_prediction_MODEL.joblib")

def call_file_kids():
    #import Check_Heart
    #Check_Heart.Train()
    Area_Name = tk.StringVar()
    Group_Name = tk.StringVar()
    Sub_Group_Name = tk.StringVar()
    Year = tk.IntVar()
    
    def Detect():
        e1=Area_Name.get()
        print(e1)
        e2=Year.get()
        print(e2)
        #e3=Group_Name.get()
        #print(e3)
        e4=Sub_Group_Name.get()
        print(e4)
       
        #########################################################################################
        
        from joblib import dump , load
        a1=load('kidnapping_prediction_MODEL.joblib')
        v= a1.predict([[e1, e2,e4]])
        #v=[v]
        
        #predicted_price=listToString(v) 
        yes = tk.Label(root,text="Predicted Kidnapping Cases are  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=20,y=500)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.grid(row=0, column=0, sticky='nw')
    frame_alpr.place(x=400, y=80)
    l1=tk.Label(root,text="Area_Name",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=500,y=100)
    #Area_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Area_Name)
    #Area_Name.place(x=200,y=1)
    R5 = Radiobutton(root, text="Andaman & Nicobar Islands", variable=Area_Name, value=1).place(x=800,y=100)
    R6 = Radiobutton(root, text="Andhra Pradesh", variable=Area_Name, value=2).place(x=800,y=150)
    R7 = Radiobutton(root, text="Arunachal Pradesh", variable=Area_Name, value=3).place(x=800,y=200)
    R8 = Radiobutton(root, text="Assam", variable=Area_Name, value=4).place(x=1000,y=100)
    

    l2=tk.Label(root,text="Year",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=500,y=250)
    Year=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Year)
    Year.place(x=800,y=250)
    

    
    l4=tk.Label(root,text="Sub_Group_Name",background="purple",font=('times', 20, ' bold '))
    l4.place(x=500,y=300)
   # Sub_Group_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Sub_Group_Name)
    #Sub_Group_Name.place(x=200,y=160)
    R1 = Radiobutton(root, text="1. For Adoption", variable=Sub_Group_Name, value=1).place(x=800,y=300)
    R2 = Radiobutton(root, text="2. For Sale", variable=Sub_Group_Name, value=2).place(x=800,y=350)
    R3 = Radiobutton(root, text="3. For Marriage", variable=Sub_Group_Name, value=3).place(x=800,y=400)
    R4 = Radiobutton(root, text="4. For begging", variable=Sub_Group_Name, value=4).place(x=800,y=450)
    

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=700,y=600)

        
def Data_Preprocessing_women():
    data = pd.read_csv('rape.csv')
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Rape_Cases_Reported','ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['ID']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=5)
        
def Model_Training_women():
    data = pd.read_csv("rape.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    #data['Auto_Theft_Stolen'] = le.fit_transform(data['Auto_Theft_Stolen'])
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    #print(data['fractal_dimension_mean'])
    #data['Thal'] = le.fit_transform(data['Thal'])
    #print("thal Encoding")
   

    #data['Thal'] = le.fit_transform(data['Thal'])
    #data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['Rape_Cases_Reported','ID'], axis=1)
    
    data = data.dropna()

    print(type(x))
    y = data['Rape_Cases_Reported']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear', C = 1.0 , gamma = 'scale')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as rape_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    from joblib import dump
    dump (svcclassifier,"rape_prediction_MODEL.joblib")
    print("Model saved as rape_prediction_MODEL.joblib")

def call_file_women():
    #import Check_Heart
    #Check_Heart.Train()
    Area_Name = tk.StringVar()
    Group_Name = tk.StringVar()
    Sub_Group_Name = tk.StringVar()
    Year = tk.IntVar()
    
    def Detect():
        e1=Area_Name.get()
        print(e1)
        e2=Year.get()
        print(e2)
        #e3=Group_Name.get()
        #print(e3)
        e4=Sub_Group_Name.get()
        print(e4)
       
        #########################################################################################
        
        from joblib import dump , load
        a1=load('crime_prediction_MODEL.joblib')
        v= a1.predict([[e1, e2,e4]])
        #v=[v]
        
        #predicted_price=listToString(v) 
        yes = tk.Label(root,text="Predicted Auto_Theft_Stolen is  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=10,y=600)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.grid(row=0, column=0, sticky='nw')
    frame_alpr.place(x=400, y=80)
    l1=tk.Label(root,text="Area_Name",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=500,y=100)
    #Area_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Area_Name)
    #Area_Name.place(x=200,y=1)
    R5 = Radiobutton(root, text="Andaman & Nicobar Islands", variable=Area_Name, value=1).place(x=800,y=100)
    R6 = Radiobutton(root, text="Andhra Pradesh", variable=Area_Name, value=2).place(x=800,y=150)
    R7 = Radiobutton(root, text="Arunachal Pradesh", variable=Area_Name, value=3).place(x=800,y=200)
    R8 = Radiobutton(root, text="Assam", variable=Area_Name, value=4).place(x=1000,y=100)
    

    l2=tk.Label(root,text="Year",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=500,y=250)
    Year=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Year)
    Year.place(x=800,y=250)
    

    
    l4=tk.Label(root,text="Sub_Group_Name",background="purple",font=('times', 20, ' bold '))
    l4.place(x=500,y=300)
   # Sub_Group_Name=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Sub_Group_Name)
    #Sub_Group_Name.place(x=200,y=160)
    R1 = Radiobutton(root, text="1. Motor Cycles/ Scooters", variable=Sub_Group_Name, value=1).place(x=800,y=300)
    R2 = Radiobutton(root, text="2. Motor Car/Taxi/Jeep", variable=Sub_Group_Name, value=2).place(x=800,y=350)
    R3 = Radiobutton(root, text="3. Buses", variable=Sub_Group_Name, value=3).place(x=800,y=400)
    R4 = Radiobutton(root, text="4. Goods carrying vehicles (Trucks/Tempo etc)", variable=Sub_Group_Name, value=4).place(x=800,y=450)
    

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=700,y=600)

   
#---------------------------------------------------------------------------------------------


button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Data_Preprocessing", command=Data_Preprocessing_theft, width=15, height=2)
button2.place(x=5, y=90)

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_murder, width=15, height=2)
button2.place(x=100, y=90)

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_kids, width=15, height=2)
button2.place(x=200, y=90)

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_women, width=15, height=2)
button2.place(x=300, y=90)


#---------------------------------------------------------------------------------------------------
button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_theft, width=15, height=2)
button3.place(x=5, y=170)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training_murder, width=15, height=2)
button3.place(x=25, y=170)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_kids, width=15, height=2)
button3.place(x=50, y=170)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_women, width=15, height=2)
button3.place(x=75, y=170)

#-------------------------------------------------------------------------------------------------



button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Theft Crime Prediction", command=call_file_theft, width=15, height=2)
button4.place(x=5, y=250)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Murder Crime Prediction", command=call_file_murder, width=15, height=2)
button4.place(x=25, y=350)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Kidnapping Crime Prediction", command=call_file_kids, width=15, height=2)
button4.place(x=50, y=250)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Rape Crime Prediction", command=call_file_women, width=15, height=2)
button4.place(x=75, y=250)





def window():
  root.destroy()
    
#label_l1 = tk.Label(root, text="Data_Display",font=("Times New Roman", 35, 'bold'),
 #                   background="#152238", fg="white")
#label_l1.place(x=25, y=100)    
button1 = tk.Button(root,command=theft_Display,text="Theft ",bg="gold",fg="black",font=("Times New Roman",15,"italic"),width=20)
button1.place(x=200,y=100)
button1 = tk.Button(root,command=murder_Display,text="Murder",bg="gold",fg="black",font=("Times New Roman",15,"italic"),width=20)
button1.place(x=430,y=100)
button1 = tk.Button(root,command=kidn_Display,text="Kidnapping",bg="gold",fg="black",font=("Times New Roman",15,"italic"),width=20)
button1.place(x=650,y=100)
button1 = tk.Button(root,command=women_Display,text="Rape Crimes",bg="gold",fg="black",font=("Times New Roman",15,"italic"),width=20)
button1.place(x=1100,y=100)

button4 = tk.Button(root,command=window,text="Exit",bg="red",fg="black",width=15,font=("Times New Roman",15,"italic"))
button4.place(x=25,y=330)





root.mainloop()