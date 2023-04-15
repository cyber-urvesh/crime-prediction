
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
  


root=tk.Tk()

root.title("Crime Prediction System")
w = tk.Label(root, text="Crime Prediction System Using Machine Learning",background="skyblue",width=40,height=2,font=("Times new roman",19,"bold"))
w.pack(padx=0, side= TOP, anchor="w")



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="skyblue")


from tkinter import messagebox as ms





def Data_Preprocessing_theft():
    data = pd.read_csv('dataset/rape.csv')
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
   
    data['Area_Name'] = le.fit_transform(data['Area_Name'])
    data['Sub_Group_Name'] = le.fit_transform(data['Sub_Group_Name'])
    

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
    load.place(x=300, y=120)
        
def Model_Training_theft():
    data = pd.read_csv("dataset/last.csv")
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
    
    # label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    # label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Model Traning Completed \n Model saved as crime_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=200)
    from joblib import dump
    dump (svcclassifier,"crime_prediction_MODEL.joblib")
    print("Model Traning Completed \n Model saved as crime_prediction_MODEL.joblib")
     
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
        yes = tk.Label(root,text="Predicted Thefts are  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=600,y=500)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")

    frame_alpr.place(x=400, y=90)
    l1=tk.Label(root,text="Area_Name",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=500,y=100)
    


    monthchoosen = ttk.Combobox(root, width = 30,textvar=Area_Name )
    monthchoosen['values'] = (' Andaman & Nicobar Islands', 
                   'Andhra Pradesh',
                   'Arunachal Pradesh',
                   'Assam',
                   'Bihar',
                   'Chandigarh',
                   'Chhattisgarh',
                   'Dadra & Nagar Haveli',
                   'Delhi',
                   'Gujarat',
                   'Goa',
                   'Haryana',
                   'Himachal Pradesh',
                   'Jammu & Kashmir',
                   'Jharkhand',
                   'Karnataka',
                   'Kerala',
                   'Madhya Pradesh',
                   'Manipur',
                   'Maharashtra',
                   'Meghalaya',
                   'Mizoram',
                   'Nagaland',
                   'Odisha',
                   'Puducherry',
                   'Punjab',
                   'Rajasthan',
                   'Sikkim',
                   'Tamil Nadu',
                   'Tripura',
                   'Uttar Pradesh',
                   'Uttarakhand',
                   'West Bengal'
                       )
    monthchoosen.place(x=800,y=150)  




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
    data = pd.read_csv('dataset/Murder_victim_age_sex.csv')
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
    load.place(x=300, y=120)
        
def Model_Training_murder():
    data = pd.read_csv("dataset/Murder_victim_age_sex.csv")
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
    
    #label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    #label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Model Traning is Completed \nModel saved as murder_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=200)
    from joblib import dump
    dump (svcclassifier,"murder_prediction_MODEL.joblib")
    print("Model Traning is Completed \n Model saved as murder_prediction_MODEL.joblib")

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
        yes.place(x=600,y=500)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.place(x=400, y=90)
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
    data = pd.read_csv('dataset/kidnapping.csv')
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
    load.place(x=300, y=120)
        
def Model_Training_kids():
    data = pd.read_csv("dataset/kidnapping.csv")
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
    
    # label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    # label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Model Traning is Completed \nModel saved as kidnapping_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=200)
    from joblib import dump
    dump (svcclassifier,"kidnapping_prediction_MODEL.joblib")
    print("Model Traning Completed \n Model saved as kidnapping_prediction_MODEL.joblib")

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
        yes.place(x=600,y=500)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.place(x=400, y=90)
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
    data = pd.read_csv('dataset/rape.csv')
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
    load.place(x=300, y=120)
        
def Model_Training_women():
    data = pd.read_csv("dataset/rape.csv")
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
    
    # label4 = tk.Label(root,text =str(repo),width=40,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    # label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Model Traning is Completed \nModel saved as rape_prediction_MODEL.joblib",width=40,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=300,y=200)
    from joblib import dump
    dump (svcclassifier,"rape_prediction_MODEL.joblib")
    print("Model Traning Completed \n Model saved as rape_prediction_MODEL.joblib")

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
        a1=load('rape_prediction_MODEL.joblib')
        v= a1.predict([[e1, e2,e4]])
        #v=[v]
        
        #predicted_price=listToString(v) 
        yes = tk.Label(root,text="Predicted Rape Crimes are  "+str(v),background="red",foreground="white",font=('times', 20, ' bold '))
        yes.place(x=600,y=500)
        


    frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=900, height=600, bd=5, font=('times', 14, ' bold '),bg="tomato")
    frame_alpr.place(x=400, y=90)
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
    R1 = Radiobutton(root, text="1. Children (00-14 years)", variable=Sub_Group_Name, value=1).place(x=800,y=300)
    R2 = Radiobutton(root, text="2. Youth(15-24 years)", variable=Sub_Group_Name, value=2).place(x=800,y=350)
    R3 = Radiobutton(root, text="3. Adults(25-64)", variable=Sub_Group_Name, value=3).place(x=800,y=400)
    #R4 = Radiobutton(root, text="4. Goods carrying vehicles (Trucks/Tempo etc)", variable=Sub_Group_Name, value=4).place(x=800,y=450)
    

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=700,y=600)

   
#---------------------------------------------------------------------------------------------



def theft():
   bg = Image.open(r"BG.jpg")
   bg.resize((1500,200),Image.ANTIALIAS)
   print(w,h)
   bg_img = ImageTk.PhotoImage(bg)
   bg_lbl = tk.Label(root,image=bg_img)
   bg_lbl.place(x=0,y=93)
   button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Data_Preprocessing", command=Data_Preprocessing_theft, width=20, height=2)
   button2.place(x=15, y=120)
   button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_theft, width=20, height=2)
   button3.place(x=15, y=220)
   button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Theft Crime Prediction", command=call_file_theft, width=25, height=2)
   button4.place(x=15, y=320)

def kidn():
    bg = Image.open(r"BG.jpg")
    bg.resize((1500,200),Image.ANTIALIAS)
    print(w,h)
    bg_img = ImageTk.PhotoImage(bg)
    bg_lbl = tk.Label(root,image=bg_img)
    bg_lbl.place(x=0,y=93)
    button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_kids, width=20, height=2)
    button2.place(x=15, y=120)
    button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_kids, width=20, height=2)
    button3.place(x=15, y=220)
    button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Kidnapping Crime Prediction", command=call_file_kids, width=25, height=2)
    button4.place(x=15, y=320)

def rape():
    bg = Image.open(r"BG.jpg")
    bg.resize((1500,200),Image.ANTIALIAS)
    print(w,h)
    bg_img = ImageTk.PhotoImage(bg)
    bg_lbl = tk.Label(root,image=bg_img)
    bg_lbl.place(x=0,y=93)
    button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_women, width=20, height=2)
    button2.place(x=15, y=120)
    button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                     text="Model Training", command=Model_Training_women, width=20, height=2)
    button3.place(x=15, y=220)
    button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Rape Crime Prediction", command=call_file_women, width=25, height=2)
    button4.place(x=15, y=320)



def murder():
    bg = Image.open(r"BG.jpg")
    bg.resize((1500,200),Image.ANTIALIAS)
    print(w,h)
    bg_img = ImageTk.PhotoImage(bg)
    bg_lbl = tk.Label(root,image=bg_img)
    bg_lbl.place(x=0,y=93)
    button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing_murder, width=20, height=2)
    button2.place(x=15, y=120)
    button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training_murder, width=20, height=2)
    button3.place(x=15, y=220)
    button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Murder Crime Prediction", command=call_file_murder, width=25, height=2)
    button4.place(x=15, y=320)
    
    
    
def theft1():
    from subprocess import call
    call(["python","theft1.py"])
    
  
def murder1():
    from subprocess import call
    call(["python","murder1.py"])
    
def kidnap():
    from subprocess import call
    call(["python","kidnap.py"])
    
def rape1():
    from subprocess import call
    call(["python","rape1.py"])    
#def graph():
    print('hello')
    
 
    button5 = tk.Button(root, text="Theft", command=theft1, width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
    button5.place(x=50, y=130)

    button6 = tk.Button(root, text="Murder",command=murder1,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
    button6.place(x=50, y=240)

    button7 = tk.Button(root, text="Kidnapping",command=kidnap,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
    button7.place(x=50, y=330)

    button8 = tk.Button(root, text="Rape",command=rape1,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
    button8.place(x=50, y=420)
    
    
# image2 =Image.open('n1.png')
bg = Image.open(r"BG.jpg")
bg.resize((1500,200),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)




wlcm=tk.Label(root,text="......Welcome to Crime Prediction System ......",width=85,bd=0,height=2,background="skyblue",foreground="black",font=("Times new roman",22,"bold"))
wlcm.place(x=0,y=620)

#Disease6=tk.Button(root,text="Graph ",command=graph,width=7,bd=0,height=2,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
#Disease6.place(x=800,y=18)

Disease1=tk.Button(root,text="Theft",command=theft,width=7,bd=0,height=2,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
Disease1.place(x=900,y=18)


Disease2=tk.Button(root,text="Murder ",command=murder,width=7,bd=0,height=2,background="skyblue",foreground="black",font=("times new roman",14,"bold"))
Disease2.place(x=1000,y=18)


Disease3=tk.Button(root,text="Kidnapping ",command=kidn,width=14,bd=0,height=2,background="skyblue",foreground="black",font=("Times new roman",14,"bold"))
Disease3.place(x=1100,y=18)

Disease3=tk.Button(root,text="Rape",command=rape,width=8,bd=0,height=2,background="skyblue",foreground="black",font=("Times new roman",14,"bold"))
Disease3.place(x=1260,y=18)

root.mainloop()
