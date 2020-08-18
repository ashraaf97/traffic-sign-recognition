import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#load the trained model to classify sign
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import os
import imageio
import imgaug as ia
from imgaug import augmenters as iaa

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model('D:\\Downloads\\Model2fyp_model.h5')
# labels for bars 
tick_label = ['Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
            'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 
            'Speed limit (100km/h)', 
            'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons']
 
#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

subclasses = { 1:'Prohibitory',
            2:'Prohibitory', 
            3:'Prohibitory', 
            4:'Prohibitory', 
            5:'Prohibitory', 
            6:'Prohibitory', 
            7:'Prohibitory', 
            8:'Prohibitory', 
            9:'Prohibitory', 
            10:'Prohibitory', 
            11:'Prohibitory', 
            12:'Priority,Warning', 
            13:'Priority', 
            14:'Priority,Warning', 
            15:'Priority,Warning', 
            16:'Prohibitory', 
            17:'Prohibitory', 
            18:'Prohibitory', 
            19:'Warning', 
            20:'Warning', 
            21:'Warning', 
            22:'Warning', 
            23:'Warning', 
            24:'Warning', 
            25:'Warning', 
            26:'Warning', 
            27:'Warning', 
            28:'Warning', 
            29:'Warning', 
            30:'Warning', 
            31:'Warning',
            32:'Warning', 
            33:'Prohibitory', 
            34:'Mandatory', 
            35:'Mandatory', 
            36:'Mandatory', 
            37:'Mandatory', 
            38:'Mandatory', 
            39:'Mandatory', 
            40:'Mandatory', 
            41:'Mandatory, Priority', 
            42:'Prohibitory', 
            43:'Prohibitory' }

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classifier 1.0')
top.configure(background='#ffffff')
label=Label(top,background='#ffffff', font=('arial',15,'bold'))
label2=Label(top,background='#ffffff', font=('arial',15,'bold'))
label3=Label(top,background='#ffffff', font=('arial',12,'bold'),text="Augmentor Adjuster")
sign_image = Label(top)

def augmentate():
    global file_path, w1,w2,w3
    image = imageio.imread(file_path)
    
    crop = iaa.Crop(percent=w2.get())
    rotate = iaa.Affine(rotate=w1.get())
    gaussiannoise = iaa.AdditiveGaussianNoise(scale=w3.get())

    image = crop(image=image)
    image = rotate(image=image)
    image = gaussiannoise(image=image)
    imageio.imwrite((os.path.join(directory_of_python_script, "temp.jpg")),image)
    file_path = os.path.join(directory_of_python_script, "temp.jpg")
    uploaded=Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
        
def classify():
    global label_packed
    print(file_path)
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = tf.cast(image, tf.float32)
    pred = model.predict_classes([image])[0]
    probab = model.predict(([image])[0]).ravel()
    typesign = subclasses[pred+1]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#000000', text='Prediction: '+sign+', Signs Type: '+typesign)
    #label2.configure(foreground='#011638', text='Signs Type: '+typesign)

def probab_calc():
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = tf.cast(image, tf.float32)
    pred = model.predict_classes([image])[0]
    probab = model.predict(([image])[0]).ravel()
    # plotting the points
    plt.plot(probab.tolist(),tick_label, color='green', linestyle='dashed', linewidth = 3, marker='.', markerfacecolor='blue', markersize=2) 
    
    plt.yticks(fontsize=5)
    # naming the x axis 
    plt.xlabel('Classes')
    # naming the y axis 
    plt.ylabel('Probabilites')
  
    # giving a title to my graph 
    plt.title('Probabilites vs Classes') 
      
    # function to show the plot
    plt.show()

def show_augmentate_button():
   
    augmentate_b=Button(top,text="Augment",command=lambda: augmentate(),padx=10,pady=5)
    augmentate_b.configure(background='#006c62', foreground='white',font=('arial',10,'bold'))
    augmentate_b.place(relx=0.26,rely=0.76)
    
def show_classify_button():
    classify_b=Button(top,text="Classify Image",command=lambda: classify(),padx=10,pady=5)
    classify_b.configure(background='#006c62', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.42,rely=0.76)

def show_probab_button():
    probab_b=Button(top,text="Softmax Probabilities",command=lambda: probab_calc(),padx=10,pady=5)
    probab_b.configure(background='#006c62', foreground='white',font=('arial',10,'bold'))
    probab_b.place(relx=0.60,rely=0.76)
    
def upload_image():
    try:
        global file_path, w1,w2,w3
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button()
        show_probab_button()
        show_augmentate_button()
        w1 = Scale(top, from_=-180, to=180,orient=HORIZONTAL)
        w1.place(relx=0.10,rely=0.37)
        w2 = Scale(top, from_=0, to=0.5,resolution=0.1,orient=HORIZONTAL)
        w2.place(relx=0.10,rely=0.48)
        w3 = Scale(top, from_=0, to=60,orient=HORIZONTAL)
        w3.place(relx=0.10,rely=0.26)
        l1 = Label(top, text = "AdditiveGaussianNoise",font =("arial", 10),background='#ffffff',foreground='#000000')
        l1.place(relx=0.10,rely=0.23)
        l2 = Label(top, text = "Crop",font =("arial", 10),background='#ffffff',foreground='#000000')
        l2.place(relx=0.10,rely=0.44)
        l3 = Label(top, text = "Rotation",font =("arial", 10),background='#ffffff',foreground='#000000')
        l3.place(relx=0.10,rely=0.33)
        label2.place(relx=0.70,rely=0.20)
        label2.configure(text="The prohibition signs forbid a \ncertain action.\n\nThe warning signs warn road users of a \npotentially dangerous traffic situation.\n\nThe priority signs have an influence on\n the priority rules.\n\nThe mandatory signs impose an \nobligation that road users must comply",justify=LEFT,font =("arial", 10))
        label3.place(relx=0.08,rely=0.18)

    except:
        pass
    
    
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#006c62', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
label.pack(side=BOTTOM,expand=True)
sign_image.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Traffic Sign Classifier 1.0",pady=20, font=('arial',20,'bold'))
heading.configure(background='#ffffff',foreground='#000000')
heading.pack()
top.mainloop()
input('Press ENTER to exit')

