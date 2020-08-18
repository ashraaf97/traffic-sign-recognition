# traffic-sign-recognition
A final year project that train GTSRB dataset using python using covolutional network model. This model use RGB colour to gain high accuracy despite of converting into grayscale. This is because specific traffic sign colour conveys specific message for the driver. For example, red is for prohibitory and blue is for mandatory. In this project also the unbalanced dataset is converted into balanced dataset to avoid the bias issue. The dataset is augmented two times to get the best results.

The best to view the project is run it in Google Colab. Library need to be installed first if local Jupyter Notebook is used.
To run the UI, model in format of .h5 is required. It cannot run in google collab and must run in your own computer.

Requirements:
```sh
python software
keras
Sci-kit learn
matplotlib
Pillow
```

### All 43 Classes from the GTSRB dataset:
![image 1](https://i.imgur.com/1xurEDT.jpg)
### The composition for each classes in the training dataset:
![image 2](https://i.imgur.com/sd49tBl.png)
### The composition for each classes in the training dataset after balancing using augmentation:
![image 3](https://i.imgur.com/Dk6C5ss.png)
### The accuracy and valdation accuracy for every epoch:
![image 4](https://i.imgur.com/66g53Qi.png)
### The accuracy and valdation loss for every epoch:
![image 5](https://i.imgur.com/TEKrl71.png)
### Simple Program with user interface to classify the image:
![image 6](https://i.imgur.com/0zf2Bj0.png)
### Example of results when using the program:
![image 7](https://i.imgur.com/QE3fsIA.jpg)
