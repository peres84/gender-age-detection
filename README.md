# Gender and Age Detection

![Alt](/img/header.png)




# :warning: About



The goal of this project is detect the gender and age of any person, using a webcam as the device to interact with the enviroment, the project consist in a gui interface that will allow to use your camera and predict the person in front of it. 

In future versions, will allow to save the data of any record, live webcam, or video, in a mongo database, that will help to find new data of people around. 


### Database Description 

The dataset is composed of 5 folds to allow 5-fold 'leave one out' cross validation. To prevent overfitting, each fold contains different subjects. 

Each fold is described by a csv file with 12 columns:

    - user_id - the folder in the dataset containing the image. 
    - original_image - image name in the dataset.
    - face_id - the Face ID in the original Flickr image, can be ignored. 
    - age - age label of the face.
    - gender - gender label of the face.
    - x, y, dx, dy - bounding box of the face in the original Flickr image, can be ignored.
    - tilt_ang, fiducial_yaw_angle - pose of the face in the original Flickr image, can be ignored. 
    - fiducial_score - score of the landmark detector, can be ignored. 

If you use the dataset, please cite: Eran Eidinger, Roee Enbar, Tal Hassner. Age and Gender Estimation of Unfiltered Faces. Transactions on Information Forensics and Security (IEEE-TIFS), special issue on Facial Biometrics in the Wild, Volume 9, Issue 12, pages 2170 - 2179, 2014.

## :computer: Installation

Before start We need to install the dependencies requeriments.txt and environment.yml,

Some basic Git commands are:


```

#create new directory
mkdir <folder name>

#clone the repository 
$ git init
$ git clone https://github.com/peres84/gender-age-detection.git

#setup your environment as 'cv' 
$ conda create -f environment.yml

#adding dependencies 
$ pip install -r requirements.txt 

```

## :books: Resources 

- [references](https://www.kaggle.com/manarbinowayid/age-gender-classification-project)
- [database](https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification)


## :envelope: Contact

peresrjavier@gmail.com
