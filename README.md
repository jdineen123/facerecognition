# facerecognition
Face Recognition with MLP for Binomial Classification.

-Uses Geitgey's python wrappers of Dlib for feature extraction from images. (https://github.com/ageitgey/face_recognition)

Defined UDFs for data munging, feature extraction. 

Loops to pull data from directories and munge.

Uses Keras for model creation. N hidden layers with relu activation and softmax act. on the output layer.

Uses OpenCV for webcam activation and frame by frame feed into MLP.

  OpenCV for drawing box around face in the image.
  
   OpenCV for textbox with Name and softmax outputs (prob. dists of a known class)
   
   
Model created and stored, called, within the same file, but could call from a separate .py file to avoid rebuilding the model everytime.


#Next Steps

Install Raspi + PiCamera.

Add conditional statement pointing to August API to lock door/unlock door if softmax output > threshold.

text alerts/email alerts if door is unlocked/locked. text alerts if face identified is not my own.


Potentially, recreate as a multinomial class problem. 

Currently only works to register probabilities of Jake v NotJake, which works for the purpose of unlocking a door.

Essay on Facerec here:

https://docs.google.com/document/d/1y1686M5OGtVe0uXv3HMK9fIEPLW32EeWks_7TmHs9PU/edit


I wanted to research and determine if there was distance present between frames in the case of a still imaged adversarial attacked - Ref: FaceRec_ResearchingLivenessDetection.py.







