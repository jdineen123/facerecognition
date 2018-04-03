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




References:

Crawford, Patrick W. “Application of Template Matching.” Lynda.com, 22 Sept. 2017, www.lynda.com/Python-tutorials/Application-template-matching/601786/660496-4.html. 

Geitgey, Adam. “Machine Learning Is Fun! Part 4: Modern Face Recognition with Deep Learning.” Medium, Medium, 24 July 2016, medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78. 

Goodfellow, Ian, et al. Deep Learning. MIT Press, 2016. 

Gulli, Antonio, and Sujit Pal. Deep Learning with Keras: Implement Neural Networks with Keras on Theano and TensorFlow. Packt, 2017. 

“KDnuggets.” KDnuggets Analytics Big Data Data Mining and Data Science, www.kdnuggets.com/2017/08/convolutional-neural-networks-image-recognition.html.

King, Davis. “Dlib-models”. Github.com, 17 Sept, 2017,     https://github.com/davisking/dlib-models

Sharif, Mahmood, et al. “Accessorize to a Crime.” Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security - CCS'16, 2016, doi:10.1145/2976749.2978392. 

Ageitgey. “Ageitgey/face_recognition.” GitHub, 9 Mar. 2018, github.com/ageitgey/face_recognition. 

“Tensorflow Tutorial 2: Image Classifier Using Convolutional Neural Network.” CV-Tricks.com, 26 Feb. 2017, cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/. 

King, Davis. “High Quality Face Recognition with Deep Metric Learning.” Dlib C++ Library, 1 Jan. 1970, blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html. 

“Facial Recognition Using Deep Learning – Towards Data Science.” Towards Data Science, Towards Data Science, 22 Mar. 2017, towardsdatascience.com/facial-recognition-using-deep-learning-a74e9059a150. 

“Online Video to GIF Converter.” Online Animated GIF Tools, ezgif.com/video-to-gif.

Andrej Karpathy Academic Website, cs.stanford.edu/people/karpathy/.

“CS231n Convolutional Neural Networks for Visual Recognition.” CS231n Convolutional Neural Networks for Visual Recognition, cs231n.github.io/. 

“Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures.” Exsilio Blog, 11 Nov. 2016, blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/. 

www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf

www.usenix.org/conference/usenixsecurity16/technical-sessions/.../xu.
