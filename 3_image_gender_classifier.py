###############################
"""
Part 3 of 4
Script used for testing the model previus saved with some random images from internet

"""
###############################

import cv2
from tensorflow.keras.models import load_model

SCALE = (200, 200)

model = load_model('model.h5')

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image = cv2.imread('images/couple.jpg')

# returns the extremes of the faces found into the image
rects = face_cascade.detectMultiScale(image, 1.1, 15) 

# iter for each faces
for rect in rects:
    face = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] # face cut out from the image

    small_img = cv2.resize(face, SCALE) # resize images to common size
    x = small_img.astype(float) # cast to float...
    x/=255. # ... to be able to normalize the image

    # the predict method need a list of images so we cast as a list of a single image
    x = x.reshape(1, x.shape[0], x.shape[1], 3) 
    y = model.predict(x)

    y = y[0][0] # value of the prediction
    print(y)

    # show on image the prediction with sex and %
    label = "Man" if y>0.5 else "Woman"
    percentage = y if y>0.5 else 1.0-y
    percentage = round(percentage*100,1)

    cv2.rectangle(image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2) # green rectangle around the detected face

    cv2.rectangle(image, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED) # rectangle for label
    cv2.putText(image, label+" ("+str(percentage)+"%)", (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2) # label

cv2.imshow("Gender Classifier", image)
cv2.waitKey(0)
