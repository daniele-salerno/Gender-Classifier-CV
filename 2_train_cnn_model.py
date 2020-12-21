from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import os

# dataset from -> https://s3.eu-west-3.amazonaws.com/profession.ai/datasets/faces_gender.zip

SCALE = (200, 200)

# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# prepare dataset with faces only images

os.mkdir("new_faces_gender")
os.mkdir("new_faces_gender/female")
os.mkdir("new_faces_gender/male")

for dir in ["female","male"]:
  for f in os.listdir("faces_gender/"+dir+"/"):
    img = cv2.imread("faces_gender/"+dir+"/"+f)
    if(img is None):
      continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, 1.1, 15)
    for rect in rects:
      face = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
      small_img = cv2.resize(face, SCALE)
      cv2.imwrite("new_faces_gender/"+dir+"/"+f, small_img)

DATASET_DIR = "new_faces_gender/"
BATCH_SIZE = 16

# istanziamo il generatore...
datagen = ImageDataGenerator(
        validation_split = 0.1, # train on 90%
        rescale = 1./255, # norm images

        # crezione immagini usa e getta aggiuntive
        horizontal_flip = True,
        zoom_range = 0.2,
        brightness_range = [1,2]
)

# ... per leggere le immagini direttamente da disco
train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = SCALE, # ridimensiamo le immagini a dimensione comune
        batch_size = BATCH_SIZE,
        class_mode = "binary", # uomo/donna
        subset = "training"
)

test_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = SCALE,
        batch_size = BATCH_SIZE,
        class_mode = "binary",
        subset = "validation" # test
)

# stampiamo i labels (0/1) associati alle classi
print(train_generator.class_indices) 

model = Sequential()
 
model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu", input_shape=(200,200, 3)))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_generator, epochs=100)

metrics_train = model.evaluate(train_generator)
metrics_test = model.evaluate(test_generator)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

model.save("model.h5")