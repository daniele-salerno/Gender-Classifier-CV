import cv2
from tensorflow.keras.models import load_model

import random

SCALE = (200,200)

model = load_model('model_augmented.h5')

# catturiamo solo i volti delle immagini catturate
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

n = random.randint(1, 5000)
frame = cv2.imread('faces_gender/female/'+str(n)+'.jpg')
frame = cv2.imread('images/four_women.JPG')

# ritorna gli estremi dei volti trovati nel frame
rects = face_cascade.detectMultiScale(frame, 1.1, 15) 

# iteriamo su tutti i volti trovati
for rect in rects:
    img = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] # immagine ritaglita del solo volto

    small_img = cv2.resize(img, SCALE) # scaliamo l'immagine in dimensione univoca
    x = small_img.astype(float) # castiamo i valori a float...
    x/=255. # ... per poter normalizzare l'immagine

    # num immagini, larghezza, altezza, canali, dal momento che predict si aspetta una lista di valori
    x = x.reshape(1, x.shape[0], x.shape[1], 3) 
    y = model.predict(x)

    y = y[0][0] # singolo valore della previsione
    print(y)

    # predizione
    label = "Uomo" if y>0.5 else "Donna"
    percentage = y if y>0.5 else 1.0-y
    percentage = round(percentage*100,1)

    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2) # rettangolo verde intorno al volto
    
    cv2.rectangle(frame, (rect[0], rect[1]-20), (rect[0]+170, rect[1]), (0,255,0), cv2.FILLED) # rettangolo per label uomo/donna
    cv2.putText(frame, label+" ("+str(percentage)+"%)", (rect[0]+5, rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.4, (255,255,255), 2)

cv2.imshow("Gender Classifier", frame)
cv2.waitKey(0)
