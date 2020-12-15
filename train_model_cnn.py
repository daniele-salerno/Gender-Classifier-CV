from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "faces_gender/"
# DATASET_DIR = "Training/"
BATCH_SIZE = 16

# istanziamo il generatore...
datagen = ImageDataGenerator(
        validation_split = 0.3, # train sul 70%
        rescale = 1./255, # norm immagini

        # crezione immagini usa e getta aggiuntive
        horizontal_flip = True,
        zoom_range = 0.2,
        brightness_range = [1,2]
)

# ... per leggere le immagini direttamente da disco
train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (150,150), # ridimensiamo le immagini a dimensione comune
        batch_size = BATCH_SIZE,
        class_mode = "binary", # uomo/donna
        subset = "training"
)

test_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (150,150),
        batch_size = BATCH_SIZE,
        class_mode = "binary",
        subset = "validation" # test
)

# stampiamo i labels (0/1) associati alle classi
print(train_generator.class_indices) 

model = Sequential()
 
model.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu", input_shape=(150,150, 3)))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_generator, epochs=40)

metrics_train = model.evaluate(train_generator)
metrics_test = model.evaluate(test_generator)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

model.save("model.h5")