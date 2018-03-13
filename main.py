import numpy as np
import os
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Reshape, GRU, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import keras
import time

label = int(time.time())
root = '.'
batch_size = 32
dropout = 0.0
l2 = 0.0
batch_norm = True
num_units = 500

train_list_raw = []
test_list_raw = []


if K.image_data_format() == 'channels_first':
    input_shape = (1, 858, 128)
else:
    input_shape = (858, 128, 1)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
if batch_norm:
    model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
if batch_norm:
    model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
if batch_norm:
    model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
if batch_norm:
    model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=32, kernel_size=3))

model.add(Reshape((32 * 8, 54)))

model.add(GRU(num_units, dropout=dropout))
if batch_norm:
    model.add(BatchNormalization())

model.add(Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l2(l2)))

print(model.summary())

with open('models/model_{}.json'.format(label), "w") as json_file:
    json_file.write(model.to_json())

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = ImageDataGenerator(rescale=1./256).flow_from_directory(
    'data/train',
    target_size=(858, 128),
    color_mode='grayscale'
)

model.fit_generator(
    train_generator,
    epochs=20
)

model.save_weights('models/model_{}.h5'.format(label))


# bengali_female_mono
# gujarati_female_mono
# hindi_female_mono
# kannada_female_mono
# marathi_female_mono
# tamil_female_mono
# telugu_female_mono