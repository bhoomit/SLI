import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize, imshow
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def init():
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load woeights into new model
    loaded_model.load_weights("models/model.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.001),
        metrics=['accuracy']
    )

    generator = ImageDataGenerator(rescale=1. / 256, validation_split=0.2).flow_from_directory(
        'data/predict',
        target_size=(858, 128),
        color_mode='grayscale'
    )

    evaluate_resp = loaded_model.evaluate_generator(generator)
    # print(np.argmax(predict_resp, axis=1))
    print(evaluate_resp)
    print(loaded_model.metrics_names)
    graph = tf.get_default_graph()

    return loaded_model, graph
