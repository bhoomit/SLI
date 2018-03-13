from flask import Flask, request, jsonify
import traceback
import requests
import time

import imageio
from scipy.misc import imresize, imshow
# for importing our keras model
import keras.models
# for regular expressions, saves time dealing with string data
from scripts import spectrograms
# system level operations (like loading files)
import sys
# for reading operating system data
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# tell our app where our saved model is
sys.path.append(os.path.abspath("./models"))
from load import *

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()

languages = {
    0: 'hindi',
    1: 'gujarati',
    2: 'kannada'
}


@app.route('/predict/', methods=['POST'])
def hello_world():
    try:
        body = request.get_json()
        url = body.get('url', '').replace('https', 'http') + '.mp3'
        filename = int(time.time() * 1000000)
        mp3file = '/tmp/{0}.mp3'.format(filename)
        # mp3file = '/Users/Bhoomit/work/hackathons/gi/SLI/data/train/2/zfnnofoa0ea.png'
        r = requests.get(url, allow_redirects=True)
        open(mp3file, 'wb').write(r.content)
        x = spectrograms.process_single_file(mp3file)
        # read the image into memory
        x = imageio.imread(load_img(x, grayscale=True, target_size=(858, 128)))
        # convert to a 4D tensor to feed into our model
        x = np.reshape(x, (1, 858, 128, 1))

        with graph.as_default():
            # perform the prediction
            start = time.time()
            out = model.predict(x)[0]
            print(out)
            print(np.argmax(out))
            # convert the response to a string
            response = int(np.argmax(out))
            return jsonify({
                'language': languages.get(response, 'english'),
                'time': time.time() - start
            })
    except:
        print(traceback.format_exc())
        return jsonify(result={"status": 500})


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
# optional if we want to run in debugging mode
# app.run(debug=True)
