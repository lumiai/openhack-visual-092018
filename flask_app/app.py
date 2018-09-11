# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import model_from_json
from PIL import Image, ImageOps
import numpy as np
import flask
import io
import json
import os
import tensorflow as tf

#init flask app
app = flask.Flask(__name__)
global model, graph

gear_categories = ['axes', 'boots', 'carabiners', 'crampons', 'gloves', 'hardshell_jackets', 'harnesses', 'helmets', 'insulated_jackets','pulleys', 'rope', 'tents']

def load_model():
    with open('./model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model.h5")
    print("Loaded Model from disk")
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return loaded_model, graph

model, graph = load_model()

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_w, image_h, image_ch = img_to_array(image).shape
    delta_w, delta_h = max(image_w, image_h) - image_w, max(image_w, image_h) - image_h
    padding = (delta_h//2, delta_w//2, delta_h - (delta_h//2), delta_w - delta_w//2) 
    return img_to_array(ImageOps.equalize(ImageOps.expand(image, padding, fill="white").resize(target))) / 255.0



@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
        # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(128, 128))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(image.reshape(((1,) + image.shape)))
            # results = imagenet_utils.decode_predictions(preds)
            data["prediction"] = gear_categories[int(preds.argmax())]

            # # loop over the results and add them to the list of
            # # returned predictions
            # for (imagenetID, label, prob) in results[0]:
            #     r = {"label": label, "probability": float(prob)}
            #     data["predictions"].append(r)

            # indicate that the request was a success
            data["probability"] = [float(pred) for pred in preds.ravel()]
            data["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')