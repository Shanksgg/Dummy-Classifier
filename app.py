from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import keras.utils as image
import os
from werkzeug.utils import secure_filename
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import string

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'static/models/model_resnet.h5'

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Save pretrained model from Keras
# from keras.applications.resnet import ResNet50
#
# model = ResNet50(weights='imagenet')
# model.save(MODEL_PATH)


def model_predict(img_path, pmodel):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # Add new dimension
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # Default preprocess mode is caffee
    x = preprocess_input(x)

    preds = pmodel.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process result
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ')
        result = result.capitalize()
        return result
    return None


if __name__ == '__main__':
    app.run()
