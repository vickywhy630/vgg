import subprocess

# Install necessary dependencies
subprocess.run(["apt-get", "update"])
subprocess.run(["apt-get", "install", "-y", "libgl1-mesa-glx"])

filename = "/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py"
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image
from keras.models import load_model
import tensorflow_probability as tfp
import keras.utils.generic_utils as keras_utils
#from keras.utils import custom_object_scope
from keras_vggface import utils
import requests
import tempfile
import io

# Replace 'YOUR_LINK_HERE' with the actual Google Drive link
model_link = 'https://drive.google.com/file/d/1qT-NHwjmkKLN9G7Wu-BbO15q8rxwl9QI/view?usp=share_link'
response = requests.get(model_link)
model_content = response.content

temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(model_content)
temp_file.close()


def pearson_correlation(y_true,y_pred):
    return tfp.stats.correlation(y_true,y_pred)

def custom_object_scope(custom_objects):
    return keras_utils.CustomObjectScope(custom_objects)

# Usage example:
with custom_object_scope({'pearson_correlation': pearson_correlation}):
    # Your code here
    custom_model = load_model(temp_file.name)
# Register the custom metric function in the custom object scope
#with custom_object_scope({'pearson_correlation': pearson_correlation}):
    # Load the model
    #custom_model = load_model('vgg_model.h5')
#vmodel = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#last_layer = vmodel.get_layer('global_average_pooling2d').output
#x = Flatten(name='flatten')(last_layer)
#x = Dense(512, activation='relu', name='fc6')(x)
#output_layer = Dense(1, activation='linear')(x)
#custom_model = Model(vmodel.input, output_layer)
#custom_model = load_model('vgg_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize the image to match the input size of VGGFace
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
    image = np.array(image).astype('float64')  # Convert image to array
    #image = preprocess_input(image)
    return image

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # Preprocess the image
    preprocessed_img = preprocess_image(cv2_img)

    # Reshape the preprocessed image to match the input shape of VGGFace
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    
    preprocessed_img = utils.preprocess_input(preprocessed_img, version=2)

    # Extract the embeddings using the VGGFace model
    embeddings = custom_model.predict(preprocessed_img)

    # Extract the BMI value from the embeddings
    bmi = embeddings[0][0]

    # Perform BMI prediction based on the extracted BMI value
    # Implement your own logic here to predict BMI based on the extracted BMI value

    # Display the predicted BMI
    st.write(f"BMI: {bmi:.2f}")
