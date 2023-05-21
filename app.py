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
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
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

st.title("VGGFace Face Recognition")

def pearson_correlation(y_true,y_pred):
    return tfp.stats.correlation(y_true,y_pred)

def custom_object_scope(custom_objects):
    return keras_utils.CustomObjectScope(custom_objects)

# Usage example:
with custom_object_scope({'pearson_correlation': pearson_correlation}):
    # Your code here
    custom_model = load_model('vgg_model.h5')
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

def preprocess_image2(image):
    image = image.resize((224, 224))  # Resize the image to match the input size of VGGFace
    image = image.convert('RGB')  # Convert image to RGB format if necessary
    #image = preprocess_input(np.array(image))  # Preprocess the image
    image = np.array(image).astype('float64')  # Convert image to array of float64 data type
    image = preprocess_input(image) 
    return image

def recognize_faces(image):
    #image = preprocess_image(image)
    #embeddings = vggface_model.predict(np.expand_dims(image, axis=0))
    # Perform face recognition tasks using the embeddings
    # Implement your own logic here, such as comparing embeddings with a database of known faces
    #img = np.expand_dims(image, axis=0)
    #img= utils.preprocess_input(img, version=1) # or version=2
    embeddings2 = custom_model.predict(preprocessed_img2)
    #preds = vggface_model.predict(img)
	
	
with st.sidebar:
    choose = option_menu("App Gallery", ["Camera", "Upload a Photo","BMI Chart"],
                         icons=['camera fill','person lines fill',"camera video"],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
		
		
if choose == "Camera":
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
		
		st.metric(label="BMI", value=bmi)

elif choose == "Upload a Photo":
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if uploaded_file is not None:
		image2 = Image.open(uploaded_file)
		st.image(image2, caption='Uploaded Image', use_column_width=True)
		
		if st.button('Recognize Faces'):
			#recognize_faces(image)
			# Display the results or perform additional actions
			preprocessed_img2 = preprocess_image2(image2)
			preprocessed_img2 = np.expand_dims(preprocessed_img2, axis=0)
			preprocessed_img2 = utils.preprocess_input(preprocessed_img2, version=2)
			# Extract the embeddings using the VGGFace model
			recognize_faces(image2)
			embeddings2 = custom_model.predict(preprocessed_img2)
			# Extract the BMI value from the embeddings
			bmi = embeddings2[0][0]
			
			#st.write(f"BMI: {bmi:.2f}")
			st.metric(label="BMI", value=bmi)
			



    

