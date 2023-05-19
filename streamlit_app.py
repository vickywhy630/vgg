import streamlit as st
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras_vggface.utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("VGGFace Face Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),pooling='avg')

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to match the input size of VGGFace
    image = image.convert('RGB')  # Convert image to RGB format if necessary
    #image = preprocess_input(np.array(image))  # Preprocess the image
    image = np.array(image).astype('float64')  # Convert image to array of float64 data type
    image = preprocess_input(image) 
    return image

# Function to recognize faces in the image
def recognize_faces(image):
    #image = preprocess_image(image)
    #embeddings = vggface_model.predict(np.expand_dims(image, axis=0))
    # Perform face recognition tasks using the embeddings
    # Implement your own logic here, such as comparing embeddings with a database of known faces
    #img = np.expand_dims(image, axis=0)
    #img= utils.preprocess_input(img, version=1) # or version=2
    embeddings = vggface_model.predict(np.expand_dims(image, axis=0))
    #preds = vggface_model.predict(img)
    #print(preds)
    
def predict_bmi(embeddings):
    bmi = embeddings[0][0]
    #if bmi < 18.5:
        #category = 'Underweight'
    #elif bmi < 24.9:
        #category = 'Normal weight'
    #elif bmi < 29.9:
        #category = 'Overweight'
    #elif bmi < 34.9:
        #category = 'Moderately Obese'
    #else:
        #category = 'Severely Obese'
    
    st.write(f"BMI: {bmi:.2f}")
    #st.write(f"Category: {category}")


if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Recognize faces
    if st.button('Recognize Faces'):
        #recognize_faces(image)
        # Display the results or perform additional actions
        image = preprocess_image(image)
        recognize_faces(image)
        embeddings = vggface_model.predict(np.expand_dims(image, axis=0))
        predict_bmi(embeddings)
