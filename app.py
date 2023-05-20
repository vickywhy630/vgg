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

def main():
	'''Set main() function. Includes sidebar navigation and respective routing.'''

	st.sidebar.title("Explore")
	app_mode = st.sidebar.selectbox( "Choose an Action", [
		"Camera",
		"Upload a Photo"
	])

	# clear tmp
	#clear_tmp()

	# nav
	if   app_mode == "Camera":              camera()
	elif app_mode == "Upload a Photo":  upload()
	
    def camera():
	
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

    def upload():
        df = load_data()
        non_label_cols = ['track_id', 'track_title', 'artist_name', 'track_popularity', 'artist_popularity']
        dims = [c for c in df.columns.tolist() if c not in non_label_cols]



    

