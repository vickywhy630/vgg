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
	if   app_mode == "Camera":              show_about()
	elif app_mode == "Upload a Photo":  explore_classified()
	
    def show_about():
        st.title('Learning to Listen, to Feel')
        for line in read_text(path('about.txt')):
            st.write(line)
            
    def explore_classified():
        df = load_data()
        non_label_cols = ['track_id', 'track_title', 'artist_name', 'track_popularity', 'artist_popularity']
        dims = [c for c in df.columns.tolist() if c not in non_label_cols]



    

