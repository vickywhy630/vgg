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
import av
from turn import get_ice_servers
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

#class VideoProcessor:
    #def recv(self, frame):
        #img = frame.to_ndarray(format="bgr24")
        #img = process(img)
        #return av.VideoFrame.from_ndarray(img, format="bgr24")
	
def pearson_correlation(y_true,y_pred):
	return tfp.stats.correlation(y_true,y_pred)

def custom_object_scope(custom_objects):
	return keras_utils.CustomObjectScope(custom_objects)

# Usage example:
with custom_object_scope({'pearson_correlation': pearson_correlation}):
	custom_model = load_model('vgg_model.h5')

        
def preprocess_image(image):
	image = cv2.resize(image, (224, 224))  # Resize the image to match the input size of VGGFace
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
	image = np.array(image).astype('float64')  # Convert image to array
	#image = preprocess_input(image)
	return image
        
        
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
def predict_bmi(frame):
	pred_bmi = []
	faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.15,minNeighbors = 5,minSize = (30,30))
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		image = frame[y:y+h, x:x+w]
		preprocessed_img= preprocess_image(image)
		preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
		preprocessed_img = utils.preprocess_input(preprocessed_img, version=2)
		embeddings = custom_model.predict(preprocessed_img)
		pred_bmi.append(embeddings[0][0])
		cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	return pred_bmi, frame
 
        
class VideoProcessor:
	def __init__(self):
		self.frame_lock = threading.Lock()
		self.out_image = None
		self.pred_bmi = []
		
	def recv(self, frame):
		frm = frame.to_ndarray(format='bgr24')
		pred_bmi, frame_with_bmi = predict_bmi(frm)
		with self.frame_lock:
			self.out_image = frame_with_bmi
			self.pred_bmi = pred_bmi
		return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 


#RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

#webrtc_ctx = webrtc_streamer(key="WYH",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoProcessor,\media_stream_constraints={"video": True, "audio": False},async_processing=True)

webrtc_streamer(key="example",video_processor_factory=VideoProcessor,rtc_configuration={"iceServers": get_ice_servers()},sendback_audio=False)
  
