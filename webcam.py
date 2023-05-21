import av
from turn import get_ice_servers
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

#class VideoProcessor:
    #def recv(self, frame):
        #img = frame.to_ndarray(format="bgr24")
        #img = process(img)
        #return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
def predict_bmi(frame):
    pred_bmi = []
    faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.15,minNeighbors = 5,minSize = (30,30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        image = frame[y:y+h, x:x+w]
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = np.array(img).astype(np.float64)
        features = get_fc6_feature(img)
        preds = svr_model.predict(features)
        pred_bmi.append(preds[0])
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
            #elf.out_image = frame_with_bmi
            self.pred_bmi = pred_bmi

        return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 


#RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

#webrtc_ctx = webrtc_streamer(key="WYH",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoProcessor,\media_stream_constraints={"video": True, "audio": False},async_processing=True)

webrtc_streamer(key="example",video_processor_factory=VideoProcessor,rtc_configuration={"iceServers": get_ice_servers()},sendback_audio=False)
  
