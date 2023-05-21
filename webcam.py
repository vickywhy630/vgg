import av
from turn import get_ice_servers
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

#class VideoProcessor:
    #def __init__(self):
        #self.frame_lock = threading.Lock()
        #self.out_image = None
        #self.pred_bmi = []

    #def recv(self, frame):
        #frm = frame.to_ndarray(format='bgr24')
        #pred_bmi, frame_with_bmi = predict_bmi(frm)
        #with self.frame_lock:
            #self.out_image = frame_with_bmi
            #self.pred_bmi = pred_bmi

        #return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 


#RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

#webrtc_ctx = webrtc_streamer(key="WYH",mode=WebRtcMode.SENDRECV,rtc_configuration=RTC_CONFIGURATION,video_processor_factory=VideoProcessor,\media_stream_constraints={"video": True, "audio": False},async_processing=True)

webrtc_streamer(key="example",video_processor_factory=VideoProcessor,rtc_configuration={"iceServers": get_ice_servers()},sendback_audio=False)
  
