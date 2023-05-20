

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_ctx = webrtc_streamer(
		key="WYH",
		mode=WebRtcMode.SENDRECV,
		rtc_configuration=RTC_CONFIGURATION,
		video_processor_factory=VideoProcessor,\
		media_stream_constraints={"video": True, "audio": False},
		async_processing=True)

