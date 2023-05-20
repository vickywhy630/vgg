

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
