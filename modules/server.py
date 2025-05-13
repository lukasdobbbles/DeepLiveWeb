# server.py
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from modules.processors.frame.face_swapper import FaceSwapper
from av import VideoFrame


class SwapTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source_track, swapper: FaceSwapper):
        super().__init__()  # initialize base
        self.track = source_track
        self.swapper = swapper

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        out = self.swapper.process_frame(img)  # core face-swap API
        new_frame = VideoFrame.from_ndarray(out, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def run_server():
    signaling = TcpSocketSignaling("0.0.0.0", 9999)
    await signaling.connect()
    pc = RTCPeerConnection()

    # on offer from client…
    offer = await signaling.receive()
    await pc.setRemoteDescription(offer)

    # attach transform to each incoming video track
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # initialize with the user’s chosen source image
            swapper = FaceSwapper(source_image_path="user_face.jpg")
            pc.addTrack(SwapTrack(track, swapper))

    # send back answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await signaling.send(pc.localDescription)
