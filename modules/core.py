#!/usr/bin/env python3

import os
import sys
import logging

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith("--execution-provider") for arg in sys.argv):
    os.environ["OMP_NUM_THREADS"] = "1"
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
from typing import List, Optional
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import cv2
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCIceCandidate,
)
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
import json
import websockets
import base64
import uuid
import time
import aiortc  # For logging version
from av import VideoFrame

# Assuming modules.globals, modules.metadata etc. are in the same directory or Python path
import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.processors.frame import face_swapper as face_swapper_processor
from modules.face_analyser import get_one_face
from modules.typing_util import Face
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
    detect_fps,
    create_video,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
    create_temp,
    move_temp,
    clean_temp,
    normalize_output_path,
)
import av

if (
    hasattr(modules.globals, "execution_providers")
    and "ROCMExecutionProvider" in modules.globals.execution_providers
):
    del torch

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

logging.basicConfig(level=logging.INFO)
logging.getLogger("aiortc").setLevel(logging.INFO)
logging.getLogger("aioice").setLevel(logging.INFO)
logging.info(f"aiortc version: {aiortc.__version__}")  # Log aiortc version

SOURCE_FACE: Optional[Face] = None
TEMP_UPLOAD_DIR = "temp_uploads"
pc_config = RTCConfiguration(
    iceServers=[
        RTCIceServer(
            urls=[
                "stun:stun.l.google.com:19302",
            "stun:stun.l.google.com:5349"],
        )
    ]
)


def ensure_temp_upload_dir_exists():
    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        logging.info(f"Created temporary upload directory: {TEMP_UPLOAD_DIR}")


class SwapTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.face_swapper_model = face_swapper_processor.get_face_swapper()
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        self.last_pts = None

    async def recv(self):
        global SOURCE_FACE

        # 1) pull the next decoded frame
        try:
            frame = await self.track.recv()
        except Exception as e:
            print(f"Error receiving frame in SwapTrack: {e}")
            raise

        queue = getattr(self.track, "_queue", None)
        if queue is not None:
            try:
                while queue.qsize() > 0:
                    # get_nowait is synchronous; do NOT await it
                    frame = queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        # end-of-stream
        if frame is None:
            return None

        # keep frame timing sane
        if self.last_pts is not None and frame.pts <= self.last_pts:
            print(
                f"SwapTrack: PTS went backward (prev={self.last_pts}, curr={frame.pts})"
            )
        self.last_pts = frame.pts

        # 2) get a BGR numpy array for face detection + swapping
        bgr = frame.to_ndarray(format="bgr24")
        face = get_one_face(bgr)

        # 3) convert to RGB for the swapper
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 4) do the swap (in RGB space)
        if SOURCE_FACE and face and self.face_swapper_model:
            try:
                swapped_rgb = face_swapper_processor.swap_face(SOURCE_FACE, face, rgb)
            except Exception as swap_err:
                print(f"Swap error: {swap_err}")
                swapped_rgb = rgb
        else:
            swapped_rgb = rgb

        # 5) convert _that_ back to BGR
        result_bgr = cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR)

        # 6) wrap into a proper AV frame for WebRTC
        new_frame = VideoFrame.from_ndarray(result_bgr, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame


connected_clients = set()


async def reload_source_face(new_source_path: str, is_temp_file: bool = False):
    global SOURCE_FACE
    logging.info(f"Loading source face: {new_source_path}")
    if not os.path.exists(new_source_path):
        logging.error(f"Src path DNE: {new_source_path}")
        return False

    img = cv2.imread(new_source_path)
    if img is None:
        logging.error(f"Could not read img: {new_source_path}")
        return False

    face = get_one_face(img)
    if face is None:
        logging.error(f"No face in: {new_source_path}")
        return False

    SOURCE_FACE = face
    if not is_temp_file:
        modules.globals.source_path = new_source_path

    logging.info(f"Source face reloaded: {new_source_path}")
    return True


def parse_ice_candidate(
    cand_str: str, sdp_mid: Optional[str] = None, sdp_mline_index: Optional[int] = None
) -> RTCIceCandidate:
    """
    Convert an SDP ICE candidate string into an aiortc RTCIceCandidate.

    Parameters
    ----------
    cand_str : str
        The raw candidate line, e.g.
        "candidate:1 1 UDP 2122194687 192.168.1.2 61996 typ host"
    sdp_mid : Optional[str]
        The media stream identification tag from the SDP ("a=mid"), if known.
    sdp_mline_index : Optional[int]
        The media line index (audio=0, video=1, etc.), if known.

    Returns
    -------
    RTCIceCandidate
        An instance ready to pass to RTCPeerConnection.addIceCandidate().
    """
    # Split into fields per ICE
    print(cand_str)
    parts = cand_str.split()
    # foundation is after 'candidate:'
    foundation = parts[0].split(":", 1)[1]
    component = int(parts[1])
    protocol = parts[2]
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    # parts[6] should be "typ"
    cand_type = parts[7]
    # Instantiate aiortc candidate
    return RTCIceCandidate(
        component=component,
        foundation=foundation,
        ip=ip,
        port=port,
        priority=priority,
        protocol=protocol,
        type=cand_type,
        relatedAddress=None,
        relatedPort=None,
        sdpMid=sdp_mid,
        sdpMLineIndex=sdp_mline_index,
        tcpType=None,
    )


async def handle_websocket_signaling(ws):
    global SOURCE_FACE
    pc = RTCPeerConnection(configuration=pc_config)
    logging.info(f"[SIGNAL] New conn: {ws.remote_address}")
    connected_clients.add(ws)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            payload = {
                "type": "candidate",
                "candidate": candidate.to_sdp(),
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
            }
        else:
            payload = {"type": "candidate", "candidate": None}

        try:
            await ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            logging.warning(f"[SIGNAL] ICE send failed, {ws.remote_address} is closed")

    @pc.on("track")
    def on_track(track):
        logging.info(f"[PC] Track {track.kind} rcvd for {ws.remote_address}")
        if track.kind == "video":
            pc.addTrack(SwapTrack(track))

    @pc.on("connectionstatechange")
    async def on_state():
        logging.info(f"[PC] state={pc.connectionState} for {ws.remote_address}")
        if (
            pc.connectionState in ("failed", "closed", "disconnected")
            and pc.signalingState != "closed"
        ):
            logging.warning(
                f"[PC] Conn state {pc.connectionState}, closing PC for {ws.remote_address}."
            )
            await pc.close()

    try:
        async for msg in ws:
            data = json.loads(msg)
            t = data.get("type")

            if t == "offer":
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data["sdp"], type=t)
                )
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send(
                    json.dumps(
                        {
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                        }
                    )
                )

            elif t == "candidate":
                print(data)
                cand = data.get("candidate")
                if cand is None:
                    logging.info(f"[PC] {ws.remote_address} sent end-of-candidates")
                else:
                    sdp_mid = data.get("candidate").get("sdpMid")
                    sdp_index = data.get("candidate").get("sdpMLineIndex")
                    # candidate data:
                    cand = cand.get("candidate")
                    # build a real RTCIceCandidate
                    ice_candidate = parse_ice_candidate(cand, sdp_mid, sdp_index)

                    try:
                        await pc.addIceCandidate(ice_candidate)
                        logging.info(
                            f"[PC] Added ICE candidate from {ws.remote_address}: {cand[:60]}…"
                        )
                    except Exception as e:
                        logging.error(
                            f"[PC] Failed to add ICE candidate: {e}", exc_info=True
                        )
            elif t == "update_settings":

                s = data.get("settings", {})
                path = s.get("sourcePath")
                dataurl = s.get("sourceFileData")
                if dataurl and dataurl.startswith("data:image"):
                    header, enc = dataurl.split(",", 1)
                    imgdata = base64.b64decode(enc)
                    ensure_temp_upload_dir_exists()
                    tmp = os.path.join(
                        TEMP_UPLOAD_DIR,
                        f"upload_{uuid.uuid4().hex}_{int(time.time())}.png",
                    )
                    with open(tmp, "wb") as f:
                        f.write(imgdata)
                    await reload_source_face(tmp, is_temp_file=True)
                elif path:
                    await reload_source_face(path, is_temp_file=False)

                # update globals
                modules.globals.mouth_mask = s.get(
                    "mouthMask", getattr(modules.globals, "mouth_mask", False)
                )
                modules.globals.color_correction = s.get(
                    "colorCorrection",
                    getattr(modules.globals, "color_correction", False),
                )
                modules.globals.show_mouth_mask_box = s.get(
                    "showMouthMaskBox",
                    getattr(modules.globals, "show_mouth_mask_box", False),
                )
                logging.info(
                    f"Settings updated for {ws.remote_address}: "
                    f"MM={modules.globals.mouth_mask}, "
                    f"CC={modules.globals.color_correction}, "
                    f"SMB={modules.globals.show_mouth_mask_box}"
                )

            else:
                logging.warning(
                    f"[SIGNAL] Unknown msg type {t} from {ws.remote_address}"
                )

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"[SIGNAL] Conn closed for {ws.remote_address}.")
    except json.JSONDecodeError as e:
        logging.error(
            f"[SIGNAL] JSON decode err {ws.remote_address}: {e}. Msg: {msg[:200] if isinstance(msg, str) else type(msg)}"
        )
    except Exception as e:
        logging.error(
            f"[SIGNAL] Unhandled err for {ws.remote_address}: {e}", exc_info=True
        )

    finally:
        logging.info(f"[SIGNAL] Cleaning up {ws.remote_address}")
        connected_clients.discard(ws)
        if pc and pc.signalingState != "closed":
            await pc.close()
        logging.info(
            f"[SIGNAL] Closed {ws.remote_address}. PC state: {getattr(pc, 'signalingState', 'N/A')}"
        )


def run_server_main_task():
    if modules.globals.source_path and not SOURCE_FACE:
        return reload_source_face(modules.globals.source_path)

    host, port = "0.0.0.0", 9999
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    # handle SIGINT/SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: stop_event.set())
        except NotImplementedError:
            signal.signal(sig, lambda s, f: stop_event.set())

    return websockets.serve(handle_websocket_signaling, host, port), stop_event


def parse_args():
    ap = argparse.ArgumentParser(
        description=(getattr(modules.metadata, "name", "App") + " WebRTC Server")
    )
    ap.add_argument("-s", "--source", dest="source_path")
    ap.add_argument(
        "--mouth-mask", action="store_true", dest="mouth_mask", default=False
    )
    ap.add_argument(
        "--max-memory",
        type=int,
        default=getattr(modules.globals, "max_memory", None),
        dest="max_memory",
    )
    ap.add_argument(
        "--execution-provider",
        nargs="+",
        default=getattr(
            modules.globals, "execution_providers", ["CPUExecutionProvider"]
        ),
        dest="execution_provider",
    )
    args = ap.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(
        args.execution_provider
    )


def decode_execution_providers(reqs: List[str]) -> List[str]:
    try:
        avail = onnxruntime.get_available_providers()
    except Exception as e:
        logging.error(f"ONNX prov err: {e}. Default CPU.")
        return ["CPUExecutionProvider"]

    enc = [p.replace("ExecutionProvider", "").lower() for p in avail]
    out = []
    for r in reqs:
        for p_avail, e_avail in zip(avail, enc):
            if r.lower() == e_avail:
                out.append(p_avail)
                break

    if not out:
        logging.warning("No valid exec prov found, default CPU.")
        return ["CPUExecutionProvider"]

    logging.info(f"Using exec providers: {out}")
    return out


def limit_resources():
    if "tensorflow" in sys.modules:
        try:
            for dev in tensorflow.config.experimental.list_physical_devices("GPU"):
                tensorflow.config.experimental.set_memory_growth(dev, True)
            if tensorflow.config.experimental.list_physical_devices("GPU"):
                logging.info("TF GPU mem growth enabled.")
        except Exception as e:
            logging.warning(f"Could not config TF GPU mem growth: {e}")


async def main():
    # Start server and wait for shutdown signal
    serve, stop_event = run_server_main_task()
    server = await serve
    logging.info(
        f"WebSocket server started ws://{server.sockets[0].getsockname()[0]}:{server.sockets[0].getsockname()[1]}"
    )
    await stop_event.wait()
    logging.info("Cleaning temp files...")
    clean_temp()
    logging.info("App finished.")


def run():
    # defaults
    defaults = {
        "source_path": None,
        "mouth_mask": False,
        "color_correction": False,
        "show_mouth_mask_box": False,
        "max_memory": None,
        "execution_providers": ["CPUExecutionProvider"],
    }
    for k, v in defaults.items():
        if not hasattr(modules.globals, k):
            setattr(modules.globals, k, v)

    parse_args()
    limit_resources()
    logging.info("App starting...")
    asyncio.run(main())


if __name__ == "__main__":
    run()
