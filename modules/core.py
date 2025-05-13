import os
import sys
import logging
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
import json
import websockets
import base64
import uuid
import time
from websockets.connection import State

import modules.globals
import modules.metadata
from modules.processors.frame.core import get_frame_processors_modules
from modules.processors.frame import face_swapper as face_swapper_processor
from modules.face_analyser import get_one_face
from modules.typing_util import Face
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
# from modules.processors.frame import face_enhancer as face_enhancer_processor # COMMENT OUT or REMOVE

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Configure aiortc logging (add this near the top, after imports)
logging.basicConfig(level=logging.INFO) # Or logging.DEBUG for very verbose output
logging.getLogger("aiortc").setLevel(logging.INFO) # Or logging.DEBUG
logging.getLogger("aioice").setLevel(logging.INFO) # For ICE related messages

# --- WebRTC Server Code ---

# Global variable to hold the loaded source face
SOURCE_FACE: Optional[Face] = None

TEMP_UPLOAD_DIR = "temp_uploads"

# Define ICE server configuration (using Google's public STUN server)
pc_config = RTCConfiguration(iceServers=[
    RTCIceServer(urls="stun:stun.l.google.com:19302")
])

def ensure_temp_upload_dir_exists():
    """Ensures the temporary upload directory exists."""
    if not os.path.exists(TEMP_UPLOAD_DIR):
        try:
            os.makedirs(TEMP_UPLOAD_DIR)
            print(f"Created temporary upload directory: {TEMP_UPLOAD_DIR}")
        except OSError as e:
            print(f"Error creating temporary upload directory {TEMP_UPLOAD_DIR}: {e}")
            # Depending on desired behavior, might want to raise an error or exit


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
        
        # Introduce a very small sleep at the beginning of each recv call.
        # This ensures that even if self.track.recv() returns immediately
        # and processing is very fast, we yield to the event loop.
        # This can sometimes help with pacing issues in aiortc with custom tracks.
        # Value is experimental: 0 means yield "soon", 0.001 is 1ms.
        await asyncio.sleep(0.001) # EXPERIMENTAL: Try 0 or 0.001 or 0.005

        try:
            frame = await self.track.recv()
        except Exception as e:
            print(f"Error receiving frame in SwapTrack: {e}")
            raise e

        if frame is None:
            print("SwapTrack: Original track sent None frame (EOS).")
            return None

        if self.last_pts is not None and frame.pts <= self.last_pts:
            print(f"SwapTrack: Warning - PTS did not advance or went backward. Prev: {self.last_pts}, Curr: {frame.pts}")
        self.last_pts = frame.pts

        process_start_time = time.monotonic()

        img = frame.to_ndarray(format="bgr24")
        to_ndarray_time = time.monotonic()

        target_face = get_one_face(img)
        get_target_face_time = time.monotonic()
        
        out_img = img
        
        color_correction_applied = False
        color_correction_start_time = None
        if modules.globals.color_correction:
            color_correction_start_time = time.monotonic()
            try:
                out_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                color_correction_applied = True
            except Exception as cc_err:
                print(f"Color correction error: {cc_err}")
                out_img = img
        color_correction_end_time = time.monotonic()

        current_source_face = SOURCE_FACE
        swap_applied = False
        face_swap_start_time = None
        if current_source_face and target_face and self.face_swapper_model:
            face_swap_start_time = time.monotonic()
            try:
                out_img = face_swapper_processor.swap_face(
                    current_source_face, target_face, out_img
                )
                swap_applied = True
            except Exception as swap_err:
                 print(f"Error during face swap: {swap_err}")
                 pass 
        face_swap_end_time = time.monotonic()
        
        enhancer_applied = False
        face_enhancer_start_time = None
        if modules.globals.fp_ui.get('face_enhancer', False):
            face_enhancer_start_time = time.monotonic()
            pass
        face_enhancer_end_time = time.monotonic()

        final_conversion_applied = False
        final_conversion_start_time = None
        if color_correction_applied:
            final_conversion_start_time = time.monotonic()
            try:
                final_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                final_conversion_applied = True
            except Exception as final_cc_err:
                 print(f"Final color conversion error: {final_cc_err}")
                 final_img = out_img
        else:
             final_img = out_img
        final_conversion_end_time = time.monotonic()

        create_frame_start_time = time.monotonic()
        new_frame = VideoFrame.from_ndarray(final_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        create_frame_end_time = time.monotonic()

        process_end_time = time.monotonic()

        self.frame_count += 1
        current_time = time.monotonic()
        if current_time - self.last_log_time >= 1.0:
            print(f"--- Frame {self.frame_count} Timing (ms) ---")
            print(f"  to_ndarray: {(to_ndarray_time - process_start_time) * 1000:.2f}")
            print(f"  get_target_face: {(get_target_face_time - to_ndarray_time) * 1000:.2f}")
            if modules.globals.color_correction:
                print(f"  color_correction (initial): {(color_correction_end_time - color_correction_start_time) * 1000:.2f} (Applied: {color_correction_applied})")
            if face_swap_start_time:
                 print(f"  face_swap: {(face_swap_end_time - face_swap_start_time) * 1000:.2f} (Applied: {swap_applied})")
            if face_enhancer_start_time and modules.globals.fp_ui.get('face_enhancer', False) :
                print(f"  face_enhancer (skipped): {(face_enhancer_end_time - face_enhancer_start_time) * 1000:.2f}")
            if final_conversion_start_time and color_correction_applied:
                print(f"  final_color_conversion: {(final_conversion_end_time - final_conversion_start_time) * 1000:.2f} (Applied: {final_conversion_applied})")
            print(f"  VideoFrame.from_ndarray: {(create_frame_end_time - create_frame_start_time) * 1000:.2f}")
            total_proc_time = (process_end_time - process_start_time)
            print(f"  SWAPTRACK_PROC total: {total_proc_time * 1000:.2f}")
            print(f"  Target Face Found: {target_face is not None}, Source Face Loaded: {current_source_face is not None}")
            self.last_log_time = current_time
        
        return new_frame


# --- Signaling Server and WebRTC Handling ---

connected_clients = set()

async def reload_source_face(new_source_path: str, is_temp_file: bool = False):
    """Helper function to load/reload the source face.
    If is_temp_file is True, the file at new_source_path will be deleted after processing.
    """
    global SOURCE_FACE
    print(f"Attempting to load new source face from: {new_source_path}")
    if not new_source_path or not os.path.exists(new_source_path):
        print(f"Invalid or non-existent source path: {new_source_path}")
        if is_temp_file and os.path.exists(new_source_path): # Try to clean up if it was created but is somehow invalid before cv2.imread
             try:
                os.remove(new_source_path)
                print(f"Cleaned up invalid temporary file: {new_source_path}")
             except OSError as e:
                print(f"Error removing temporary file {new_source_path} during cleanup: {e}")
        return False

    face_loaded_successfully = False
    try:
        source_img = cv2.imread(new_source_path)
        if source_img is None:
            print(f"Error: Could not read new source image file: {new_source_path}")
            return False # Keep is_temp_file handling in finally
        
        new_face = get_one_face(source_img)
        if new_face is None:
            print(f"Error: Could not detect face in new source image: {new_source_path}")
            return False # Keep is_temp_file handling in finally
        
        SOURCE_FACE = new_face
        if not is_temp_file:
            modules.globals.source_path = new_source_path 
        else:
            # For temp files, we don't update modules.globals.source_path
            # as this path is transient. The UI won't reflect this temp path.
            print(f"Source face loaded from temporary file: {new_source_path}. Global source_path not updated.")
        
        print(f"Successfully loaded new source face from {new_source_path}")
        face_loaded_successfully = True
        return True
    except Exception as e:
        print(f"Exception loading new source face from {new_source_path}: {e}")
        return False
    finally:
        if is_temp_file and os.path.exists(new_source_path):
            try:
                os.remove(new_source_path)
                print(f"Removed temporary uploaded file: {new_source_path} (Success: {face_loaded_successfully})")
            except OSError as e:
                print(f"Error removing temporary file {new_source_path}: {e}")


async def handle_websocket_signaling(websocket):
    """Callback for handling incoming WebSocket signaling connections."""
    global SOURCE_FACE
    local_pc: Optional[RTCPeerConnection] = None
    peer_addr = websocket.remote_address
    print(f"WebSocket signaling connection from {peer_addr}")
    connected_clients.add(websocket)

    # --- Define PC Event Handlers Inline (WITHOUT DECORATORS) ---

    def on_track(track):
        nonlocal local_pc
        print(f"Track {track.kind} received from {peer_addr}")

        if track.kind == "video":
            processed_track = SwapTrack(track)
            if local_pc:
                local_pc.addTrack(processed_track)
                print(f"Added SwapTrack directly for {peer_addr}")
            else:
                 print(f"Cannot add SwapTrack, local_pc is None for {peer_addr}")


        @track.on("ended")
        async def on_ended():
            print(f"Track {track.kind} ended for {peer_addr}")

    async def on_connectionstatechange():
        nonlocal local_pc
        if local_pc:
             print(f"PC connection state is {local_pc.connectionState} for {peer_addr}")
             if local_pc.connectionState == "failed" or local_pc.connectionState == "closed":
                 print(f"Closing PC due to state: {local_pc.connectionState} for {peer_addr}")
                 await close_pc()
             elif local_pc.connectionState == "connected":
                 print(f"WebRTC connection established for {peer_addr}")

    async def on_icecandidate(candidate):
        nonlocal local_pc
        if candidate and local_pc:
             print(f"Sending ICE candidate to {peer_addr}")
             try:
                 if websocket.state == State.OPEN:
                     await websocket.send(json.dumps({"type": "candidate", "ice": candidate.to_sdp()}))
                 else:
                      print(f"WebSocket not open (state: {websocket.state}) before sending ICE candidate to {peer_addr}.")
                      await close_pc()
             except websockets.exceptions.ConnectionClosed:
                 print(f"Failed to send ICE candidate to {peer_addr}: WebSocket connection was closed.")
                 await close_pc()

    # --- Helper to close the local PC ---
    async def close_pc():
        nonlocal local_pc
        if local_pc and local_pc.connectionState != "closed":
             print(f"Closing RTCPeerConnection for {peer_addr}")
             await local_pc.close()
             local_pc = None

    # --- Create the PeerConnection for this specific client ---
    try:
        local_pc = RTCPeerConnection(configuration=pc_config)
        print(f"Created new RTCPeerConnection for {peer_addr}")

        # --- Attach the event handlers AFTER pc is created ---
        local_pc.on("track")(on_track)
        local_pc.on("connectionstatechange")(on_connectionstatechange)
        local_pc.on("icecandidate")(on_icecandidate)
        print(f"Attached event handlers for {peer_addr}")

    except Exception as e:
        print(f"Error creating RTCPeerConnection for {peer_addr}: {e}")
        connected_clients.discard(websocket)
        return

    # --- Message Handling Loop ---
    try:
        async for message_str in websocket:
            if local_pc is None or local_pc.connectionState == "closed":
                print(f"PC for {peer_addr} is None or closed, ignoring message.")
                break # Exit loop if PC is closed

            message = json.loads(message_str)
            msg_type = message.get("type")
            print(f"Received message type: {msg_type} from {peer_addr}")

            if msg_type == "offer":
                offer = RTCSessionDescription(sdp=message["sdp"], type=message["type"])

                # Now we should be in a 'stable' state as pc was just created
                print(f"Current PC state for {peer_addr}: {local_pc.signalingState}")
                await local_pc.setRemoteDescription(offer)
                print(f"Remote description (offer) set for {peer_addr}.")

                # Create Answer
                print(f"Creating answer for {peer_addr}...")
                answer = await local_pc.createAnswer()
                await local_pc.setLocalDescription(answer)
                print(f"Local description (answer) set for {peer_addr}.")

                response = {"sdp": local_pc.localDescription.sdp, "type": local_pc.localDescription.type}
                if websocket.state == State.OPEN:
                    await websocket.send(json.dumps(response))
                    print(f"Answer sent to {peer_addr}.")
                else:
                    print(f"WebSocket not open (state: {websocket.state}) before sending answer to {peer_addr}.")
                    await close_pc()


            elif msg_type == "candidate":
                 ice_candidate_sdp = message.get("ice")
                 ice_dict = message.get("candidate")

                 candidate_to_add = None
                 if ice_candidate_sdp:
                      candidate_to_add = RTCIceCandidate.from_sdp(ice_candidate_sdp)
                      candidate_to_add.sdpMid = message.get("sdpMid")
                      candidate_to_add.sdpMLineIndex = message.get("sdpMLineIndex")

                 elif ice_dict and isinstance(ice_dict, dict) and 'candidate' in ice_dict:
                       candidate_info = ice_dict
                       candidate_to_add = RTCIceCandidate(
                           sdpMid=candidate_info.get("sdpMid"),
                           sdpMLineIndex=candidate_info.get("sdpMLineIndex"),
                           sdp=candidate_info.get("candidate")
                       )
                 else:
                      print(f"Received incomplete or unexpected ICE candidate format from {peer_addr}: {message}")


                 if candidate_to_add:
                      print(f"Received ICE candidate from {peer_addr}, adding.")
                      try:
                          await local_pc.addIceCandidate(candidate_to_add)
                      except Exception as e:
                           if "adapter type mismatch" in str(e):
                               print(f"Error adding ICE candidate for {peer_addr}: {e}. This often means mismatch between network interfaces used for STUN/TURN and local connection.")
                           else:
                               print(f"Error adding ICE candidate for {peer_addr}: {e}")
                 else:
                      print(f"No valid ICE candidate found in message from {peer_addr}")


            elif msg_type == "update_settings":
                settings = message.get('settings', {})
                # Handle settings update (including source face data)
                source_path = settings.get('sourcePath')
                source_data_url = settings.get('sourceFileData')
                temp_file_path = None
                is_temp = False

                if source_data_url and source_data_url.startswith('data:image'):
                     print(f"Processing image data URL upload from {peer_addr}")
                     try:
                          header, encoded = source_data_url.split(',', 1)
                          image_data = base64.b64decode(encoded)
                          ensure_temp_upload_dir_exists()
                          temp_filename = f"upload_{uuid.uuid4()}.png"
                          temp_file_path = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
                          with open(temp_file_path, 'wb') as f:
                              f.write(image_data)
                          print(f"Saved uploaded image data to temporary file: {temp_file_path}")
                          await reload_source_face(temp_file_path, is_temp_file=True)
                          is_temp = True

                     except Exception as e:
                          print(f"Error processing uploaded image data from {peer_addr}: {e}")
                          if temp_file_path and os.path.exists(temp_file_path):
                              try:
                                  os.remove(temp_file_path)
                              except OSError: pass
                elif source_path:
                     print(f"Processing image path update from {peer_addr}: {source_path}")
                     await reload_source_face(source_path, is_temp_file=False)
                else:
                     print(f"Received settings update from {peer_addr} without valid source data.")

                # Update global settings from the message
                modules.globals.mouth_mask = settings.get('mouthMask', modules.globals.mouth_mask)
                modules.globals.color_correction = settings.get('colorCorrection', modules.globals.color_correction)
                modules.globals.show_mouth_mask_box = settings.get('showMouthMaskBox', modules.globals.show_mouth_mask_box)
                print(f"Applied settings for {peer_addr}. Current globals: MouthMask={modules.globals.mouth_mask}, ColorCorrection={modules.globals.color_correction}, ShowMouthMaskBox={modules.globals.show_mouth_mask_box}")


            else:
                print(f"Received unknown message type {msg_type} from {peer_addr}")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"WebSocket connection closed normally by {peer_addr}.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error for {peer_addr}: {e}")
    except Exception as e:
        print(f"Error during WebSocket signaling with {peer_addr}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Cleaning up resources for {peer_addr}")
        connected_clients.discard(websocket)
        await close_pc()
        print(f"Finished cleanup for {peer_addr}")


async def run_server():
    """Sets up and runs the WebSocket signaling server."""
    global SOURCE_FACE

    if modules.globals.source_path and not SOURCE_FACE:
        print("Pre-loading default source face...")
        await reload_source_face(modules.globals.source_path)

    host = "0.0.0.0"
    port = 9999
    print(f"Starting WebSocket signaling server on ws://{host}:{port}")

    server = await websockets.serve(handle_websocket_signaling, host, port)

    print("WebSocket server running. Waiting for connections...")
    await server.wait_closed()


# --- Argument Parsing and Main Execution ---

def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(description=f"{modules.metadata.name} WebRTC Server")
    program.add_argument('-s', '--source', help='select an initial source face image (optional, can be set via UI)', dest='source_path', required=False, default=None)
    program.add_argument('--frame-processor', help='pipeline of frame processors (face_enhancer is currently disabled)', dest='frame_processor', default=[], choices=[], nargs='*')
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('-l', '--lang', help='Ui language (for console messages)', default="en", dest='lang')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider(s) (e.g., cuda, cpu)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.frame_processors = args.frame_processor
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    modules.globals.fp_ui['face_enhancer'] = False
    
    # Deprecated args translation (keep for compatibility if needed, or remove)
    # ... (keep or remove deprecated arg handling)


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available_providers = onnxruntime.get_available_providers()
    encoded_available_providers = encode_execution_providers(available_providers)
    decoded_providers = []
    for requested_provider in execution_providers:
        found = False
        for provider, encoded_provider in zip(available_providers, encoded_available_providers):
             if requested_provider == encoded_provider:
                decoded_providers.append(provider)
                found = True
                break
             elif requested_provider in encoded_provider:
                 pass
        if not found:
            print(f"Warning: Requested execution provider '{requested_provider}' not available or not recognized. Available: {encoded_available_providers}")

    unique_decoded = list(dict.fromkeys(decoded_providers))
    if not unique_decoded:
        print("Warning: No valid execution providers selected or available. Falling back to CPU.")
        if 'CPUExecutionProvider' in available_providers:
             return ['CPUExecutionProvider']
        else:
             print("Error: CPUExecutionProvider is not available.")
             return []
    return unique_decoded


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        suggested = max(4, int(total_memory_gb / 2))
        return min(suggested, 16)
    except ImportError:
        return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    default_threads = 8
    try:
        import psutil
        core_count = psutil.cpu_count(logical=False)
        if core_count:
            default_threads = max(1, core_count - 1 if core_count > 1 else 1)
    except ImportError:
        pass

    if any(ep in modules.globals.execution_providers for ep in ['DmlExecutionProvider', 'ROCMExecutionProvider', 'CoreMLExecutionProvider']):
         return 1

    return default_threads


def limit_resources() -> None:
    try:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Tensorflow GPU setup failed (ignoring): {e}")
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        try:
            if platform.system().lower() == 'darwin':
                 print("Note: Memory limiting on macOS is often advisory.")
            elif platform.system().lower() == 'windows':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                if not kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory // 2), ctypes.c_size_t(memory)):
                     print("Warning: Failed to set process working set size (memory limit).")
            else:
                import resource
                resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
            print(f"Attempted to limit memory usage to ~{modules.globals.max_memory} GB.")
        except ImportError:
             print("Warning: 'resource' module not available on this platform. Cannot limit memory.")
        except Exception as e:
             print(f"Warning: Failed to limit memory: {e}")


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        try:
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
        except Exception as e:
            print(f"Failed to clear CUDA cache: {e}")


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        print('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed. Please install ffmpeg and ensure it is in your PATH.')
        return False
    print("Core pre-checks passed.")
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')


def destroy(to_quit=True) -> None:
    print("Destroy called. Shutting down server components if necessary...")
    release_resources()
    if os.path.exists(TEMP_UPLOAD_DIR):
         try:
             shutil.rmtree(TEMP_UPLOAD_DIR)
             print(f"Removed temporary upload directory: {TEMP_UPLOAD_DIR}")
         except OSError as e:
             print(f"Error removing temporary upload directory {TEMP_UPLOAD_DIR}: {e}")

    if to_quit:
        print("Exiting.")
        quit()


def run() -> None:
    """Main entry point."""
    parse_args()
    limit_resources()
    if not pre_check():
        return

    if not face_swapper_processor.pre_check():
        return

    try:
        print("Starting asyncio event loop for server...")
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Server stopped by user.")
    finally:
        print("Cleaning up resources...")
        clean_temp()
        release_resources()
        if os.path.exists(TEMP_UPLOAD_DIR):
             try:
                 shutil.rmtree(TEMP_UPLOAD_DIR)
                 print(f"Removed temporary upload directory: {TEMP_UPLOAD_DIR}")
             except OSError as e:
                 print(f"Error removing temporary upload directory {TEMP_UPLOAD_DIR}: {e}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run()
