"use strict";

// ... (references for localVideo, remoteVideo, buttons, statusDiv, source inputs, mouthMaskCheckbox) ...
const sourceImagePathInput = document.getElementById("sourceImagePath");
const sourceImageFileInput = document.getElementById("sourceImageFile");
const sourcePreview = document.getElementById("sourcePreview");
const mouthMaskCheckbox = document.getElementById("mouthMaskCheckbox");
const applySettingsButton = document.getElementById("applySettingsButton");

// New references
const showMouthMaskBoxCheckbox = document.getElementById(
  "showMouthMaskBoxCheckbox"
);
const colorCorrectionCheckbox = document.getElementById(
  "colorCorrectionCheckbox"
);
const faceEnhancerCheckbox = document.getElementById("faceEnhancerCheckbox");

const sampleFacesContainer = document.getElementById("sampleFacesContainer");
const sampleFaceImages = document.querySelectorAll(".sample-face-img");

let localStream;
let pc; // PeerConnection
let ws; // WebSocket

const signalingServerUrl = "ws://localhost:9999";
const pcConfig = {
  iceTransportPolicy: "relay",
  iceServers: [
    {
      urls: "turns:standard.relay.metered.ca:443?transport=tcp",
      username: "536f9e1ab6fbecbdb07daeb6",
      credential: "OGZFRgmsTXbAiMwQ",
    },
  ],
};

// --- Event Listeners ---
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
startButton.onclick = start;
stopButton.onclick = stop;
applySettingsButton.onclick = sendSettingsUpdate;

// Add listeners for new checkboxes
mouthMaskCheckbox.onchange = sendSettingsUpdate;
showMouthMaskBoxCheckbox.onchange = sendSettingsUpdate;
colorCorrectionCheckbox.onchange = sendSettingsUpdate;
faceEnhancerCheckbox.onchange = sendSettingsUpdate;

// ... (source file/path input listeners as before) ...
sourceImageFileInput.onchange = (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      sourcePreview.src = e.target.result;
      sourcePreview.style.display = "block";
      sourceImagePathInput.value = "";
      sendSettingsUpdate();
    };
    reader.readAsDataURL(file);
  }
};

sourceImagePathInput.oninput = () => {
  if (sourceImagePathInput.value) {
    sourceImageFileInput.value = "";
    sourcePreview.style.display = "none";
    sourcePreview.src = "#";
  }
  sendSettingsUpdate();
};

// Sample face clicks
sampleFaceImages.forEach((img) => {
  img.addEventListener("click", () => {
    const samplePath = img.getAttribute("data-sample-path");
    if (!samplePath) return;
    sourceImagePathInput.value = samplePath;
    sourceImageFileInput.value = "";
    sourcePreview.style.display = "none";
    sourcePreview.src = "#";
    sampleFaceImages.forEach((o) => o.classList.remove("selected-sample"));
    img.classList.add("selected-sample");
    sendSettingsUpdate();
  });
});

// --- Core Functions ---
async function start() {
  updateStatus("Starting webcam...");
  startButton.disabled = true;
  stopButton.disabled = false;

  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 15 },
      },
    });
    localVideo.srcObject = localStream;
    updateStatus("Webcam started. Connecting to signaling server...");
    connectSignaling();
  } catch (e) {
    alert(`Error starting webcam: ${e.message || e}`);
    updateStatus(`Error: ${e}`);
    stop();
  }
}

function stop() {
  updateStatus("Stopping...");
  if (localStream) localStream.getTracks().forEach((t) => t.stop());
  localStream = null;
  localVideo.srcObject = null;
  if (pc) {
    pc.getSenders().forEach((s) => s.track && pc.removeTrack(s));
    pc.close();
    pc = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  remoteVideo.srcObject = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  updateStatus("Stopped. Ready to start.");
}

function sendSettingsUpdate() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    updateStatus("Error: Signaling connection not open for settings.");
    return;
  }
  let sp = sourceImagePathInput.value,
    sf = null;
  if (
    sourceImageFileInput.files[0] &&
    sourcePreview.src.startsWith("data:image")
  ) {
    sf = sourcePreview.src;
    sp = "";
  }
  const settings = {
    sourcePath: sp,
    sourceFileData: sf,
    mouthMask: mouthMaskCheckbox.checked,
    showMouthMaskBox: showMouthMaskBoxCheckbox.checked,
    colorCorrection: colorCorrectionCheckbox.checked,
  };
  ws.send(JSON.stringify({ type: "update_settings", settings }));
}

function connectSignaling() {
  ws = new WebSocket(signalingServerUrl);
  ws.onopen = async () => {
    updateStatus("Connected to signaling server. Setting up WebRTC...");
    if (!pc) await createPeerConnection();
    sendSettingsUpdate();
    if (localStream && pc.signalingState === "stable") {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      ws.send(JSON.stringify({ sdp: offer.sdp, type: offer.type }));
      updateStatus("Offer sent to server.");
    }
  };

  ws.onmessage = async (evt) => {
    const msg = JSON.parse(evt.data);

    if (msg.type === "candidate") {
      // full trickle ICE
      try {
        await pc.addIceCandidate(msg.candidate);
        console.log("Added remote ICE candidate:", msg.candidate);
      } catch (e) {
        console.warn("Failed to add ICE candidate:", e);
      }
      return;
    }

    if (msg.sdp) {
      if (msg.type === "answer") {
        await pc.setRemoteDescription(msg);
        updateStatus("WebRTC connection established.");
      }
      // (handle incoming offer if ever)
    }
  };

  ws.onerror = (e) => {
    updateStatus("Signaling server connection error.");
    stop();
  };

  ws.onclose = () => {
    updateStatus("Disconnected from signaling server.");
    stopButton.disabled = true;
    startButton.disabled = false;
  };
}

async function createPeerConnection() {
  pc = new RTCPeerConnection(pcConfig);
  pc.onicecandidate = (e) => {
    const payload = {
      type: "candidate",
      candidate: e.candidate ? e.candidate.toJSON() : null,
    };
    ws.send(JSON.stringify(payload));
    console.log("Sent local ICE candidate:", payload.candidate);
  };

  pc.ontrack = (e) => {
    remoteVideo.srcObject = e.streams[0];
    updateStatus("Receiving remote video stream.");
  };

  localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

  pc.oniceconnectionstatechange = () => {
    updateStatus(`ICE State: ${pc.iceConnectionState}`);
    if (pc.iceConnectionState === "failed") console.error("WebRTC failed");
  };
  pc.onconnectionstatechange = () => {
    updateStatus(`Connection State: ${pc.connectionState}`);
  };
}

function updateStatus(msg) {
  document.getElementById("status").textContent = msg;
  console.log("Status:", msg);
}

stopButton.disabled = true;
