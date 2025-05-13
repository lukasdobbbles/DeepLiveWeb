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
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
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
      sourcePreview.src = e.target.result; // e.target.result is the data URL
      sourcePreview.style.display = "block";
      console.log(
        "File selected for upload. Its data URL is ready in sourcePreview.src."
      );
      sourceImagePathInput.value = ""; // Clear path input as we are using file data
      // Automatically send settings update when a new file is chosen and loaded for preview
      sendSettingsUpdate();
    };
    reader.readAsDataURL(file); // Reads the file as a data URL string
  }
};

sourceImagePathInput.oninput = () => {
  // Changed from onchange to oninput for more responsive UI
  if (sourceImagePathInput.value) {
    sourceImageFileInput.value = ""; // Clear file input
    sourcePreview.style.display = "none";
    sourcePreview.src = "#";
  }
  // Debounce or call sendSettingsUpdate directly, or rely on Apply button
  // For simplicity, let's make it so Apply button is needed for path changes for now,
  // or user can just call sendSettingsUpdate if they want it to be instant.
  // For consistency with file input, let's try to send it.
  sendSettingsUpdate();
};

// Add click listener for sample faces
sampleFaceImages.forEach((img) => {
  img.addEventListener("click", () => {
    // Get the path from the data attribute
    const samplePath = img.getAttribute("data-sample-path");
    if (samplePath) {
      console.log(`Sample face clicked: ${samplePath}`);

      // Update the source image path input
      sourceImagePathInput.value = samplePath;

      // Clear the file input and hide preview
      sourceImageFileInput.value = ""; // Clear file input selection
      sourcePreview.style.display = "none";
      sourcePreview.src = "#"; // Reset preview src

      // Update visual selection indicator
      sampleFaceImages.forEach((otherImg) =>
        otherImg.classList.remove("selected-sample")
      );
      img.classList.add("selected-sample");

      // Send the update to the server
      sendSettingsUpdate();
    } else {
      console.warn(
        "Clicked sample face image is missing data-sample-path attribute."
      );
    }
  });
});

// --- Core Functions ---
async function start() {
  console.log(
    "Start button clicked. Attempting to start webcam and connect..."
  );
  updateStatus("Starting webcam...");
  startButton.disabled = true;
  stopButton.disabled = false;

  try {
    console.log(
      "Requesting user media (webcam) with 720p and explicitly low initial FPS constraint for testing..."
    );
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 10 },
      },
    });
    localVideo.srcObject = localStream;
    const trackSettings = localStream.getVideoTracks()[0].getSettings();
    console.log(
      `Local stream obtained. Actual resolution: ${trackSettings.width}x${trackSettings.height}, FPS: ${trackSettings.frameRate}`
    );
    updateStatus("Webcam started. Connecting to signaling server...");

    // Connect to signaling server
    connectSignaling();
  } catch (e) {
    console.error("Error starting webcam or initial connection:", e);
    // Check if the error is due to resolution constraints
    if (
      e.name === "OverconstrainedError" ||
      e.name === "ConstraintNotSatisfiedError"
    ) {
      alert(
        `Could not obtain 720p resolution. Your webcam might not support it. Error: ${e.message}`
      );
    } else {
      alert(`Error starting webcam: ${e.toString()}`);
    }
    updateStatus(`Error: ${e.toString()}`);
    stop(); // Reset UI
    return;
  }
}

function stop() {
  console.log("Stop button clicked.");
  updateStatus("Stopping...");

  // Stop local video track
  if (localStream) {
    localStream.getTracks().forEach((track) => track.stop());
    console.log("Local stream tracks stopped.");
  }
  localStream = null;
  localVideo.srcObject = null; // Clear local video display

  // Close PeerConnection
  if (pc) {
    // Remove tracks before closing to avoid issues
    pc.getSenders().forEach((sender) => {
      if (sender.track) {
        // Check if track still exists before trying to remove
        try {
          pc.removeTrack(sender);
          console.log("PC sender track removed.");
        } catch (e) {
          console.warn("Error removing track from sender:", e);
        }
      }
    });
    pc.close();
    console.log("PeerConnection closed.");
    pc = null; // Ensure pc is reset
  }

  // Close WebSocket
  if (ws) {
    ws.close();
    console.log("WebSocket closed.");
    ws = null; // Ensure ws is reset
  }

  remoteVideo.srcObject = null; // Clear remote video display
  startButton.disabled = false;
  stopButton.disabled = true;
  updateStatus("Stopped. Ready to start.");
}

function sendSettingsUpdate() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn(
      "WebSocket not open. Cannot send settings. WS state:",
      ws ? ws.readyState : "null"
    );
    updateStatus("Error: Signaling connection not open for settings.");
    return;
  }

  let sourcePathValue = sourceImagePathInput.value;
  let sourceFileDataValue = null; // New variable for file data

  // Check if a file was selected via input and its preview (data URL) is available
  if (
    sourceImageFileInput.files[0] &&
    sourcePreview.src && // Ensure src is populated
    sourcePreview.src.startsWith("data:image")
  ) {
    sourceFileDataValue = sourcePreview.src; // Send the data URL
    sourcePathValue = ""; // Clear path value since we're sending file data
    console.log("Preparing to send image file data (Data URL) via WebSocket.");
  } else if (sourcePathValue) {
    console.log("Preparing to send image path via WebSocket.");
  } else {
    console.log("No source image path or file data to send for update.");
  }

  const settings = {
    sourcePath: sourcePathValue,
    sourceFileData: sourceFileDataValue,
    mouthMask: mouthMaskCheckbox.checked,
    showMouthMaskBox: showMouthMaskBoxCheckbox.checked,
    colorCorrection: colorCorrectionCheckbox.checked,
    // faceEnhancer is effectively disabled on backend
  };

  console.log("Sending settings update:", settings);
  updateStatus("Sending settings to server...");
  ws.send(JSON.stringify({ type: "update_settings", settings: settings }));
}

function connectSignaling() {
  console.log("connectSignaling called. Attempting WebSocket connection...");
  if (
    ws &&
    (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
  ) {
    console.log("WebSocket already open or connecting. State:", ws.readyState);
    return;
  }

  ws = new WebSocket(signalingServerUrl);
  console.log(`WebSocket created for ${signalingServerUrl}`);

  ws.onopen = async () => {
    console.log("WebSocket connection opened.");
    updateStatus("Connected to signaling server. Setting up WebRTC...");
    if (!pc) {
      // Create PeerConnection only after WebSocket is open
      await createPeerConnection();
    } else {
      console.log(
        "PeerConnection already exists. Signaling state:",
        pc.signalingState
      );
    }

    // Send the current settings (including potentially uploaded image data)
    console.log("WebSocket open, sending current settings...");
    sendSettingsUpdate();

    // If local stream is ready, and PC is ready, create offer
    if (localStream && pc && pc.signalingState === "stable") {
      console.log("Creating offer as WebSocket is open and PC is stable.");
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log(
          "Offer created and local description set. Sending offer..."
        );
        ws.send(JSON.stringify({ sdp: offer.sdp, type: offer.type }));
        updateStatus("Offer sent to server.");
      } catch (e) {
        console.error("Error creating or sending offer:", e);
        updateStatus(`Error creating offer: ${e.toString()}`);
      }
    } else {
      console.warn(
        "Cannot create offer. localStream:",
        !!localStream,
        "pc:",
        !!pc,
        "pc.signalingState:",
        pc ? pc.signalingState : "N/A"
      );
    }
  };

  ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);
    console.log("WebSocket message received:", message);

    if (!pc) {
      console.error(
        "Received WebSocket message, but PeerConnection (pc) is null!"
      );
      return;
    }

    try {
      // Wrap message handling in try-catch
      if (message.sdp) {
        console.log(`Received ${message.type} SDP from server.`);
        if (message.type === "answer") {
          await pc.setRemoteDescription(new RTCSessionDescription(message));
          console.log("Remote description (answer) set.");
          updateStatus("WebRTC connection established.");
        } else if (message.type === "offer") {
          // Handle offer if acting as callee (less common for this setup)
          console.warn(
            "Received offer from server - unexpected in this client role?"
          );
          // Basic handling: set remote, create answer, set local, send answer
          await pc.setRemoteDescription(new RTCSessionDescription(message));
          console.log("Remote description (offer) set.");
          const answer = await pc.createAnswer();
          await pc.setLocalDescription(answer);
          console.log(
            "Answer created and local description set. Sending answer..."
          );
          ws.send(JSON.stringify({ sdp: answer.sdp, type: answer.type }));
          updateStatus("Answer sent to server.");
        } else {
          console.warn("Received unknown SDP type:", message.type);
        }
      } else if (message.ice) {
        console.log("Received ICE candidate from server.");
        await pc.addIceCandidate(new RTCIceCandidate(message.ice));
        console.log("Added ICE candidate.");
      } else if (message.type === "error") {
        console.error("Received error message from server:", message.message);
        updateStatus(`Server Error: ${message.message}`);
        // Optionally stop connection on server error
        // stop();
      } else {
        console.log("Received non-SDP/ICE message:", message);
        // Handle other message types if necessary
      }
    } catch (e) {
      console.error("Error processing WebSocket message:", e);
      updateStatus(`Error processing message: ${e.toString()}`);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
    updateStatus("Signaling server connection error.");
    // alert("WebSocket error. Check console."); // Potentially annoying
    stop(); // Reset UI elements
  };

  ws.onclose = () => {
    console.log("WebSocket connection closed.");
    updateStatus("Disconnected from signaling server.");
    if (pc && pc.connectionState !== "closed") {
      console.log("Closing PeerConnection as WebSocket closed.");
      // pc.close(); // This was causing issues if pc.close was also called in stop()
    }
    // pc = null; // Reset pc if connection is fully torn down.
    // ws = null; // Already handled by subsequent connectSignaling calls.
    stopButton.disabled = true;
    startButton.disabled = false;
  };
}

async function createPeerConnection() {
  console.log("createPeerConnection called.");
  if (pc) {
    console.warn("PeerConnection already exists. Closing existing one first.");
    // Ensure existing connection is fully closed before creating new one
    if (pc.connectionState !== "closed") {
      console.log(
        "Closing existing PC before creating new one. State:",
        pc.connectionState
      );
      pc.close();
    }
    pc = null; // Explicitly nullify before reassigning
  }
  try {
    pc = new RTCPeerConnection(pcConfig);
    console.log("New RTCPeerConnection created.");

    pc.onicecandidate = (event) => {
      if (event.candidate && ws && ws.readyState === WebSocket.OPEN) {
        console.log("Local ICE candidate found. Sending to server...");
        ws.send(JSON.stringify({ ice: event.candidate }));
      } else {
        console.log("ICE gathering finished or WebSocket not open.");
      }
    };

    pc.ontrack = (event) => {
      console.log("Remote track received.");
      if (remoteVideo.srcObject !== event.streams[0]) {
        remoteVideo.srcObject = event.streams[0];
        console.log("Set remote stream to video element.");
        updateStatus("Receiving remote video stream.");
      }
    };

    // Add local tracks *after* setting up event handlers
    if (localStream) {
      localStream.getTracks().forEach((track) => {
        try {
          pc.addTrack(track, localStream);
          console.log(`Local track added to PeerConnection: ${track.kind}`);
        } catch (e) {
          console.error("Error adding track:", e);
        }
      });
    } else {
      console.warn("createPeerConnection called but localStream is null!");
    }

    pc.oniceconnectionstatechange = () => {
      console.log(`ICE connection state changed: ${pc.iceConnectionState}`);
      updateStatus(`ICE State: ${pc.iceConnectionState}`);
      if (
        pc.iceConnectionState === "failed" ||
        pc.iceConnectionState === "disconnected" ||
        pc.iceConnectionState === "closed"
      ) {
        // Handle connection failure/closure
        console.error(`WebRTC connection ${pc.iceConnectionState}.`);
        // Optionally attempt restart or just stop
        // stop(); // Consider if auto-stop is desired on failure
      }
    };

    pc.onconnectionstatechange = () => {
      console.log(`Connection state change: ${pc.connectionState}`);
      updateStatus(`Connection State: ${pc.connectionState}`);
      if (pc.connectionState === "connected") {
        // Connection successful
      } else if (pc.connectionState === "failed") {
        console.error("WebRTC connection failed.");
        // stop(); // Consider stopping
      } else if (
        pc.connectionState === "disconnected" ||
        pc.connectionState === "closed"
      ) {
        console.log(`WebRTC connection ${pc.connectionState}.`);
        // May already be handled by stop() or ICE state change
      }
    };

    pc.onsignalingstatechange = () => {
      console.log(`Signaling state change: ${pc.signalingState}`);
      if (pc.signalingState === "closed") {
        console.log("Signaling state is closed.");
        // Ensure cleanup if needed
      }
    };
  } catch (e) {
    console.error("Failed to create PeerConnection:", e);
    updateStatus(`Error creating PeerConnection: ${e.toString()}`);
    stop(); // Cleanup on failure
  }
}

function updateStatus(message) {
  const statusDiv = document.getElementById("status");
  if (statusDiv) {
    statusDiv.textContent = message;
  }
  console.log(`Status: ${message}`);
}

// Initial state
stopButton.disabled = true;
