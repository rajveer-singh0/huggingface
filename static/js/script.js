document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadBox = document.getElementById("uploadBox");
  const fileInput = document.getElementById("fileInput");
  const startCameraBtn = document.getElementById("startCamera");
  const cameraBox = document.getElementById("cameraBox");

  const cameraModal = document.getElementById("cameraModal");
  const resultsModal = document.getElementById("resultsModal");
  const closeModalBtns = document.querySelectorAll(".close");

  const cameraFeed = document.getElementById("cameraFeed");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const captureBtn = document.getElementById("captureBtn");
  const retakeBtn = document.getElementById("retakeBtn");
  const useImageBtn = document.getElementById("useImageBtn");

  const resultImage = document.getElementById("resultImage");
  const detectionResults = document.getElementById("detectionResults");
  const resultsList = document.getElementById("resultsList");
  const detectionBadge = document.getElementById("detectionBadge");
  const badgeText = document.getElementById("badgeText");

  let stream = null;
  let capturedImage = null;

  // =========================
  // Event Listeners
  // =========================

  // Click to open file picker
  uploadBox.addEventListener("click", () => fileInput.click());

  // File selected via input
  fileInput.addEventListener("change", handleFileUpload);

  // Start camera modal
  startCameraBtn.addEventListener("click", openCameraModal);

  // Close modals (X buttons)
  closeModalBtns.forEach((btn) => {
    btn.addEventListener("click", closeModalHandler);
  });

  // Camera actions
  captureBtn.addEventListener("click", captureImage);
  retakeBtn.addEventListener("click", retakeImage);
  useImageBtn.addEventListener("click", useCapturedImage);

  // Drag & Drop
  uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.classList.add("dragover");
  });

  uploadBox.addEventListener("dragleave", () => {
    uploadBox.classList.remove("dragover");
  });

  uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.classList.remove("dragover");

    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileUpload();
    }
  });

  // Close modals when clicking outside
  window.addEventListener("click", (event) => {
    if (event.target === cameraModal) {
      closeModalHandler();
    }
    if (event.target === resultsModal) {
      closeModalHandler();
    }
  });

  // =========================
  // Functions
  // =========================

  function handleFileUpload() {
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];

      // Validate file type
      const validTypes = ["image/jpeg", "image/jpg", "image/png"];
      if (!validTypes.includes(file.type)) {
        alert("Please upload a valid image file (JPEG, JPG, or PNG)");
        return;
      }

      processImage(file);
    }
  }

  function openCameraModal() {
    cameraModal.style.display = "block";
    startCamera();
  }

  function closeModalHandler() {
    cameraModal.style.display = "none";
    resultsModal.style.display = "none";

    // Stop camera stream when modal is closed
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }

    // Reset camera UI
    captureBtn.style.display = "inline-block";
    retakeBtn.style.display = "none";
    useImageBtn.style.display = "none";
    capturedImage = null;
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      cameraFeed.srcObject = stream;
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert(
        "Could not access the camera. Please make sure you have granted camera permissions."
      );
      cameraModal.style.display = "none";
    }
  }

  function captureImage() {
    const context = cameraCanvas.getContext("2d");
    cameraCanvas.width = cameraFeed.videoWidth;
    cameraCanvas.height = cameraFeed.videoHeight;

    context.drawImage(
      cameraFeed,
      0,
      0,
      cameraCanvas.width,
      cameraCanvas.height
    );

    capturedImage = cameraCanvas.toDataURL("image/jpeg");

    captureBtn.style.display = "none";
    retakeBtn.style.display = "inline-block";
    useImageBtn.style.display = "inline-block";
  }

  function retakeImage() {
    captureBtn.style.display = "inline-block";
    retakeBtn.style.display = "none";
    useImageBtn.style.display = "none";
    capturedImage = null;
  }

  function useCapturedImage() {
    if (capturedImage) {
      // Convert data URL to blob, then to File object
      fetch(capturedImage)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "captured.jpg", {
            type: "image/jpeg",
          });

          processImage(file);

          // Close camera modal, stop camera
          cameraModal.style.display = "none";
          if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
          }
        });
    }
  }

  async function processImage(file) {
    // Show loading UI
    const loadingHTML = `
      <div style="text-align: center; padding: 3rem;">
        <div style="display: inline-block; width: 60px; height: 60px; border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 1rem;"></div>
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">Processing Image...</h3>
        <p style="color: #666;">Our AI is analyzing your image</p>
      </div>
      <style>
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      </style>
    `;

    resultsList.innerHTML = loadingHTML;
    detectionBadge.style.display = "none";
    resultImage.src = "";
    resultsModal.style.display = "block";

    // Build FormData to send to Flask backend
    const formData = new FormData();
    formData.append("file", file); // must match Flask key

    try {
      const response = await fetch("/process_image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // Server returned 4xx/5xx
        const text = await response.text();
        console.error("Server error:", response.status, text);
        throw new Error("Server error: " + response.status);
      }

      const data = await response.json();

      if (data.success) {
        // Show the processed image (or fallback to local file)
        resultImage.src =
          data.processed_image && data.processed_image !== ""
            ? data.processed_image
            : URL.createObjectURL(file);

        if (data.detections && data.detections.length > 0) {
          const mainDetection = data.detections[0];
          const mainConfidence = Math.round(mainDetection.confidence * 100);

          // Show badge
          detectionBadge.style.display = "flex";
          badgeText.textContent = `${mainDetection.class} (${mainConfidence}%)`;

          // Build detection cards
          let resultsHTML = "";
          data.detections.forEach((detection, index) => {
            const confidence = Math.round(detection.confidence * 100);
            const iconClass =
              detection.class === "uniform" ? "fa-user-tie" : "fa-user";

            resultsHTML += `
              <div class="result-item" style="display: flex; align-items: center; justify-content: space-between; padding: 1.5rem; background: rgba(255, 255, 255, 0.7); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.1); transition: all 0.3s ease; position: relative; overflow: hidden; animation: slideInUp 0.5s ease-out ${
                index * 0.1
              }s both;">
                <div class="detection-info" style="display: flex; align-items: center; gap: 1rem;">
                  <div class="detection-icon" style="width: 50px; height: 50px; border-radius: 50%; background: linear-gradient(45deg, #667eea, #764ba2); display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem;">
                    <i class="fas ${iconClass}"></i>
                  </div>
                  <div>
                    <div class="detection-class" style="font-weight: 600; color: #2c3e50; font-size: 1.2rem; text-transform: capitalize;">${
                      detection.class
                    }</div>
                    <div class="detection-label" style="color: #666; font-size: 0.9rem;">Detection Result</div>
                  </div>
                </div>
                <div class="confidence-score" style="display: flex; align-items: center; gap: 1rem;">
                  <div class="confidence-bar" style="width: 120px; height: 8px; background: rgba(102, 126, 234, 0.2); border-radius: 4px; overflow: hidden; position: relative;">
                    <div class="confidence-fill" style="height: 100%; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 4px; width: ${confidence}%; transition: width 1s ease-out 0.5s;"></div>
                  </div>
                  <div class="confidence-percentage" style="font-weight: 600; color: #667eea; font-size: 1.1rem; min-width: 50px;">${confidence}%</div>
                </div>
              </div>
            `;
          });

          resultsList.innerHTML = resultsHTML;
        } else {
          // No detections
          detectionBadge.style.display = "none";
          resultsList.innerHTML = `
            <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.7); border-radius: 15px; border: 1px solid rgba(255, 193, 7, 0.3);">
              <div style="font-size: 2rem; color: #ffc107; margin-bottom: 1rem;">
                <i class="fas fa-search"></i>
              </div>
              <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">No Uniforms Detected</h3>
              <p style="color: #666;">No uniform patterns were found in this image.</p>
            </div>
          `;
        }
      } else {
        // Backend returned success: false
        detectionBadge.style.display = "none";
        resultsList.innerHTML = `
          <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.7); border-radius: 15px; border: 1px solid rgba(220, 53, 69, 0.3);">
            <div style="font-size: 2rem; color: #dc3545; margin-bottom: 1rem;">
              <i class="fas fa-exclamation-triangle"></i>
            </div>
            <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">Processing Error</h3>
            <p style="color: #666;">${
              data.error || "Unknown error occurred while processing the image."
            }</p>
          </div>
        `;
      }
    } catch (error) {
      console.error("Error:", error);
      detectionBadge.style.display = "none";
      resultsList.innerHTML = `
        <div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.7); border-radius: 15px; border: 1px solid rgba(220, 53, 69, 0.3);">
          <div style="font-size: 2rem; color: #dc3545; margin-bottom: 1rem;">
            <i class="fas fa-exclamation-triangle"></i>
          </div>
          <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">Network Error</h3>
          <p style="color: #666;">An error occurred while processing the image.</p>
        </div>
      `;
    }
  }
});
