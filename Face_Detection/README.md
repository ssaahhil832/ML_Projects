# Face Detection

This project demonstrates two face-detection approaches using OpenCV:

1) Haar Cascade for a single image.
2) DNN (Caffe SSD) for real-time webcam detection.

## What I did
- Built a simple image-based detector using a pre-trained Haar Cascade classifier.
- Built a webcam-based detector using a pre-trained Caffe SSD face model.
- Added visualization: bounding boxes, confidence scores, and a live face count.

## How it works
### 1) Haar Cascade (image)
- Loads the cascade XML file (pre-trained face features).
- Reads a test image and converts it to grayscale.
- Runs `detectMultiScale` to find faces.
- Draws rectangles around detected faces.

### 2) DNN SSD (webcam)
- Loads a pre-trained Caffe SSD model (architecture + weights).
- Captures frames from the webcam.
- Converts each frame to a 300x300 blob and runs a forward pass.
- Filters detections by confidence (> 0.5), draws bounding boxes,
  and shows a live count.

## Data / models used
- Haar Cascade: `haarcascade_frontalface_default.xml`
- DNN architecture: `deploy.prototxt`
- DNN weights: `res10_300x300_ssd_iter_140000.caffemodel`
- Test image: `test.jpg` (place your image in this folder)

## Files
- `face_detect_image.py` - Haar Cascade image detector
- `face_detect_webcam.py` - DNN SSD webcam detector
- `haarcascade_frontalface_default.xml`
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

## How to run
### Image detection
1. Put your image as `test.jpg` in this folder.
2. Run:

```bash
python face_detect_image.py
```

### Webcam detection
```bash
python face_detect_webcam.py
```

Press `q` to quit the webcam window.

-----------------------------------------------------------------------------------
Here’s a concise workflow/components list for building a face-detection app like yours (software-only):

Problem definition: real‑time webcam vs image, accuracy target, speed target, output (boxes, count).
Model choice: classic Haar Cascade or DNN SSD; pick based on speed/accuracy tradeoff.
Assets: pre‑trained model files (XML or prototxt + caffemodel) and sample images for testing.
Preprocessing pipeline: resize, color conversion (BGR→gray or blob), normalization.
Inference pipeline: load model once, run detection per image/frame, threshold confidences.
Post‑processing: scale boxes back to original size, draw boxes/labels/count.
UI/IO: image input, webcam stream, display window, key controls.
Evaluation: test on varied images/lighting/poses; adjust thresholds.
Packaging: requirements list, README, run commands, file layout.
-----------------------------------------------------------------------------------