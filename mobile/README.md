# Mobile App Integration (Bharat Pashudhan App)

## Features
- Integrate TensorFlow Lite model for breed prediction
- Camera-based image capture
- UI for breed prediction, feedback submission, and offline access
- Sync predictions and feedback with BPA backend using secure APIs

## Integration Steps
1. Add `breed_predictor.tflite` to the app assets
2. Implement camera capture and image preprocessing
3. Use TFLite Interpreter for on-device inference
4. Design UI for prediction results and feedback
5. Connect to backend API for feedback and dataset expansion
6. Support OTA model updates

## Directory Structure
- `app/` : Android source code
  - `MainActivity.java` : Main UI and camera logic
  - `breed_predictor.tflite` : Model file
  - `feedback_module.java` : Feedback submission logic

Refer to the backend API documentation for endpoint details and authentication.
