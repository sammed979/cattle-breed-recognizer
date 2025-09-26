package com.bharatpashudhan.breedrecognition;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_IMAGE_CAPTURE = 101;
    private static final int REQUEST_GALLERY_IMAGE = 102;
    
    private Button captureButton;
    private Button galleryButton;
    private Button predictButton;
    private ImageView imagePreview;
    private TextView resultText;
    private ProgressBar progressBar;
    private Bitmap imageBitmap;
    
    private Interpreter tflite;
    private final int imageSizeX = 224;
    private final int imageSizeY = 224;
    private final int NUM_CLASSES = 74;
    
    // Mapping of breed indices to names (to be populated from resources)
    private Map<Integer, String> breedLabels = new HashMap<>();
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize UI components
        captureButton = findViewById(R.id.captureButton);
        galleryButton = findViewById(R.id.galleryButton);
        predictButton = findViewById(R.id.predictButton);
        imagePreview = findViewById(R.id.imagePreview);
        resultText = findViewById(R.id.resultText);
        progressBar = findViewById(R.id.progressBar);
        
        // Load TFLite model
        try {
            tflite = new Interpreter(loadModelFile());
            loadBreedLabels();
        } catch (IOException e) {
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
        
        // Set click listeners
        captureButton.setOnClickListener(v -> {
            if (checkCameraPermission()) {
                openCamera();
            } else {
                requestCameraPermission();
            }
        });
        
        galleryButton.setOnClickListener(v -> openGallery());
        
        predictButton.setOnClickListener(v -> {
            if (imageBitmap != null) {
                progressBar.setVisibility(View.VISIBLE);
                predictBreed(imageBitmap);
            } else {
                Toast.makeText(MainActivity.this, "Please capture or select an image first", Toast.LENGTH_SHORT).show();
            }
        });
    }
    
    private boolean checkCameraPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
    
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
    }
    
    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }
    
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_GALLERY_IMAGE);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE && data != null) {
                Bundle extras = data.getExtras();
                imageBitmap = (Bitmap) extras.get("data");
                imagePreview.setImageBitmap(imageBitmap);
            } else if (requestCode == REQUEST_GALLERY_IMAGE && data != null) {
                Uri selectedImage = data.getData();
                try {
                    imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                    imagePreview.setImageBitmap(imageBitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    
    private MappedByteBuffer loadModelFile() throws IOException {
        String MODEL_FILE = "breed_predictor.tflite";
        FileInputStream inputStream = new FileInputStream(getAssets().openFd(MODEL_FILE).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = getAssets().openFd(MODEL_FILE).getStartOffset();
        long declaredLength = getAssets().openFd(MODEL_FILE).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    private void loadBreedLabels() {
        // In a real app, load from a resource file
        // For this example, we'll add a few sample breeds
        breedLabels.put(0, "Gir");
        breedLabels.put(1, "Sahiwal");
        breedLabels.put(2, "Red Sindhi");
        // Add more breeds...
    }
    
    private void predictBreed(Bitmap bitmap) {
        // Resize the bitmap to the required input size
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
        
        // Convert bitmap to ByteBuffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY * 3);
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();
        
        int[] intValues = new int[imageSizeX * imageSizeY];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());
        
        // Normalize pixel values to [0, 1]
        for (int pixelValue : intValues) {
            inputBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
            inputBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
            inputBuffer.putFloat((pixelValue & 0xFF) / 255.0f);
        }
        
        // Output buffer
        float[][] outputBuffer = new float[1][NUM_CLASSES];
        
        // Run inference
        tflite.run(inputBuffer, outputBuffer);
        
        // Find the class with highest probability
        int maxIndex = 0;
        float maxProb = outputBuffer[0][0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (outputBuffer[0][i] > maxProb) {
                maxProb = outputBuffer[0][i];
                maxIndex = i;
            }
        }
        
        // Get breed name and display result
        String breedName = breedLabels.getOrDefault(maxIndex, "Unknown");
        float confidence = maxProb * 100;
        
        runOnUiThread(() -> {
            progressBar.setVisibility(View.GONE);
            resultText.setText(String.format("Breed: %s\nConfidence: %.1f%%", breedName, confidence));
            
            // Show feedback option
            showFeedbackOption(breedName, maxIndex);
        });
    }
    
    private void showFeedbackOption(String predictedBreed, int breedIndex) {
        // In a real app, implement a dialog or UI for feedback
        // For this example, we'll just show a toast
        Toast.makeText(this, "Tap to provide feedback if prediction is incorrect", Toast.LENGTH_LONG).show();
        
        // In a real app, you would call FeedbackModule here
        FeedbackModule.getInstance().setupFeedbackUI(this, predictedBreed, breedIndex, imageBitmap);
    }
}
