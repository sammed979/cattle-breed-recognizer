
package com.bharatpashudhan.breedrecognition;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Module for handling user feedback on breed predictions
 * Implements a singleton pattern for easy access across the app
 */
public class FeedbackModule {
    private static FeedbackModule instance;
    private static final String API_BASE_URL = "https://api.bharatpashudhan.org/";
    private static final String FEEDBACK_ENDPOINT = "feedback";
    private static final String UPLOAD_ENDPOINT = "upload";
    
    // Cache of breed names and IDs
    private Map<Integer, String> breedMap = new HashMap<>();
    
    private FeedbackModule() {
        // Initialize breed map
        initializeBreedMap();
    }
    
    public static synchronized FeedbackModule getInstance() {
        if (instance == null) {
            instance = new FeedbackModule();
        }
        return instance;
    }
    
    private void initializeBreedMap() {
        // In a real app, this would be loaded from a resource file or API
        breedMap.put(0, "Gir");
        breedMap.put(1, "Sahiwal");
        breedMap.put(2, "Red Sindhi");
        // Add more breeds...
    }
    
    /**
     * Sets up and displays the feedback UI
     * 
     * @param context The activity context
     * @param predictedBreed The breed predicted by the model
     * @param breedIndex The index of the predicted breed
     * @param image The image that was classified
     */
    public void setupFeedbackUI(Context context, String predictedBreed, int breedIndex, Bitmap image) {
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        LayoutInflater inflater = ((Activity) context).getLayoutInflater();
        View dialogView = inflater.inflate(R.layout.feedback_dialog, null);
        
        // Set up the spinner with breed options
        Spinner breedSpinner = dialogView.findViewById(R.id.breedSpinner);
        List<String> breedNames = new ArrayList<>(breedMap.values());
        ArrayAdapter<String> adapter = new ArrayAdapter<>(context, android.R.layout.simple_spinner_item, breedNames);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        breedSpinner.setAdapter(adapter);
        
        // Set the spinner to the predicted breed
        int spinnerPosition = adapter.getPosition(predictedBreed);
        breedSpinner.setSelection(spinnerPosition);
        
        // Set up the buttons
        Button correctButton = dialogView.findViewById(R.id.correctButton);
        Button incorrectButton = dialogView.findViewById(R.id.incorrectButton);
        Button cancelButton = dialogView.findViewById(R.id.cancelButton);
        
        builder.setView(dialogView);
        AlertDialog dialog = builder.create();
        
        // Set up button click listeners
        correctButton.setOnClickListener(v -> {
            // User confirms prediction is correct
            submitFeedback(context, breedIndex, breedIndex, image, true);
            dialog.dismiss();
        });
        
        incorrectButton.setOnClickListener(v -> {
            // User indicates prediction is incorrect
            int selectedBreedIndex = getBreedIndexByName((String) breedSpinner.getSelectedItem());
            submitFeedback(context, breedIndex, selectedBreedIndex, image, false);
            dialog.dismiss();
        });
        
        cancelButton.setOnClickListener(v -> dialog.dismiss());
        
        dialog.show();
    }
    
    private int getBreedIndexByName(String breedName) {
        for (Map.Entry<Integer, String> entry : breedMap.entrySet()) {
            if (entry.getValue().equals(breedName)) {
                return entry.getKey();
            }
        }
        return -1; // Not found
    }
    
    /**
     * Submits feedback to the API
     * 
     * @param context The activity context
     * @param predictedBreedIndex The index of the breed predicted by the model
     * @param actualBreedIndex The index of the actual breed (from user feedback)
     * @param image The image that was classified
     * @param isCorrect Whether the prediction was correct
     */
    private void submitFeedback(Context context, int predictedBreedIndex, int actualBreedIndex, 
                               Bitmap image, boolean isCorrect) {
        new FeedbackTask(context, predictedBreedIndex, actualBreedIndex, image, isCorrect).execute();
    }
    
    /**
     * AsyncTask to handle API communication in the background
     */
    private static class FeedbackTask extends AsyncTask<Void, Void, Boolean> {
        private Context context;
        private int predictedBreedIndex;
        private int actualBreedIndex;
        private Bitmap image;
        private boolean isCorrect;
        
        public FeedbackTask(Context context, int predictedBreedIndex, int actualBreedIndex, 
                           Bitmap image, boolean isCorrect) {
            this.context = context;
            this.predictedBreedIndex = predictedBreedIndex;
            this.actualBreedIndex = actualBreedIndex;
            this.image = image;
            this.isCorrect = isCorrect;
        }
        
        @Override
        protected Boolean doInBackground(Void... voids) {
            try {
                // First, submit feedback
                submitFeedbackToApi();
                
                // If prediction was incorrect, also upload the image for dataset improvement
                if (!isCorrect) {
                    uploadImageToApi();
                }
                
                return true;
            } catch (Exception e) {
                e.printStackTrace();
                return false;
            }
        }
        
        @Override
        protected void onPostExecute(Boolean success) {
            if (success) {
                Toast.makeText(context, "Thank you for your feedback!", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(context, "Failed to submit feedback. Please try again later.", 
                              Toast.LENGTH_SHORT).show();
            }
        }
        
        private void submitFeedbackToApi() throws IOException, JSONException {
            URL url = new URL(API_BASE_URL + FEEDBACK_ENDPOINT);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            
            JSONObject feedbackData = new JSONObject();
            feedbackData.put("predicted_breed_id", predictedBreedIndex);
            feedbackData.put("actual_breed_id", actualBreedIndex);
            feedbackData.put("is_correct", isCorrect);
            
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = feedbackData.toString().getBytes("utf-8");
                os.write(input, 0, input.length);
            }
            
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP error code: " + responseCode);
            }
            
            connection.disconnect();
        }
        
        private void uploadImageToApi() throws IOException, JSONException {
            URL url = new URL(API_BASE_URL + UPLOAD_ENDPOINT);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setDoOutput(true);
            
            // Convert bitmap to base64 string
            ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);
            byte[] byteArray = byteArrayOutputStream.toByteArray();
            String base64Image = Base64.getEncoder().encodeToString(byteArray);
            
            JSONObject uploadData = new JSONObject();
            uploadData.put("image", base64Image);
            uploadData.put("breed_id", actualBreedIndex);
            
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = uploadData.toString().getBytes("utf-8");
                os.write(input, 0, input.length);
            }
            
            int responseCode = connection.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new IOException("HTTP error code: " + responseCode);
            }
            
            connection.disconnect();
        }
    }
}
