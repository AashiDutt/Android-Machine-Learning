package com.example.mnist_app;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
public class MainActivity extends AppCompatActivity {

    // UI elements
    ImageView imageView;
    TextView textView;
    static{
        System.loadLibrary("tensorflow_inference");
    }
    // variables for communicating with model
    private static final String MODEL_FILE = "file:///android_asset/optimized_frozen_mnist_model.pb";
    private static final String INPUT_NODE = "x_input";
    private static final int[] INPUT_SHAPE = {1,784};
    private static final String OUTPUT_NODE = "y_actual";
    private TensorFlowInferenceInterface inferenceInterface;


    // image list indexing variable
    private int imageListIndex=9;
    // getting images of digits
    private int[] imageIDList = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9};


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // set up UI elements
        imageView =findViewById(R.id.image_view);
        textView = findViewById(R.id.text_view);

        // initialize inference variables to use our model
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(),MODEL_FILE);
    }

    private float[] convertImage() {
        // Convert current image to a scaled 28 x 28 bitmap
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(),
                imageIDList[imageListIndex]);

        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 28, 28, true);
        imageView.setImageBitmap(imageBitmap);

        int[] imageAsIntArray = new int[784];
        float[] imageAsFloatArray = new float[784];

        // Get the pixel values of the bitmap and store them in a flattened int array
        imageBitmap.getPixels(imageAsIntArray, 0, 28, 0, 0, 28, 28);
        // Convert the int array into a float array
        for (int i = 0; i < 784; i++) {
            imageAsFloatArray[i] = imageAsIntArray[i] / -16777216;
        }
        return imageAsFloatArray;
    }
    // function to call when user presses predict button
    public void predictDigitClick(View view){
        float[] pixelbuffer = convertImage();

        float[] results = formPrediction(pixelbuffer);
        //for (float result : results){
        //    Log.d("result", String.valueOf(result));
        //}
        printResults(results);

    }


    private void printResults(float[] results){
        // finding 2 max. probability values to predict results
        float maxVal = 0;
        float secondMaxVal = 0;
        int maxValIdx = 0;
        int secondMaxValIdx = 0;

        for (int i=0; i<10; i++){
            if (results[i] > maxVal){
                secondMaxVal = maxVal;
                secondMaxValIdx = maxValIdx;
                maxVal = results[i];
                maxValIdx = i;
            }
            else if (results[i] < maxVal && results[i] > secondMaxVal){
                secondMaxVal = results[i];
                secondMaxValIdx = i;
            }
        }

        String output = "Model Prediction: " + String.valueOf(maxValIdx)  +
                ", Second Prediction: " + String.valueOf(secondMaxValIdx);
        textView.setText(output);
    }

    private float[] formPrediction(float[] pixelBuffer) {
        // Fill the input node with the pixel buffer
        inferenceInterface.fillNodeFloat(INPUT_NODE,INPUT_SHAPE,pixelBuffer);

        // Make the prediction by running inference on our model and store results in output node
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] results = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // Store value of output node (results) into a float array
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return results;
    }

    // flattering images
    // function converts images into float array
    // returns a float array for input into model


    // function to call when user presses next image button
    public void loadNextImageClick(View view){
        if (imageListIndex >=9){
            imageListIndex =0;
        }
        else{
            imageListIndex +=1;
        }
        //Log.d("Next Image", String.valueOf(imageListIndex));
        imageView.setImageDrawable(getDrawable(imageIDList[imageListIndex]));  // view image
    }
}
