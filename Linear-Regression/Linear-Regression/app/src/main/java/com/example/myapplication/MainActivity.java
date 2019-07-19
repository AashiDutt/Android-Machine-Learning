package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    static{
        System.loadLibrary("tensorflow_inference");
    }
    private static final String MODEL_NAME ="file:///android_asset/optimized_frozen_model.pb";
    private static final String INPUT_NODE ="input_features";
    private static final String OUTPUT_NODE ="y_out";
    private static final int[] INPUT_SHAPE ={1,1};
    private TensorFlowInferenceInterface inferenceInterface;

    EditText editText;
    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editText = (EditText)findViewById(R.id.edit_text);
        textView=(TextView)findViewById(R.id.text_view);
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_NAME);
    }

    public void pressButton(View view){
        float input =Float.parseFloat(editText.getText().toString());
        Log.d("MainActivity",String.valueOf(input));
        String results = performInference(input);
        textView.setText(results);

    }
    private String performInference(float input){
        float[] floatArray ={input};
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE,floatArray);
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] results ={0.0f};
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return String.valueOf(results[0]);

    }
}
