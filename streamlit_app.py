import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import subprocess
import os
import urllib.request


# Sequential model (Conv2D + MaxPool)
# MODEL1 = tf.keras.models.load_model("baseline_resnet50.h5", compile=False)
# MODEL1 = tf.keras.models.load_model("Omdena_model1.h5", compile=False)
# MODEL1 = tf.keras.models.load_model('model1.h5', compile=False)
# MODEL1 = load_model()

MODEL1 = tf.keras.models.load_model("model_CNN1_BRACOL.h5", compile=False)
# MODEL1 = tf.keras.models.load_model("withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5", compile=False)

# Mobilenet-v2 
MODEL4 = tf.keras.models.load_model("Omdena_model4.h5", compile=False)
# MODEL4 = tf.keras.models.load_model("model_CNN1_BRACOL.h5", compile=False)

CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

#Function to get prediction array for a model (used in ensembling)
def get_all_predictions(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    return predictions[0]

#Function to get final predictions
def predict():
    image = None
    upload_file = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])
    upload_camera = st.camera_input("Or take a picture here...")
    
    if upload_file is not None:
        image = Image.open(upload_file)
        
    if upload_camera is not None:
        image = Image.open(upload_camera)
        
    if image is not None:
        # image = Image.open(upload)
        # image = image.crop((left, top, right, bottom))
        st.image(image)
        # newsize1 = (224, 224)
        newsize1 = (256, 256)
        newsize4 = (256, 256)
        image1 = image.resize(newsize1)
        image4 = image.resize(newsize4)
        image1 = np.asarray(image1)
        image4 = np.asarray(image4)
        img_batch1 = np.expand_dims(image1, 0)
        img_batch4 = np.expand_dims(image4, 0)

        #Get model predictions
        predictions1 = MODEL1.predict(img_batch1)
        predictions4 = MODEL4.predict(img_batch4)
        
        #Get model predictions for ensemble output
        all_predictions1 = get_all_predictions(MODEL1, image1)
        all_predictions4 = get_all_predictions(MODEL4, image4)
        # all_predictions_ensemble = (all_predictions1 + all_predictions4)/2

        #Get final prediction
        predicted_class1 = CLASS_NAMES[np.argmax(predictions1[0])]
        confidence1 = np.max(predictions1[0])

        predicted_class4 = CLASS_NAMES[np.argmax(predictions4[0])]
        confidence4 = np.max(predictions4[0])

        #Get final prediction for ensemble
        # predicted_class_ensemble = CLASS_NAMES[np.argmax(all_predictions_ensemble[0])]
        predicted_class_ensemble = None
        confidence_ensemble = None
        
        return {"class1": predicted_class1, "confidence1": float(confidence1), "class4": predicted_class4, "confidence4": float(confidence4), "class_ensemble": predicted_class_ensemble, "confidence_ensemble": confidence_ensemble}
    else:
        return {"class1": "No Image", "confidence1": 0, "class4": "No Image", "confidence4": "No Image", "class_ensemble": "No Image", "confidence_ensemble": "No Image"}

    
predicted_output = predict()
st.write("Model Predictions: ")
# st.write("Prediction from baseline CNN model (183877 parameters): ", predicted_output['class1'])
st.write("Prediction from Cusomized CNN (BRACOL symptoms): ", predicted_output['class1'])
st.write("Prediction from Mobilenet-v2 (2667589 parameters): ", predicted_output['class4'])
st.write("Prediction from Ensemble of Cusomized CNN (BRACOL symptoms) and mobilenet-v2 : ", predicted_output['class_ensemble'])


