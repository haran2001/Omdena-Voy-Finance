import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import subprocess
import os
import urllib.request

from tensorflow.keras.models import load_model
import gdown
from pathlib import Path

model_name = 'withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5'

save_dest = Path('models')
save_dest.mkdir(exist_ok=True)
output = f'models/{model_name}'

@st.cache_resource
def load_model_h5(path):
    return load_model(output, compile=False)

def load_model_from_gd():
    #f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1--eYkRRQl6CAuXxPFcgiFy0zdp67WTPE', output=output, quiet=False)

#Function to get prediction array for a model (used in ensembling)
def get_all_predictions(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    return predictions[0]

def get_class(image, newsize, MODEL):
    image = image.resize(newsize)
    image = np.asarray(image)
    img_batch = np.expand_dims(image, 0)

    #Get model predictions
    predictions = MODEL.predict(img_batch)
      
    #Get model predictions for ensemble output
    all_predictions = get_all_predictions(MODEL, image)

    #Get final prediction
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence

def get_class_ensemble(image, newsize, MODEL_A, MODEL_B):
    image = image.resize(newsize)
    image = np.asarray(image)
    img_batch = np.expand_dims(image, 0) 
      
    #Get model predictions for ensemble output
    all_predictions_A = get_all_predictions(MODEL_A, image)
    all_predictions_B = get_all_predictions(MODEL_B, image)
    all_predictions_ensemble = (all_predictions_A + all_predictions_B)/2

    #Get final prediction
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    #Get final prediction for ensemble
    predicted_class_ensemble = CLASS_NAMES[np.argmax(all_predictions_ensemble[0])]
    confidence_ensemble = None

    return predicted_class_ensemble, confidence_ensemble
    
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
        st.image(image)
        newsize = (256, 256)
        newsize1 = (256, 256)
        newsize3 = (224, 224)
        newsize4 = (256, 256)

        predicted_class1, confidence1 = get_class(image, newsize1, MODEL1)
        # predicted_class4, confidence4 = get_class(image, newsize4, MODEL4)
        predicted_class4, confidence4 = get_class(image, newsize3, MODEL3)
    
        predicted_class_ensemble, confidence_ensemble = get_class_ensemble(image, newsize, MODEL1, MODEL4)
        
        predicted_class_ensemble = None
        confidence_ensemble = None

        
        return {"class1": predicted_class1, "confidence1": float(confidence1), "class4": predicted_class4, "confidence4": float(confidence4), "class_ensemble": predicted_class_ensemble, "confidence_ensemble": confidence_ensemble}
    else:
        return {"class1": "No Image", "confidence1": 0, "class4": "No Image", "confidence4": "No Image", "class_ensemble": "No Image", "confidence_ensemble": "No Image"}




# All classes
CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

# Custom CNN    
MODEL1 = tf.keras.models.load_model("model_CNN1_BRACOL.h5", compile=False)

# Mobilenet-v2 
MODEL4 = tf.keras.models.load_model("Omdena_model4.h5", compile=False)

f_checkpoint = Path(f"models//{model_name}")
if not f_checkpoint.exists():
    load_model_from_gd()
else:
    MODEL3 = load_model_h5(f_checkpoint)
    
predicted_output = predict()
st.write("Model Predictions: ")
st.write("Prediction from Cusomized CNN (BRACOL symptoms): ", predicted_output['class1'])
st.write("Prediction from Mobilenet-v2 (2667589 parameters): ", predicted_output['class4'])
st.write("Prediction from Ensemble of Cusomized CNN (BRACOL symptoms) and mobilenet-v2 : ", predicted_output['class_ensemble'])


