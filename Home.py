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

@st.cache_resource
def load_model_h5(path):
    return load_model(path, compile=False)

def load_model_from_gd(model_name):
    save_dest = Path('models')
    save_dest.mkdir(exist_ok=True)
    output = f'models/{model_name}'
    # f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1--eYkRRQl6CAuXxPFcgiFy0zdp67WTPE', output=output, quiet=False)

def get_class(image, newsize, MODEL):
    image = image.resize(newsize)
    image = np.asarray(image)
    img_batch = np.expand_dims(image, 0)

    #Get model predictions
    predictions = MODEL.predict(img_batch)
      
    #Get model predictions for ensemble output
    # all_predictions = get_all_predictions(MODEL, image)

    #Get final prediction
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence
    
#Function to get prediction array for a model (used in ensembling)
def get_all_predictions(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    return predictions[0]

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

def get_image():
    image = None
    upload_file = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])
    upload_camera = st.camera_input("Or take a picture here...")
    
    if upload_file is not None:
        image = Image.open(upload_file)
        
    if upload_camera is not None:
        image = Image.open(upload_camera)
        
    if image is not None:
        st.image(image)
    
    return image
        
#Function to get final predictions
def predict(image, size, MODEL):
    # image = None
    # upload_file = st.file_uploader("Upload your image here...", type=['png', 'jpeg', 'jpg'])
    # upload_camera = st.camera_input("Or take a picture here...")
    
    # if upload_file is not None:
    #     image = Image.open(upload_file)
        
    # if upload_camera is not None:
    #     image = Image.open(upload_camera)
        
    # if image is not None:
    #     st.image(image)
    predicted_class, confidence = get_class(image, size, MODEL)
        # predicted_class_ensemble, confidence_ensemble = get_class_ensemble(image, newsize, MODEL1, MODEL4)
        
        # predicted_class_ensemble = None
        # confidence_ensemble = None

        
    return {"class": predicted_class, "confidence": float(confidence)}


# All classes
CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

# Custom CNN    
MODEL1 = load_model_h5('model_CNN1_BRACOL.h5')
# Mobilenet-v2 
MODEL4 = load_model_h5('Omdena_model4.h5')

# Resnet-v2
model_name = 'withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5'
f_checkpoint = Path(f"models//{model_name}")
if not f_checkpoint.exists():
    load_model_from_gd()
# else:
MODEL3 = load_model_h5(f_checkpoint)

newsize  = (256, 256)
newsize1 = (256, 256)
newsize3 = (224, 224)
newsize4 = (256, 256)

# Get uploaded image
image = get_image()

st.write("Model Predictions: ")
if image is not None:
    predicted_output1 = predict(newsize1, MODEL1)
    st.write("Prediction from Cusomized CNN (BRACOL symptoms): ", predicted_output1['class'])

if image is not None:
    predicted_output3 = predict(image, newsize3, MODEL3)
    st.write("Prediction from Resnet-v2: ", predicted_output3['class'])
    
if image is not None:
    predicted_output4 = predict(newsize4, MODEL4)
    st.write("Prediction from Mobilenet-v2: ", predicted_output4['class'])

# st.write("Prediction from Ensemble of Cusomized CNN (BRACOL symptoms) and mobilenet-v2 : ", predicted_output['class_ensemble'])
