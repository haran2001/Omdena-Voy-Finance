import streamlit as st
from keras.applications.vgg16 import VGG16
import pickle
from skimage.transform import resize
import numpy as np
import cv2 as cv
import joblib
from tensorflow.keras.models import load_model
import sklearn

from io import BytesIO
from PIL import Image
import tensorflow as tf
import subprocess
import os
import urllib.request
import gdown
from pathlib import Path


# Classes
CLASS_NAMES = ['Cescospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

@st.cache_resource
def load_model_h5(path):
    return load_model(path, compile=False)

def verify_checkpoint(model_name, f_checkpoint):
    if not f_checkpoint.exists():
        load_model_from_gd(model_name)
    return f_checkpoint.exists()

def load_model_from_gd(model_name):
    save_dest = Path('models')
    save_dest.mkdir(exist_ok=True)
    output = f'assets/models/{model_name}'
    # f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1--eYkRRQl6CAuXxPFcgiFy0zdp67WTPE', output=output, quiet=False)
        # gdown.download(f"https://drive.google.com/uc?id=1klOgwmAUsjkVtTwMi9Cqyheednf_U18n", output)

def get_class(image, newsize, MODEL):
    image = image.resize(newsize)
    image = np.asarray(image)
    img_batch = np.expand_dims(image, 0)

    #Get model predictions
    predictions = MODEL.predict(img_batch)
      
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

    #Get final prediction for ensemble
    predicted_class_ensemble = CLASS_NAMES[np.argmax(all_predictions_ensemble[0])]
    confidence_ensemble = np.max(all_predictions_ensemble[0])

    return predicted_class_ensemble, confidence_ensemble

#Wait until image is uplaoded and obtain image
def get_image():
    image = None
    st.subheader('Upload image here:')
    upload_file = st.file_uploader(type=['png', 'jpeg', 'jpg'])
    st.subheader('Take a photo here:')
    upload_camera = st.camera_input()
    
    if upload_file is not None:
        image = Image.open(upload_file)
        
    if upload_camera is not None:
        image = Image.open(upload_camera)
        
    if image is not None:
        st.image(image)
    
    return image
        
# Function to get final predictions
def predict(image, size, MODEL):
    predicted_class, confidence = get_class(image, size, MODEL)
    return {"class": predicted_class, "confidence": float(confidence)}

# Function to get final predictions from ensemble
def predict_ensemble(image, size, MODEL_A, MODEL_B):
    predicted_class, confidence = get_class_ensemble(image, size, MODEL_A, MODEL_B)
    return {"class": predicted_class, "confidence": float(confidence)}


def load_css():
    css_file = open('assets/style.css', 'r')
    st.markdown(f'<style>{css_file.read()}</style>', unsafe_allow_html=True)
    css_file.close()

path_pipe = 'assets/models/nn_pca_3_pipeline.sav'
path_keras = 'assets/models/nn_pca_3_keras.h5'
step = 'clf'

@st.cache_resource
def load_models():
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(125,125,3))
    for layer in VGG_model.layers:
        layer.trainable=False  
    #load the pipeline
    model_l = joblib.load(path_pipe)
    #load the keras classifier
    model_l.named_steps[step].model_ = load_model(path_keras)
    return VGG_model, model_l


def crop(img_arr):
    """
    Function for cropping images.
    Input: Images array.
    Returns: Cropped and Resized Image array.
    """
    gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1] # threshold 
    hh, ww = thresh.shape
    thresh[hh-3:hh, 0:ww] = 0 # make bottom 2 rows black where they are white the full width of the image
    white = np.where(thresh==255) # get bounds of white pixels
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])       
    crop = img_arr[ymin:ymax+3, xmin:xmax] # crop the image at the bounds adding back the two blackened rows at the bottom
    resized_img = resize(crop, (125, 125), anti_aliasing=True)
    return resized_img
