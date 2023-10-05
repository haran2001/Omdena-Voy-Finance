import streamlit as st
from keras.applications.vgg16 import VGG16
import pickle
from skimage.transform import resize
import numpy as np
import cv2 as cv
import joblib
from tensorflow.keras.models import load_model
# import keras_model as km
import sklearn
# import scikeras

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