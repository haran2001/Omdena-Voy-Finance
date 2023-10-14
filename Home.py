from helping_functions import load_model_h5, load_model_from_gd, predict, predict_ensemble, get_image, verify_checkpoint
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from keras import backend as K

st.set_page_config(page_title='Plants diseases', page_icon=':herb:', initial_sidebar_state='auto')
st.header('Omdena: Sao Paulo Chapter')
st.subheader('Plant Disease Classificattion')
st.write('The following application will help you identify the disease infecting a plant, if you provide an image os the leaf.')
st.write('You can provide an image in one of the following ways: ')
st.write('1. Upload an existing photo')
st.write('2. Take a photo through your camera')
st.write('Once done press the classifiy button to the results from each of our models')

# Custom CNN    
MODEL1 = load_model_h5('assets/models/model_CNN1_BRACOL.h5')

# Sequential CNN -v2 
MODEL2 = load_model_h5('assets/models/Omdena_model1.h5')

# Mobilenet-v2 
MODEL4 = load_model_h5('assets/models/Omdena_model4.h5')

# Resnet-v2
model3 = 'withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5'
f_checkpoint = Path(f"assets/models//{model3}")
if verify_checkpoint(model3, f_checkpoint):
    MODEL3 = load_model_h5(f_checkpoint)
    # MODEL3 = load_model(f_checkpoint, custom_objects={"K": K})
    # MODEL3 = load_model(f_checkpoint)


#Resize requirements
newsize  = (256, 256)
newsize1 = (256, 256)
newsize3 = (224, 224)
newsize4 = (256, 256)

# Get uploaded image
image = get_image()
classify_button = st.button("Classify", key='c_but', disabled=st.session_state.get("disabled", True))

st.write("Model Predictions: ")
if image is not None and classify_button:
    predicted_output1 = predict(image, newsize1, MODEL1)
    st.write("Cusomized CNN (BRACOL symptoms): ", predicted_output1['class'])

if image is not None and classify_button:
    predicted_output3 = predict(image, newsize3, MODEL3)
    st.write("Resnet-v2: ", predicted_output3['class'])
    
if image is not None and classify_button:
    predicted_output4 = predict(image, newsize4, MODEL4)
    st.write("Mobilenet-v2: ", predicted_output4['class'])

if image is not None and classify_button:
    predicted_output2 = predict_ensemble(image, newsize, MODEL2, MODEL4)
    st.write("Sequential CNN and Mobilenet-v2 (Ensemble model): ", predicted_output2['class'])
