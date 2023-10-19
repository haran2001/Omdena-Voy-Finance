from helping_functions import load_model_h5, load_model_from_gd, predict, predict_ensemble, get_image, verify_checkpoint, loadRefImages, buildPredictions, load_model_pth, get_class_pytorch
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from keras import backend as K
import torch.nn as nn

st.set_page_config(page_title='Coffee Plant Classification', page_icon=':herb:', initial_sidebar_state='auto')
st.header('Omdena: Sao Paulo Chapter', anchor=False)
st.subheader('Coffee Plant Disease Classification', anchor=False)
st.write('The following application will help you identify the disease infecting a plant, if you provide an image os the leaf.')
st.write('You can provide an image in one of the following ways: ')
st.write('1. Upload an existing photo')
st.write('2. Take a photo through your camera')
st.write('Once done press the classifiy button to the results from each of our models')
st.write('Note: Make sure your image has 3 RGB channels. Do not use images that also have a transparancy channel.')

# PyTorch Model
class CoffeeLeafClassifier(nn.Module):
    def __init__(self):
        super(CoffeeLeafClassifier, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 30 * 30, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 5) # 5 classes
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output
        x = self.fc_layers(x)
        return x

# Custom CNN    
MODEL1 = load_model_h5('assets/models/model_CNN1_BRACOL.h5')

# Sequential CNN -v2 
MODEL2 = load_model_h5('assets/models/Omdena_model1.h5')

# Mobilenet-v2 
MODEL4 = load_model_h5('assets/models/Omdena_model4.h5')

# Resnet-v2
model3 = 'withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5'
f_checkpoint = Path(f"assets/models//{model3}")
if verify_checkpoint(model3, f_checkpoint, '1--eYkRRQl6CAuXxPFcgiFy0zdp67WTPE'):
    MODEL3 = load_model_h5(f_checkpoint)
    # MODEL3 = load_model(f_checkpoint, custom_objects={"K": K})
    # MODEL3 = load_model(f_checkpoint)

# Siamese Network
model5 = 'Model_Siamese_5class_h5file.h5'
f_checkpoint = Path(f"assets/models//{model5}")
if verify_checkpoint(model5, f_checkpoint, '1klOgwmAUsjkVtTwMi9Cqyheednf_U18n'):
    MODEL5 = load_model_h5(f_checkpoint)
    
# CNN Pytorch model
model6 = 'coffee_leaf_classifier.pth'
f_checkpoint = Path(f"assets/models//{model6}")
if verify_checkpoint(model6, f_checkpoint, '1XroFNNq4FD8zE3DXDfBPj8NbD7cqYaaf'):
    MODEL6 = load_model_pth(f_checkpoint)

## Load ReferenceImages
refImages = loadRefImages()

#Resize requirements
newsize  = (256, 256)
newsize1 = (256, 256)
newsize3 = (224, 224)
newsize4 = (256, 256)
newsize5 = (256, 256)

# Get uploaded image
state = False
image = get_image()
classify_button = st.button("Classify", key='c_but', disabled=st.session_state.get("disabled", True))

st.write("Model Predictions: ")
if classify_button:
    predicted_output1 = predict(image, newsize1, MODEL1)
    st.write("Cusomized CNN (BRACOL symptoms): ", predicted_output1['class'])

if classify_button:
    predicted_output3 = predict(image, newsize3, MODEL3)
    st.write("Resnet50 deduplicated: ", predicted_output3['class'])
    
# if classify_button:
#     predicted_output4 = predict(image, newsize4, MODEL4)
#     st.write("Mobilenet-v2: ", predicted_output4['class'])

if classify_button:
    predicted_output5 = buildPredictions([image], refImages, newsize5, MODEL5, 3)
    st.write("Siamese Network: ", predicted_output5['class'])

if classify_button:
    predicted_output2 = predict_ensemble(image, newsize, MODEL2, MODEL4)
    st.write("Sequential CNN and Mobilenet-v2 (Ensemble model): ", predicted_output2['class'])
    
if classify_button:
    predicted_output6 = get_class_pytorch(image, MODEL6)
    st.write("PyTorch CNN: ", predicted_output6)
