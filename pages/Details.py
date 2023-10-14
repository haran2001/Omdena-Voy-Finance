import streamlit as st
from PIL import Image
from helping_functions import load_css

st.set_page_config(layout='wide')
load_css()

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

IMAGE_SIZE = (256, 256)

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Siamese network (MobileNetV2 architecture)")
    flowchart = Image.open('assets/Images/image6.png').resize(IMAGE_SIZE)
    st.image(flowchart)
with col2:
    st.subheader("Steps")
    st.write("The Siamese Network architecture is used for the coffee disease classification using images. \
    The main goal for using this network is to build an embedding space where the different classes are separated into distinct clusters. \
    The siamese network architecture is used to generate feature embeddings such that intra-class image features are close to each other in the embedding space, \
    whereas the inter-class image features are further apart from each other. \
    The motivation behind this setup is to allow diverse input images that are semantically similar to be close to each other. \
    In the coffee dataset, we have single leaf images indoors on a white background under different lighting conditions, \
    as well as images outdoors in natural light on the plant. \
    Siamese networks capture this diversity while still keeping the features close to each other in the embedding space. \
    To measure the distance or closeness in embedding space, we use similarity metrics like cosine similarity, euclidean distance, and dot product. \
    For the coffee disease classification problem the euclidean distance measures the closeness between image pairs, which in technical terms is also called the similarity metric. \
    The advantage of using this method is 2-fold: \
    One can use a simple K-nearest neighbour approach to classify the input image. \
    The setup can visually explain why the model chose the particular class. \
    The following K-NN strategy is used for inference: \
    ")
    
