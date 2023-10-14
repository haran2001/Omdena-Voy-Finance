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

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Siamese network (MobileNetV2 architecture)")
    flowchart = Image.open('assets/images/image6.png').resize((128, 128))
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
    
st.subheader("Data Imbalance")
col3, col4 = st.columns([1,2])
with col3:
    st.write("\n")
    st.write("- The bar graph shows a significant class imbalance, with the ALL class having approximately 7200 images \
        and the Hem class having only around 3200 images.")
    st.write("- We created new instances for the minority class using image augmentation but even then the classifiers had \
        difficulty classifying the Hem class images.")
    st.write("- This issue affected the performance of all the models that we've trained but after extensive expermentations \
        we were able to mitigate its affect as much as we could.")
with col4:
    data_imbalance = Image.open('assets/images/data_imbalance.png')
    st.image(data_imbalance)

st.subheader("Confusion Matrix Comparison")
col1,col2,col3 = st.columns(3,gap='small')
with col1:
    before_aug = Image.open('assets/images/before_aug.png')
    st.image(before_aug,width=280)
    st.write('This is the confusion matrix of a model that we trained on the data before performing image augmentation')
with col2:
    after_aug = Image.open('assets/images/after_aug.png')
    st.image(after_aug, width=290)
    st.write("This is the confusion matrix after image augmentation, as you can see the performance of the model predicting \
        the hem class has improved")
with col3:
    after_voting = Image.open('assets/images/after_voting.png')
    st.image(after_voting, width=305)
    st.write("This is the cofusion matrix of a voting classifier, as you can see the voting classifier improved the \
        performance of the model predicting the minority class even more")

with col1:
    st.write("\n")
    st.write("\n")
    st.write("\n")
    simple_nn = Image.open('assets/images/simple_nn.png')
    st.image(simple_nn)

with col2:
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("This is the confusion matrix of a simple neural network that was trained on the data. It also shows \
        how using even a simple neural network outperforms the traditional machine learning models")
    
st.subheader("Experimentations")
experiments = Image.open("assets/images/experimentations.png")
st.image(experiments)
st.header("Challenges faced during the project")

st.write("Challenges we faced while working on the project:")
st.markdown("- Imbalanced dataset")
st.markdown("- Computing resources")
st.markdown("- Too many possible ways of approaching the solution which requried a lot of experimentations")
st.markdown("- Use of traditional machine learning algorithms instead of neural networks")


st.header("Future Work")
st.markdown("- Fine Tuning the pre-trained networks for feature extraction")
st.markdown("- We only had enough time to use VGG16 and ResNet for feature extraction, but there are \
    many other pre-trained networks that could be used which may give better results")
