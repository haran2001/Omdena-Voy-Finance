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
    st.subheader("Flowchart")
    flowchart = Image.open('assets/images/flowchart.png')
    st.image(flowchart)
with col2:
    st.subheader("Steps")
    st.write("**Image Augmentation:** Existing images of the minority class are augmented to create new images, \
        increasing the dataset size.")
    st.write("**Cropping Images:** All images are cropped to remove irrelevant pixels and reduce noise.")
    st.write("**Transfer Learning:** Preprocessed images are passed through a pre-trained neural network to extract features.")
    st.write("**Classification:** The extracted features are stored in a pandas dataframe and used to train different \
        classifiers for image classification.")
    
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
