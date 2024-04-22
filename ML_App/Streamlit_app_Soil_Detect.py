# Imports
import streamlit as st
import cv2
import numpy as np
from Functions import Pipeline

st.set_page_config(
    page_title="Soil-Detect",
    page_icon="🦾",
)

st.title(""":blue[Soil Detect]""")
st.write('Welcome to Soil Detect, a product of University of Nigeria, Nsukka, Civil Engineering Laboratory.\
         We perform soil classification based on the Unified Soil Classification System (USCS)')

#st.markdown('**To use simply upload the image of the soil you want to classify.**')

#st.sidebar("Background Study")
#with st.sidebar:
#   st.write("About the developers.")

# Loading image
file = st.file_uploader("Upload an image to Predict it's class", type=["jpg", "png", "jpeg"])

with st.expander('Expand to Know more about the Image Preprocessing Steps Used.'):
    st.write('We\'re awesome')

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.write('Model is predicting...')
    seg_img = Pipeline(img)
    st.image(seg_img)




