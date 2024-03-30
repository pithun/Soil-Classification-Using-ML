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

with st.expander('Expand for Details on Image Prepeocessing Steps Uses.'):
    st.write('We\'re awesome')

#st.sidebar("Background Study")
#with st.sidebar:
#   st.write("About the developers.")

# Loading image
file = st.file_uploader("Upload an image to Predict it's class", type=["jpg", "png", "jpeg"])

if file is not None and option_file_type == 'Image':
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.write('Model is predicting...')
    seg_img = Pipeline(img)
    st.image(seg_img)




