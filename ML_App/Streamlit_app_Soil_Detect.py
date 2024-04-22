# Imports
import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from Functions import get_class_names, get_percentiles_distance, generate_perc_dist_columns, get_pixel_sum_v2, preprocess

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
    st.write('Coming Soon...')

tr_data = pd.read_csv('Data/Training-Data.csv')

st.write('Model is predicting...')
ada=AdaBoostClassifier(learning_rate=1, n_estimators=650)
ada.fit(tr_data.drop('class', axis=1), tr_data[['class']])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pre = preprocess(img, blend_x=1, blend_y=0, canny_x=50, canny_y=400)
    names=generate_perc_dist_columns(perc_distance=1)
    df_test = pd.DataFrame(columns=names)
    naive_particle_sum=get_pixel_sum_v2(pre, 5)
    percentiles=get_percentiles_distance(naive_particle_sum, perc_distance=1)
    df_test.loc[len(df_test)] = percentiles
    prd= ada.predict(df_test)
    #st.dataframe(df_test)

predict_on_img = get_class_names(prd)
im_to_display=cv.resize(img, (300,300))
st.image(im_to_display)
st.write('Image class is', predict_on_img)


