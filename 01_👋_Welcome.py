import streamlit as st
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time

st.set_page_config(
    page_title="Banana Disease Detection Web Application",
    page_icon="🍌",
)

st.write("# 👨‍🌾 Welcome to Banana Disease Detection Web Application! 🍌")

st.sidebar.success("Select the classifier 🔝")


st.markdown(
    """
    This web application is developed to help farmers in the identification of 
    different diseases on Bananas _(Musa sp.)_. 

    **Here are the list of the common banana diseases that the web-app can detect:**
    - 🔸 Bunchy Top Disease
    - 🔸 Fusarium Wilt Disease
    - 🔸 Moko Bacterial Wilt Disease
    - 🔸 Sigatoka Disease
    
    --- and it can also identify, Healthy leaves! 🌱

"""
)
st.write("""#### To use the banana disease classifier:""")
st.success("👈👈👈 Kindly select the option 'Classifier' in the sidebar section 👈👈👈 ")

