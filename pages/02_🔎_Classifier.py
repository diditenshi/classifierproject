from typing import Any
import streamlit as st
import PIL
from PIL import Image
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import preprocessing, models, layers, utils
from tensorflow.keras.models import *
from tensorflow.keras import *
import time

## this is part of web app

## ----------------------------------------------- x -----------------------------------------x-------------------------x------------------##
# fig = plt.figure()
st.set_page_config(
    page_title="Banana Disease Detection Web Application",
    page_icon="🍌",
)

st.title('👨‍🌾 Banana Disease Classifier 🍌')

st.markdown("Prediction of the Banana Diseases: Bunchy Top, Moko, Sigatoka and Fusarium Wilt")

def main():
    file_uploaded = st.file_uploader("Select an image of the banana plant leaf", type=["png","jpg","jpeg"])
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_btn = st.button("Start the Classification")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")

                predictions = predict(image)

                time.sleep(1)
                st.success('The image has been successfully classified.')
                st.markdown("""**Classification Results:**""")
                st.write(predictions)

                prev_con = view_control()
                st.markdown("""**Prevention and Control Measures:**""")
                st.write(prev_con)
               

### This code is for saved model in H5 file format.

control = ''
def predict(image):
    classifier_model = "pages\model2.h5"

    model = load_model(classifier_model)

    #test_image = tf.keras.utils.load_img(image, target_size=(224,224))
    test_image = image.resize((224,224))    
    img_array = tf.keras.utils.img_to_array(test_image)
    img_array = tf.expand_dims(img_array,0)
    #test_image = preprocessing.image.img_to_array(test_image)
    #test_image = test_image / 255.0
    #test_image = np.expand_dims(test_image, axis=0)
    #class_names = {0: 'Healthy', 1: 'Bunchy Top Disease', 2: 'Fusarium Wilt', 3: 'Moko (Bacterial Wilt) Disease', 4: 'Sigatoka Disease'}
    
    class_names = ['BUNCHY TOP DISEASE', 'FUSARIUM WILT DISEASE', 'HEALTHY PLANT', 'MOKO (BACTERIAL WILT) DISEASE', 'SIGATOKA DISEASE']

    prevent0 = '''Prevention and control measures include the conduct of early detection survey of initial 
                symptoms, eradicate promptly and enforce strict quarantine measures to prevent the movement
                of virus-infected materials into new areas. Other way is to spray all the plants, including
                all the ground and grasses, within a 6-m radius from the infected plant then excavate all the
                infected plant including all the suckers and chop the damaged parts into small pieces. The
                chopped parts must be carefully piled on top of the leaves, with the corm at the topmost position,
                to prevent the re-growth and enhance drying. Re-spray the chopped and piled-up plant parts and when
                replanting, use bunchy-free planting materials. The replanting must be done three or more days after
                eradication. However, he use of Glyphosate-impregnated sticks is also a faster, easier and more 
                economical method of eradicating bunchy-top infected than the manual way.'''
    prevent1 = '''Implement strict quarantine measures to prevent the transfer of diseased planting materials
                into new areas since economical method to eliminate the fungus from an infested soil is
                not yet available. Infected banana plants, including those within a 6-m radius, must be 
                immediately eradicated to minimize the spread of the disease.'''
    prevent2 = '''The banana plant is healthy. Proper care and monitoring is required to prevent getting infected with 
                diseases.'''
    prevent3 = '''Early detection and immediate eradication of the infected plant is needed to prevent the
                spread of this disease. Remove and chop the plants surrounding the infected mat within a
                radius of 6 meters from the infected plant. As for the Moko-infected mat, excavate it, chop
                it into smaller pieces and burn them to ashes using rice hull, old bamboo props or dried
                saw dust. Fallow the area after spraying the chopped plant debris and soil with formalin. 
                Keep the area from any weeds by spraying with Round-up (Glyphosate). Disinfect used tools
                with 10 percent formaldehyde solution. Enforce stringent plant quarantine and phytosanitary
                measures. Replant the quarantined area after one (1) year with disease-free, tissue-cultured
                plantlets. '''
    prevent4 = '''You must apply contact fungicides to kill fungal spores on the leaf surface and systemic 
                fungicides to kill fungal growth inside the leaf. Use banana oil to facilitate the entry of
                systemic fungicides into the leaf, inhibit fungal growth by itself, and improve the sticking
                ability of contact fungicides.'''

    class_prevention_options = [prevent0, prevent1, prevent2, prevent3, prevent4]

    #predictions = model.predict(test_image)
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    global control
    prevention = f"{class_prevention_options[np.argmax(scores)]}"
    control = prevention

    result = f"This image is most likely belongs to {class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} % confidence. "
    return result
  
## -----------------------------------------------------x---------------------------------------x--------------------------------------------##

def view_control():
    return control

def get_control():
    view_control()


if __name__ == "__main__":
    main()


