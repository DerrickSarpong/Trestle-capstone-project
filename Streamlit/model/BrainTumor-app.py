import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# st.sidebar.header('User Input ')

# Load your trained machine learning model
# Replace this with your model loading code
model = tf.keras.models.load_model("CancerVGG2.h5")


# Define a function to preprocess the image for prediction
def preprocess_image(image):
    # Resize the image to match the model's input shape
    image = image.resize((128, 128))
    
    if image.mode == "RGB":
        # If it's an RGB image, convert to a NumPy array and normalize
        image = np.array(image) / 255.0
    else:
        # If it's grayscale, convert to RGB by duplicating the channel
        image = np.array(image.convert("RGB")) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


# Streamlit app
st.title("Brain Tumor AI")
st.header("Classify brain tumors")

# Upload image through the Streamlit interface
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    if st.button("Classify"):
        image = Image.open(uploaded_image)
        image = preprocess_image(image)

        # Perform inference using your trained model
        prediction = model.predict(image)

        # Display the prediction result
        st.subheader('Prediction')
        labels = ['Glioma', 'Meningioma','No tumor', 'Pituitary']

        predicted_class_index = np.argmax(prediction)
        predicted_class_label = labels[predicted_class_index]
        if predicted_class_label == 'Glioma':
          st.write("This brain tumor is of type ",predicted_class_label)
    
        elif predicted_class_label == 'Meningioma':
          st.write("This brain tumor is of type ",predicted_class_label)

        elif predicted_class_label == 'Pituitary':
          st.write("This brain tumor is of type",predicted_class_label)
        
        else:
          st.write("There is no visible tumor ",predicted_class_label)
          #st.write(prediction)

        st.subheader('Prediction Probability')
        st.write(prediction)

