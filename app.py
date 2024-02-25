import streamlit as st
import numpy as np
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pre-trained model
loaded_model = load_model("cnn.h5")

# Function to make prediction
def get_predict(path):
    try:
        test_image = image.load_img(path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        labels = ['Classical style comic strip', 'Manga style comic strip']
        result = loaded_model.predict(test_image)
        prediction = labels[np.argmax(result)]
        pil_im = Image.open(path)
        im_array = np.asarray(pil_im)
        
        # Display the image with the predicted label
        st.image(im_array, caption=f"Uploaded Image - Predicted Label: {prediction}", use_column_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

def main():
    st.title("Comic Style Predictor")
    st.sidebar.title("Upload Image")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            st.write("Classifying...")

            # Make prediction using the uploaded image
            get_predict(uploaded_file)

        except Exception as e:
            st.error(f"Error in Streamlit app: {e}")

if __name__ == "__main__":
    main()
