# Streamlit App
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model once at startup
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('D:\Brain\ResNet50_model.h5')  # Update path if needed
    return model

model = load_model()

# Define tumor class names consistent with your model's training labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Function to preprocess image for prediction
def preprocess_image(image: Image.Image):
    # Resize to model's expected input size
    img = image.resize((224, 224))
    # Convert to numpy array and scale pixel values to [0,1]
    img_array = np.array(img) / 255.0
    # Ensure shape (224,224,3)
    if img_array.shape[-1] == 4:  # Remove alpha channel if exists
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim
    return img_array

# Function to predict class and confidence
def predict(image: Image.Image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]  # Get first (and only) batch's prediction
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    predicted_class = class_names[class_idx]
    return predicted_class, confidence

# Streamlit UI
st.title("Brain Tumor MRI Classification")
st.write("""
Upload a brain MRI image, and the app will classify the tumor type with confidence score.
""")

uploaded_file = st.file_uploader("Choose an MRI image (JPG/PNG format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Brain MRI", use_column_width=True)

    st.write("Classifying...")
    label, conf = predict(image)

    st.markdown(f"### Prediction: **{label.capitalize()}**")
    st.markdown(f"### Confidence: **{conf * 100:.2f}%**")

    # Optionally, display all class probabilities as a bar chart
    processed_img = preprocess_image(image)
    preds = model.predict(processed_img)[0]
    # Create dict of class and confidence
    scores = {cls: float(pred) for cls, pred in zip(class_names, preds)}
    st.bar_chart(scores)

else:
    st.write("Please upload a brain MRI image to start classification.")
