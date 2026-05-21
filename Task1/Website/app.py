import streamlit as st
import pickle
import numpy as np

# Set the page configuration for a clean, professional look
st.set_page_config(page_title="California House Price Predictor", page_icon="🏡", layout="centered")

# Header Section
st.title("🏡 California House Price Predictor")
st.markdown("This application predicts the median house value in California districts based on various housing metrics. Developed for **Task 1: AI & ML Internship**.")

st.divider()

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Error: 'model.pkl' not found. Please ensure your saved model file is in the same folder as this script.")
else:
    st.subheader("Enter Housing Details:")
    
    # Create two columns for a neat form layout
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.number_input("Median Income (in $10k)", min_value=0.0, max_value=15.0, value=5.0, step=0.1, help="Median income in block group")
        house_age = st.number_input("Housing Median Age", min_value=1.0, max_value=100.0, value=20.0, step=1.0, help="Median age of a house within a block")
        ave_rooms = st.number_input("Average Rooms", min_value=1.0, max_value=150.0, value=5.0, step=0.5, help="Average number of rooms within a block")
        ave_bedrms = st.number_input("Average Bedrooms", min_value=1.0, max_value=50.0, value=1.0, step=0.5, help="Average number of bedrooms within a block")
        
    with col2:
        population = st.number_input("Population", min_value=1.0, max_value=40000.0, value=1000.0, step=10.0, help="Total number of people residing within a block")
        ave_occup = st.number_input("Average Occupancy", min_value=1.0, max_value=1500.0, value=3.0, step=0.5, help="Average number of household members")
        latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.0, step=0.1, help="A measure of how far north a house is")
        longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.0, step=0.1, help="A measure of how far west a house is")

    st.divider()
    
    # Prediction Button
    if st.button("Predict House Price", type="primary", use_container_width=True):
        # Format inputs exactly as the model expects (2D array matching the 8 features)
        features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
        
        with st.spinner("Calculating prediction..."):
            # Predict using the loaded model
            prediction = model.predict(features)
            
            # The California housing target in scikit-learn is expressed in hundreds of thousands of dollars ($100,000)
            predicted_price = prediction[0] * 100000 
            
        st.success(f"### Estimated Median House Value: **${predicted_price:,.2f}**")
        st.caption("Note: This prediction is generated purely by the trained Linear Regression model.")
