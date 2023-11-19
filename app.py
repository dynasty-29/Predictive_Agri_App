# Import necessary libraries
import streamlit as st
import requests

# Streamlit app title
st.title("Crop Type Prediction App")

# Streamlit form for environmental data
st.subheader("Enter Environmental Data:")
temperature = st.slider("Temperature (°C)", 15.0, 39.0, 25.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
precipitation = st.slider("Precipitation", 0.0, 100.0, 50.0)
wind_speed = st.slider("Wind Speed", 0.0, 20.0, 10.0)
solar_radiation = st.slider("Solar Radiation", 0.0, 2000.0, 800.0)

# Streamlit form for optional soil nutrient levels
st.subheader("Enter Optional Soil Nutrient Levels (Leave blank if not available):")
nitrogen_level = st.slider("Nitrogen Level", 0, 100, 50)
phosphorus_level = st.slider("Phosphorus Level", 0, 100, 50)
potassium_level = st.slider("Potassium Level", 0, 100, 50)
ph_level = st.slider("Soil pH", 0.0, 14.0, 7.0)

# Button to trigger prediction
if st.button("Predict Crop"):
    # Prepare input data for prediction
    input_data = {
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "wind_speed": wind_speed,
        "solar_radiation": solar_radiation,
        "nitrogen_level": nitrogen_level,
        "phosphorus_level": phosphorus_level,
        "potassium_level": potassium_level,
        "ph_level": ph_level,
    }

    # Send a POST request to your Django API for prediction
    api_url = "your_django_api_url"  # Replace with your actual Django API URL
    response = requests.post(api_url, data=input_data)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.success(f"Predicted Crop: {prediction}")
        
        # Additional recommendation based on soil nutrient levels
        additional_recommendation = response.json().get("additional_recommendation")
        if additional_recommendation:
            st.subheader("Additional Recommendation based on Soil Nutrient Levels:")
            st.info(additional_recommendation)
    else:
        st.error("Failed to get prediction. Please try again.")

# Streamlit app footer
st.text("© 2023 Crop Prediction App. All rights reserved.")
