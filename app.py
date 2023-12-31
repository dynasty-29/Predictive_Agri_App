import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('synthetic_crop_data.csv')  

# Create a LabelEncoder object
le = LabelEncoder()

# Encode categorical features
categorical_features = ['Crop_Type']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Scale numeric features
scaler = MinMaxScaler()
numeric_features = ['Temperature', 'Humidity', 'Precipitation', 'Wind_Speed', 'Solar_Radiation',
                    'Nitrogen_Content', 'Phosphorous_Content', 'Potassium_Content', 'Soil_pH']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Split data into features and target variable
X = df.drop('Crop_Type', axis=1)
y = df['Crop_Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the gradient boosting model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Load animal data
df_animal = pd.read_csv('synthetic_Animal_data.csv')

# Create a LabelEncoder object for animals
le_animal = LabelEncoder()

# Encode categorical features for animals
categorical_features_animal = ['Breed', 'Health_Status', 'Lactation_Stage', 'Reproductive_Status', 'Milking_Frequency', 'Environmental_Housing']
for feature in categorical_features_animal:
    df_animal[feature] = le_animal.fit_transform(df_animal[feature])

# Split animal data into features and target variable
X_animal = df_animal.drop('Milk_Production', axis=1)
y_animal = df_animal['Milk_Production']

# Split animal data into training and testing sets
X_train_animal, X_test_animal, y_train_animal, y_test_animal = train_test_split(X_animal, y_animal, test_size=0.2, random_state=42)

# Train the animal gradient boosting model
model_animal = GradientBoostingRegressor()
model_animal.fit(X_train_animal, y_train_animal)


   
# Function to make personalized recommendations
def make_personalized_recommendations(crop_type):
    # Check if the crop type exists in the dataset
    if crop_type in df['Crop_Type'].values:
        # Retrieve crop data
        crop_data = df[df['Crop_Type'] == crop_type].iloc[0]

        # Compute similarity scores between the selected crop and all other crops
        df['Similarity Score'] = df[numeric_features].apply(lambda x: sum((x - crop_data[numeric_features]) ** 2), axis=1)

        # Get top 5 similar crops based on similarity scores
        similar_crops = df[df['Crop_Type'] != crop_type].nlargest(5, 'Similarity Score')

        return similar_crops['Crop_Type']
    else:
        # Return a random sample of crop types if the input is invalid
        return df['Crop_Type'].sample(5)

# Streamlit app
def main():
    st.set_page_config(page_title='Animal and Crop Farming Prediction')

    st.title('Animal and Crop Farming Prediction App')

    # Layout for crop prediction
    st.sidebar.title("Crop Type Prediction Values")

    # Sidebar for environmental data input
    st.sidebar.subheader("Enter Environmental Data:")
    temperature = st.sidebar.slider("Temperature (°C)", 15.0, 39.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
    precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 100.0, 50.0)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 20.0, 10.0)
    solar_radiation = st.sidebar.slider("Solar Radiation (W/m²)", 0.0, 2000.0, 800.0)

    # Streamlit form for optional soil nutrient levels
    st.sidebar.subheader("Enter Optional Soil Nutrient Levels (Leave blank if not available):")
    nitrogen_level = st.sidebar.slider("Nitrogen Level (%)", 0, 100, 50)
    phosphorus_level = st.sidebar.slider("Phosphorus Level (%)", 0, 100, 50)
    potassium_level = st.sidebar.slider("Potassium Level (ppm)", 0, 100, 50)
    ph_level = st.sidebar.slider("Soil pH (0 to 14)", 0.0, 14.0, 7.0)

    # Button to trigger crop prediction
    if st.button("Predict Crop"):
        # Prepare input data for prediction
        input_data = {
            "Temperature": temperature,
            "Humidity": humidity,
            "Precipitation": precipitation,
            "Wind_Speed": wind_speed,
            "Solar_Radiation": solar_radiation,
            "Nitrogen_Content": nitrogen_level,
            "Phosphorous_Content": phosphorus_level,
            "Potassium_Content": potassium_level,
            "Soil_pH": ph_level,
        }

        # Ensure that input_data is not empty or None
        if input_data:
            # Convert input data to a DataFrame
            input_df = pd.DataFrame([input_data])

            # Make the prediction
            prediction_numeric = model.predict(input_df)[0]

            # Convert the numeric prediction back to the original crop type name
            prediction_name = le.inverse_transform([prediction_numeric])[0]

            # Display the prediction at the center
            st.title("Crop Prediction Result")
            st.write(f"Predicted Crop Type: {prediction_name}")
        else:
            st.write("Input data is empty. Please provide values.")

    # Animal prediction section
    st.sidebar.title("Animal Prediction Values")
    
    breed = st.sidebar.selectbox("Breed Type",["Ayrshire","Holstein","Guernsey", "Jersey"])
    health_status = st.sidebar.selectbox("Health Status",["Healthy", "Unhealthy"])
    lactation_stage = st.sidebar.selectbox("Lactation Stage",["Mid Lactation","Early Lactation","Late Lactation"])
    reproductive_status = st.sidebar.selectbox("Reproductive Status",["Calving","Post-Calving","Pregnant"])
    milking_frequency = st.sidebar.selectbox("Milking Frequency",["Three times a day","Two times a day"])
    env_housing = st.sidebar.selectbox("Environmental Housing",["Barn","Pasture"])
    age = st.sidebar.slider("Age", 1, 10, 5)
    nutrition_protein = st.sidebar.slider("Nutrition Protein", 0.0, 100.0, 50.0)
    nutrition_carbohydrates = st.sidebar.slider("Nutrition Carbohydrates", 0.0, 100.0, 50.0)
    nutrition_minerals = st.sidebar.slider("Nutrition Minerals", 0.0, 100.0, 50.0)
    env_temperature = st.sidebar.slider("Environmental Temperature", 0.0, 40.0, 25.0)
    env_humidity = st.sidebar.slider("Environmental Humidity", 0, 100, 60)
    prev_milk_production = st.sidebar.slider("Previous Milk Production (Litres)", 0.0, 20.0, 5.0)
        
    
        # Button to trigger animal prediction
    if st.button("Predict Animal"):
        # Prepare input data for animal prediction
        input_data_animal = {
            "Breed": le_animal.transform([breed])[0] if breed in le_animal.classes_ else -1,
            "Age": age,
            "Nutrition_Protein": nutrition_protein,
            "Nutrition_Carbohydrates": nutrition_carbohydrates,
            "Nutrition_Minerals": nutrition_minerals,
            "Health_Status": le_animal.transform([health_status])[0] if health_status in le_animal.classes_ else -1,
            "Lactation_Stage": le_animal.transform([lactation_stage])[0] if lactation_stage in le_animal.classes_ else -1,
            "Reproductive_Status": le_animal.transform([reproductive_status])[0] if reproductive_status in le_animal.classes_ else -1,
            "Milking_Frequency": le_animal.transform([milking_frequency])[0] if milking_frequency in le_animal.classes_ else -1,
            "Environmental_Temperature": env_temperature,
            "Environmental_Humidity": env_humidity,
            "Environmental_Housing": le_animal.transform([env_housing])[0] if env_housing in le_animal.classes_ else -1,
            "Previous_Milk_Production": prev_milk_production,
        }

        # Convert input data to a DataFrame for animal prediction
        input_df_animal = pd.DataFrame([input_data_animal])

        # Make the animal prediction
        prediction_milk_production = model_animal.predict(input_df_animal)[0]

        # Display the animal prediction at the center
        st.title("Animal Prediction Result")
        st.write(f"Predicted Milk Production: {prediction_milk_production:.2f} Litres")



   
if __name__ == '__main__':
    main()
