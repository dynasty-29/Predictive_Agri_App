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
    precipitation = st.sidebar.slider("Precipitation", 0.0, 100.0, 50.0)
    wind_speed = st.sidebar.slider("Wind Speed", 0.0, 20.0, 10.0)
    solar_radiation = st.sidebar.slider("Solar Radiation", 0.0, 2000.0, 800.0)

    # Streamlit form for optional soil nutrient levels
    st.sidebar.subheader("Enter Optional Soil Nutrient Levels (Leave blank if not available):")
    nitrogen_level = st.sidebar.slider("Nitrogen Level", 0, 100, 50)
    phosphorus_level = st.sidebar.slider("Phosphorus Level", 0, 100, 50)
    potassium_level = st.sidebar.slider("Potassium Level", 0, 100, 50)
    ph_level = st.sidebar.slider("Soil pH", 0.0, 14.0, 7.0)

    # Button to trigger crop prediction
    if st.sidebar.button("Predict Crop"):
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
    st.title("Animal Prediction App")
    
    left_column, center_column, right_column = st.beta_columns([1, 3, 1])
    # Streamlit form for animal prediction

    with right_column:
        breed_le = LabelEncoder()
        breed_le.fit(df_animal['Breed'])
        breed_options = breed_le.classes_
        selected_breed_index = st.selectbox("Select Breed", range(len(breed_options)), format_func=lambda x: breed_options[x])
        breed = breed_options[selected_breed_index]
        
        health_status_mapping = {numeric_value: status for numeric_value, status in enumerate(df_animal['Health_Status'].unique())}
        selected_health_status = st.selectbox("Select Health Status", list(health_status_mapping.values()))
        health_status = list(health_status_mapping.keys())[list(health_status_mapping.values()).index(selected_health_status)]
    
        lactation_stage_mapping = {numeric_value: status for numeric_value, status in enumerate(df_animal['Lactation_Stage'].unique())}
        selected_lactation_stage = st.selectbox("Select Lactation_Stage", list(lactation_stage_mapping.values()))
        lactation_stage = list(lactation_stage_mapping.keys())[list(lactation_stage_mapping.values()).index(selected_lactation_stage)]
    
        reproductive_status_mapping = {numeric_value: status for numeric_value, status in enumerate(df_animal['Reproductive_Status'].unique())}
        selected_reproductive_status = st.selectbox("Select Reproductive_Status", list(reproductive_status_mapping.values()))
        reproductive_status = list(reproductive_status_mapping.keys())[list(reproductive_status_mapping.values()).index(selected_reproductive_status)]
    
        milking_frequency_mapping = {numeric_value: status for numeric_value, status in enumerate(df_animal['Milking_Frequency'].unique())}
        selected_milking_frequency = st.selectbox("Select Milking_Frequency", list(milking_frequency_mapping.values()))
        milking_frequency = list(milking_frequency_mapping.keys())[list(milking_frequency_mapping.values()).index(selected_milking_frequency)]
    
        env_housing_mapping = {numeric_value: status for numeric_value, status in enumerate(df_animal['Environmental_Housing'].unique())}
        selected_env_housing = st.selectbox("Select Environmental_Housing", list(env_housing_mapping.values()))
        env_housing = list(env_housing_mapping.keys())[list(env_housing_mapping.values()).index(selected_env_housing)]
          
        age = st.slider("Age", 1, 10, 5)
        nutrition_protein = st.slider("Nutrition Protein", 0.0, 100.0, 50.0)
        nutrition_carbohydrates = st.slider("Nutrition Carbohydrates", 0.0, 100.0, 50.0)
        nutrition_minerals = st.slider("Nutrition Minerals", 0.0, 100.0, 50.0)
        env_temperature = st.slider("Environmental Temperature", 0.0, 40.0, 25.0)
        env_humidity = st.slider("Environmental Humidity", 0, 100, 60)
        prev_milk_production = st.slider("Previous Milk Production (Litres)", 0.0, 20.0, 5.0)
        
    with center_column:
    # Button to trigger animal prediction
        if st.button("Predict Animal"):
            # Prepare input data for animal prediction
            input_data_animal = {
                "Breed": breed,
                "Age": age,
                "Nutrition_Protein": nutrition_protein,
                "Nutrition_Carbohydrates": nutrition_carbohydrates,
                "Nutrition_Minerals": nutrition_minerals,
                "Health_Status": health_status,
                "Lactation_Stage": lactation_stage,
                "Reproductive_Status": reproductive_status,
                "Milking_Frequency": milking_frequency,
                "Environmental_Temperature": env_temperature,
                "Environmental_Humidity": env_humidity,
                "Environmental_Housing": env_housing,
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
