import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
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

# Function to plot distribution of crops by type
def plot_crops_by_type():
    # Decode the encoded Crop_Type back to original labels
    df['Crop_Type'] = le.inverse_transform(df['Crop_Type'])

    crop_type_counts = df['Crop_Type'].value_counts()
    crop_type_counts.plot(kind='bar', color='blue')
    plt.xlabel('Crop Type')
    plt.ylabel('Count')
    plt.title('Distribution of Crops by Type')
    st.pyplot()  # Display the plot in Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)


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
    st.set_page_config(page_title='Agricultural Analytics Dashboard')

    st.title('Agricultural Analytics Dashboard')

    # Display distribution of crops by type
    st.header('Distribution of Crops by Type')
    plot_crops_by_type()

    # Streamlit app title for prediction
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

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make the prediction
    prediction_numeric = model.predict(input_df)[0]

    # Convert the numeric prediction back to the original crop type name
    prediction_name = le.inverse_transform([prediction_numeric])[0]
    
    st.write(f"Predicted Crop Type: {prediction_name}")


if __name__ == '__main__':
    main()
