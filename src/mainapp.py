import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import google.generativeai as genai

os.environ['GOOGLE_API_KEY'] = "AIzaSyCZGGDVIyjebUyHX8m0xO6f1pBD6KKjErc"

# Configure the Gemini API with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the chat model
chat_model = genai.GenerativeModel('gemini-2.0-flash')

# Function to load model
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Model file not found: {model_path}")
        return None
 
# Function to load model and scaler from a single file
def load_model_and_scaler(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_file:
            model_data = pickle.load(model_file)
            return model_data['model'], model_data['scaler'], model_data['feature_order']
    else:
        st.error(f"Model file not found: {model_path}")
        return None, None, None

# Load the models
crop_model_path = 'models/crop_model.pkl'
fertilizer_model_path = 'models/fertilizer_model.pkl'
solar_power_model_path = 'models/SolarPower_model.pkl'

crop_model = load_model(crop_model_path)
fertilizer_model = load_model(fertilizer_model_path)
solar_power_model, scaler, feature_order = load_model_and_scaler(solar_power_model_path)

# Set page config
st.set_page_config(page_title="AgriGenius 360°: AI-Driven Farming Optimization & Renewable Energy Management System", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About the Project", "Crop Recommendation", "Fertilizer Recommendation", "Solar Power Prediction","Chat Assistance"])

# About the Project page
if page == "About the Project":
    st.title("AgriGenius 360°: AI-Driven Farming Optimization & Renewable Energy Management System with Solar Forecasting and Chatbot Assistance")
    st.write("""
    Key components of the Project:
             
    ✅ Core agricultural features:

    1.Machine learning-powered crop recommendations

    2.Fertilizer optimization using ensemble models
             
    ✅ Energy integration: Solar power generation predictions
             
    ✅ User interface: Gemini-powered agricultural chatbot
             
    ✅ Sustainability focus: Aligns with UN SDG goals for smart farming
             
    """)

# Crop Recommendation page
elif page == "Crop Recommendation":
    st.title("Crop Recommendation")
    st.write("Enter the following details to get a crop recommendation:")

    col1, col2 = st.columns(2)
    
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, help="Amount of Nitrogen in soil")
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, help="Amount of Phosphorus in soil")
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, help="Amount of Potassium in soil")
        temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, help="Average temperature in Celsius")

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, help="Average relative humidity")
        ph = st.number_input("pH", min_value=3.5, max_value=10.0, help="pH value of the soil")
        rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, help="Average annual rainfall in mm")

    if st.button("Predict Crop"):
        if crop_model is not None:
            try:
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                scaler = crop_model['scaler']
                scaled_input = scaler.transform(input_data)
                prediction = crop_model['model'].predict(scaled_input)
                crop_dict = crop_model['crop_dict']
                recommended_crop = [key for key, value in crop_dict.items() if value == prediction[0]][0]
                st.success(f"The recommended crop is: {recommended_crop}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.error("Crop recommendation model is not available.")

# Fertilizer Recommendation page
elif page == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    st.write("Enter the following details to get a fertilizer recommendation:")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, help="Current temperature in Celsius")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, help="Current relative humidity")
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, help="Soil moisture content")
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"], help="Type of soil in the field")

    with col2:
        crop_type = st.selectbox("Crop Type", ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley", "Wheat", "Millets", "Oil seeds", "Pulses", "Ground Nuts"], help="Type of crop being grown")
        nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=140, help="Amount of Nitrogen in soil")
        potassium = st.number_input("Potassium (K)", min_value=0, max_value=205, help="Amount of Potassium in soil")
        phosphorous = st.number_input("Phosphorous (P)", min_value=0, max_value=145, help="Amount of Phosphorous in soil")

    if st.button("Predict Fertilizer"):
        if fertilizer_model is not None:
            try:
                input_data = pd.DataFrame({
                    'Temperature': [temperature],
                    'Humidity': [humidity],
                    'Moisture': [moisture],
                    'Soil Type': [soil_type],
                    'Crop Type': [crop_type],
                    'Nitrogen': [nitrogen],
                    'Potassium': [potassium],
                    'Phosphorous': [phosphorous]
                })

                categorical_cols = fertilizer_model['categorical_cols']
                label_encoders = fertilizer_model['label_encoders']
                for col in categorical_cols:
                    input_data[col] = label_encoders[col].transform(input_data[col])

                scaler = fertilizer_model['scaler']
                scaled_input = scaler.transform(input_data)

                prediction = fertilizer_model['model'].predict(scaled_input)
                st.success(f"The recommended fertilizer is: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.error("Fertilizer recommendation model is not available.")

# Solar Power Prediction page
elif page == "Solar Power Prediction":
    st.title("Solar Power Prediction")
    st.write("Enter the following details to get a solar power generation prediction:")

    col1, col2 = st.columns(2)

    with col1:
        distance_to_solar_noon = st.number_input("Distance to Solar Noon", min_value=0.0, max_value=1.0)
        temperature = st.number_input("Temperature (°C)", min_value=-5.0, max_value=35.0)
        wind_direction = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0)
        sky_cover = st.number_input("Sky Cover (%)", min_value=0.0, max_value=100.0)

    with col2:
        visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=10.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
        average_wind_speed_period = st.number_input("Average Wind Speed (m/s)", min_value=0.0, max_value=20.0)
        average_pressure_period = st.number_input("Average Pressure (hPa)", min_value=950.0, max_value=1050.0)

    if st.button("Predict Solar Power"):
        if solar_power_model is not None and scaler is not None:
            try:
                input_data = pd.DataFrame({
                    'distance-to-solar-noon': [distance_to_solar_noon],
                    'temperature': [temperature],
                    'wind-direction': [wind_direction],
                    'wind-speed': [wind_speed],
                    'sky-cover': [sky_cover],
                    'visibility': [visibility],
                    'humidity': [humidity],
                    'average-wind-speed-(period)': [average_wind_speed_period],
                    'average-pressure-(period)': [average_pressure_period]
                }, columns=feature_order)

                scaled_features = scaler.transform(input_data)
                prediction = solar_power_model.predict(scaled_features)

                st.success(f"The predicted solar power generation is: {prediction[0]:.2f} kW")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        else:
            st.error("Model is not available.")
elif page == "Chat Assistance":
    st.title("Agricultural Assistant 🌱")
    
    # Initialize chat session
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = chat_model.start_chat(history=[])
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your farming assistant. Ask me about crops, fertilizers, or solar power!"}]
    
    # Display chat history
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Type your question..."):
        try:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Stream response
            with st.spinner("Thinking..."):
                response = st.session_state.chat_session.send_message(prompt)
                full_response = response.text
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Rerun to update display
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.info("Made by: Mokshit Kaushik")
st.sidebar.text("Version 1.19")