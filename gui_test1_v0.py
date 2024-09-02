import os
import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy import exc
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import logging
import hashlib
from dotenv import load_dotenv
from sklearn import preprocessing
import joblib



def hash_string(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

def preprocess_data(df):
    # Encode categorical variables
    le_gender = preprocessing.LabelEncoder()
    le_customer_type = preprocessing.LabelEncoder()
    le_travel_type = preprocessing.LabelEncoder()
    le_class = preprocessing.LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Customer Type'] = le_customer_type.fit_transform(df['Customer Type'])
    df['Type of Travel'] = le_travel_type.fit_transform(df['Type of Travel'])
    df['Class'] = le_class.fit_transform(df['Class'])


    return df

def load_rf_model_and_scaler():
    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = joblib.load('models/rf_model.joblib')
        st.session_state.rf_scaler = joblib.load('models/rf_scaler.pkl')
        st.write("Model and scaler loaded into session state.")
    


def main():

    

    # Load environment variables
    load_dotenv()
    users = {
        "admin": os.getenv("ADMIN_PASSWORD"),
    }

    st.markdown("""
        <style>
            .quote-box {
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
        </style>
        """, unsafe_allow_html=True)

    st.title("Predictor de Satisfacción del Cliente")

    # Session state to keep track of login status
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Login Form
    if not st.session_state.logged_in:
        with st.form("login_form"):
            st.subheader("Autenticación")
            username = st.text_input("Usuario")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if username in users and users["admin"] == hash_string(password):
                    st.session_state.logged_in = True
                    st.success("Autenticación realizada con éxito")
                    st.rerun()
                else:
                    st.error("Usuario o password incorrectos")

    # Interfaz
    if st.session_state.logged_in:
        load_rf_model_and_scaler()
        rf_model = st.session_state.rf_model
        scaler = st.session_state.rf_scaler

        st.sidebar.title("Opciones:")
        option = st.sidebar.radio("Selecciona: ", ("Introducción de Datos", "Predicción ML"))

        
        if option == "Introducción de Datos":
            st.subheader("Introducción de Datos")

            # Initialize session state for sliders if not already done
            if 'slider_values' not in st.session_state:
                st.session_state.slider_values = {
                    'wifi_service': 3,
                    'departure_arrival_time': 3,
                    'online_booking': 3,
                    'gate_location': 3,
                    'food_drink': 3,
                    'online_boarding': 3,
                    'seat_comfort': 3,
                    'entertainment': 3,
                    'onboard_service': 3,
                    'leg_room': 3,
                    'baggage_handling': 3,
                    'checkin_service': 3,
                    'inflight_service': 3,
                    'cleanliness': 3
                }

            # Create input fields for all features
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Flight Distance", min_value=0, value=1000)

            # Sliders with session state
            st.session_state.slider_values['wifi_service'] = st.slider("Inflight wifi service", 0, 5, st.session_state.slider_values['wifi_service'])
            st.session_state.slider_values['departure_arrival_time'] = st.slider("Departure/Arrival time convenient", 0, 5, st.session_state.slider_values['departure_arrival_time'])
            st.session_state.slider_values['online_booking'] = st.slider("Ease of Online booking", 0, 5, st.session_state.slider_values['online_booking'])
            st.session_state.slider_values['gate_location'] = st.slider("Gate location", 0, 5, st.session_state.slider_values['gate_location'])
            st.session_state.slider_values['food_drink'] = st.slider("Food and drink", 0, 5, st.session_state.slider_values['food_drink'])
            st.session_state.slider_values['online_boarding'] = st.slider("Online boarding", 0, 5, st.session_state.slider_values['online_boarding'])
            st.session_state.slider_values['seat_comfort'] = st.slider("Seat comfort", 0, 5, st.session_state.slider_values['seat_comfort'])
            st.session_state.slider_values['entertainment'] = st.slider("Inflight entertainment", 0, 5, st.session_state.slider_values['entertainment'])
            st.session_state.slider_values['onboard_service'] = st.slider("On-board service", 0, 5, st.session_state.slider_values['onboard_service'])
            st.session_state.slider_values['leg_room'] = st.slider("Leg room service", 0, 5, st.session_state.slider_values['leg_room'])
            st.session_state.slider_values['baggage_handling'] = st.slider("Baggage handling", 0, 5, st.session_state.slider_values['baggage_handling'])
            st.session_state.slider_values['checkin_service'] = st.slider("Checkin service", 0, 5, st.session_state.slider_values['checkin_service'])
            st.session_state.slider_values['inflight_service'] = st.slider("Inflight service", 0, 5, st.session_state.slider_values['inflight_service'])
            st.session_state.slider_values['cleanliness'] = st.slider("Cleanliness", 0, 5, st.session_state.slider_values['cleanliness'])

            departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
            arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

            if st.button("Procesar Datos"):
                # Create a dataframe with the input data
                data = {
                    'Gender': [gender],
                    'Customer Type': [customer_type],
                    'Age': [age],
                    'Type of Travel': [travel_type],
                    'Class': [travel_class],
                    'Flight Distance': [flight_distance],
                    'Inflight wifi service': [st.session_state.slider_values['wifi_service']],
                    'Departure/Arrival time convenient': [st.session_state.slider_values['departure_arrival_time']],
                    'Ease of Online booking': [st.session_state.slider_values['online_booking']],
                    'Gate location': [st.session_state.slider_values['gate_location']],
                    'Food and drink': [st.session_state.slider_values['food_drink']],
                    'Online boarding': [st.session_state.slider_values['online_boarding']],
                    'Seat comfort': [st.session_state.slider_values['seat_comfort']],
                    'Inflight entertainment': [st.session_state.slider_values['entertainment']],
                    'On-board service': [st.session_state.slider_values['onboard_service']],
                    'Leg room service': [st.session_state.slider_values['leg_room']],
                    'Baggage handling': [st.session_state.slider_values['baggage_handling']],
                    'Checkin service': [st.session_state.slider_values['checkin_service']],
                    'Inflight service': [st.session_state.slider_values['inflight_service']],
                    'Cleanliness': [st.session_state.slider_values['cleanliness']],
                    'Departure Delay in Minutes': [departure_delay],
                    'Arrival Delay in Minutes': [arrival_delay]
                }
                df = pd.DataFrame(data)

                # Display the input data for verification
                st.write("Datos Registrados:")
                st.write(df)
                df.to_csv("Dataset/single_cust_data.csv")

                # Pre-process the dataframe

                preprocessed_df = preprocess_data(df)
                normalized_df = scaler.transform(preprocessed_df)

                # Store the preprocessed data in session state for use in prediction
                st.session_state.preprocessed_data = normalized_df

                st.success("Datos procesados correctamente. Puede proceder a la predicción.")
                

        elif option == "Predicción ML":
            st.subheader("Predictor Random Forest")

            if 'preprocessed_data' not in st.session_state:
                st.warning("Por favor, introduzca los datos primero en la sección 'Introducción de Datos'.")
            else:
                with st.form("form_2"):
                    st.write("Pulsa el botón para realizar la Predicción.")
                    start_prediction = st.form_submit_button("Iniciar Predicción")

                if start_prediction:
                    
                    # Make predictions
                    raw_prediction = rf_model.predict(st.session_state.preprocessed_data)

                    # Display the results
                    st.write("Resultados de la Predicción:")
                    st.write(f"Interpretación: El cliente probablemente estará:  {raw_prediction}.")

                    
if __name__ == "__main__":
    main()