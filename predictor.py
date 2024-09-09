import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy import exc, text
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import logging
import random
import hashlib
from dotenv import load_dotenv
from sklearn import preprocessing
from tensorflow.keras.models import load_model
import joblib

from aux_funtions import *


def main():
 
    # Cargamos variables de entorno
    load_dotenv()
    DATABASE_URL = os.getenv('DATABASE_URL')

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

    # Session state para el estado de log in
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
        rf_scaler = st.session_state.rf_scaler

        load_nn_model_and_scaler()
        nn_model = st.session_state.nn_model
        nn_scaler = st.session_state.nn_scaler

        st.sidebar.title("Opciones:")
        option = st.sidebar.radio("Selecciona: ", ("Datos y Predicción", "Almacenar en BBDD", "Mostrar BBDD", "Incorporar Feedback"))
        
        
        if option == "Datos y Predicción":
            st.subheader("Datos y Predicción de Satisfacción de un Nuevo Cliente")

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
            st.session_state.slider_values['wifi_service'] = st.slider("Inflight wifi service", 1, 5, st.session_state.slider_values['wifi_service'])
            st.session_state.slider_values['departure_arrival_time'] = st.slider("Departure/Arrival time convenient", 1, 5, st.session_state.slider_values['departure_arrival_time'])
            st.session_state.slider_values['online_booking'] = st.slider("Ease of Online booking", 1, 5, st.session_state.slider_values['online_booking'])
            st.session_state.slider_values['gate_location'] = st.slider("Gate location", 1, 5, st.session_state.slider_values['gate_location'])
            st.session_state.slider_values['food_drink'] = st.slider("Food and drink", 1, 5, st.session_state.slider_values['food_drink'])
            st.session_state.slider_values['online_boarding'] = st.slider("Online boarding", 1, 5, st.session_state.slider_values['online_boarding'])
            st.session_state.slider_values['seat_comfort'] = st.slider("Seat comfort", 1, 5, st.session_state.slider_values['seat_comfort'])
            st.session_state.slider_values['entertainment'] = st.slider("Inflight entertainment", 1, 5, st.session_state.slider_values['entertainment'])
            st.session_state.slider_values['onboard_service'] = st.slider("On-board service", 1, 5, st.session_state.slider_values['onboard_service'])
            st.session_state.slider_values['leg_room'] = st.slider("Leg room service", 1, 5, st.session_state.slider_values['leg_room'])
            st.session_state.slider_values['baggage_handling'] = st.slider("Baggage handling", 1, 5, st.session_state.slider_values['baggage_handling'])
            st.session_state.slider_values['checkin_service'] = st.slider("Checkin service", 1, 5, st.session_state.slider_values['checkin_service'])
            st.session_state.slider_values['inflight_service'] = st.slider("Inflight service", 1, 5, st.session_state.slider_values['inflight_service'])
            st.session_state.slider_values['cleanliness'] = st.slider("Cleanliness", 1, 5, st.session_state.slider_values['cleanliness'])

            departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
            arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

            if st.button("Predicciones"):
                # Creamos un dataframe con los datos de entrada
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
                normalized_df = rf_scaler.transform(preprocessed_df)
                normalized_df_nn = nn_scaler.transform(preprocessed_df)

                # Store the preprocessed data in session state for use in prediction
                st.session_state.preprocessed_data = normalized_df
                st.session_state.preprocessed_data_nn = normalized_df_nn

                # Realizar predicciones
                raw_prediction = rf_model.predict(st.session_state.preprocessed_data)
                st.session_state.prediction = raw_prediction[0]

                # Activar la bandera de predicción realizada
                st.session_state.prediction_made = True

                # Predicción con red neuronal
                raw_prediction_nn = nn_model.predict(st.session_state.preprocessed_data_nn)
                classified_result_nn = classify_values(raw_prediction_nn)
                st.session_state.prediction_nn = classified_result_nn[0] 

                # Muestra los resultados
                st.write("Resultado de la Predicción mediante **Random Forest:**")
                st.write(f"El cliente probablemente estará:  {st.session_state.prediction}.")
                st.write("")
                st.write("")
                st.write("Resultado de la Predicción mediante **Red Neuronal:**")
                st.write(f"El cliente probablemente estará:  {st.session_state.prediction_nn}.")

                 
                   
       
        elif option == "Almacenar en BBDD":
            st.subheader("Almacenar en BBDD")

            # Comprobar si se ha realizado una predicción antes de permitir el almacenamiento
            if 'prediction_made' not in st.session_state or not st.session_state.prediction_made:
                st.warning("Por favor, realice una predicción en la sección 'Predicción ML' antes de almacenar los datos.")
            else:
                with st.form("form_3"):
                    # Leer el archivo CSV previamente guardado
                    df_to_store = pd.read_csv("Dataset/single_cust_data.csv", index_col=0)

                   
                    #Seleccionamos al azar una de las dos predicciones
                    df_to_store['satisfaction'] = random.choice([st.session_state.prediction, st.session_state.prediction_nn])
                    

                    # Mostrar los datos que se van a almacenar para verificación
                    st.write("Datos a Almacenar:")
                    st.write(df_to_store)

                    # Botón para confirmar el almacenamiento en la base de datos
                    confirm_storage = st.form_submit_button("Confirmar Almacenamiento")

                if confirm_storage:
                    # Comprobar si la sesión de SQLAlchemy está creada, sino crearla
                    if 'db_session' not in st.session_state:
                        engine = create_engine(DATABASE_URL)
                        Base.metadata.create_all(engine)
                        Session = sessionmaker(bind=engine)
                        st.session_state.db_session = Session()

                    # Usar la sesión de st.session_state
                    session = st.session_state.db_session

                    # Llamar a la función para almacenar los datos en la base de datos
                    if st.session_state.prediction_made:
                        add_customer(session, df_to_store)
                        st.success("Datos almacenados correctamente en la base de datos.")

                        # Desactivar la bandera de predicción realizada después de almacenar
                        st.session_state.prediction_made = False
        
        
        
        elif option == "Mostrar BBDD":
            st.subheader("Mostrar BBDD")
            
            # Crear un nuevo formulario
            with st.form("form_4"):
                # Botón para mostrar la base de datos
                show_db = st.form_submit_button("Mostrar la Base de Datos")
            
            if show_db:
                try:
                    # Crear conexión a la base de datos
                    engine = create_engine(DATABASE_URL)

                    query = "SELECT * FROM customers"
                    df = pd.read_sql_query(query, engine)
                    
                    
                    # Mostrar el contenido de la base de datos
                    st.dataframe(df, hide_index=True)
                except pd.errors.DatabaseError as e:
                    st.error(f"Error cargando contenido de la BBDD: {str(e)}")
                except Exception as e:
                    st.error(f"Error inesperado: {str(e)}")
                finally:
                    engine.dispose()

        elif option == "Incorporar Feedback":
            st.subheader("Incorporar Feedback")
            
            with st.form("form_feedback"):
                user_id = st.number_input("ID de Usuario", min_value=1, step=1, value=1)
                real_satisfaction = st.selectbox(
                    "Satisfacción Real",
                    options=['satisfied', 'neutral or dissatisfied'],
                    index=1  # Por defecto selecciona 'neutral or dissatisfied'
                )
                submit_feedback = st.form_submit_button("Enviar Feedback")

            if submit_feedback:
                try:
                    # Comprobar si la sesión de SQLAlchemy está creada, sino crearla
                    if 'db_session' not in st.session_state:
                        engine = create_engine(DATABASE_URL)
                        Base.metadata.create_all(engine)
                        Session = sessionmaker(bind=engine)
                        st.session_state.db_session = Session()
                    
                    # Usar la sesión de st.session_state
                    session = st.session_state.db_session
                    
                    # Llamar a la función para actualizar el feedback
                    update_feedback(session, int(user_id), real_satisfaction)
                    st.success(f"Feedback incorporado correctamente para el usuario {user_id}.")
                except Exception as e:
                    st.error(f"Error al incorporar el feedback: {str(e)}")



                    
if __name__ == "__main__":
    main()