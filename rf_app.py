import os
import pandas as pd
import streamlit as st
from sklearn import preprocessing
import joblib
import hashlib

# Función para hashear una cadena (útil para crear identificadores únicos)
def hash_string(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

# Función para preprocesar datos
def preprocess_data(df):
    le_gender = preprocessing.LabelEncoder()
    le_customer_type = preprocessing.LabelEncoder()
    le_travel_type = preprocessing.LabelEncoder()
    le_class = preprocessing.LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Customer Type'] = le_customer_type.fit_transform(df['Customer Type'])
    df['Type of Travel'] = le_travel_type.fit_transform(df['Type of Travel'])
    df['Class'] = le_class.fit_transform(df['Class'])

    return df

# Función para cargar el modelo y el scaler
def load_rf_model_and_scaler():
    current_dir = os.getcwd()
    st.write(f"Current working directory: {current_dir}")
    
    model_path = os.path.join(current_dir, 'rf_model.joblib')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    
    st.write(f"Looking for model at: {model_path}")
    st.write(f"Looking for scaler at: {scaler_path}")
    
    st.write(f"Files in current directory: {os.listdir(current_dir)}")
    
    if os.path.exists(model_path):
        st.write("Model file found!")
        st.session_state.rf_model = joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}")
    
    if os.path.exists(scaler_path):
        st.write("Scaler file found!")
        st.session_state.rf_scaler = joblib.load(scaler_path)
    else:
        st.error(f"Scaler file not found at {scaler_path}")
    
    if 'rf_model' in st.session_state and 'rf_scaler' in st.session_state:
        st.write("Model and scaler loaded into session state.")
    else:
        st.error("Failed to load model and/or scaler.")

def main():
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

    # Inicializar st.session_state.page si no está definido
    if 'page' not in st.session_state:
        st.session_state.page = "Introducción de Datos"

    # Cargar el modelo y el scaler
    load_rf_model_and_scaler()
    rf_model = st.session_state.rf_model
    scaler = st.session_state.rf_scaler

    # Manejo del formulario en la misma página
    st.subheader("Introducción de Datos")

    # Inicializar estado de los sliders si no está hecho
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

    # Crear campos de entrada para todas las características
    gender = st.selectbox("Gender", ["Female", "Male"])
    customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    flight_distance = st.number_input("Flight Distance", min_value=0, value=1000)

    # Sliders con estado de sesión
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
        # Crear un dataframe con los datos de entrada
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

        # Mostrar los datos ingresados para verificación
        st.write("Datos Registrados:")
        st.write(df)

        # Preprocesar el dataframe
        preprocessed_df = preprocess_data(df)
        normalized_df = scaler.transform(preprocessed_df)

        # Guardar los datos preprocesados en el estado de sesión para su uso en la predicción
        st.session_state.preprocessed_data = normalized_df

        # Realizar la predicción
        raw_prediction = rf_model.predict(normalized_df)

        # Mostrar los resultados
        st.write("Resultados de la Predicción:")
        st.write(f"Interpretación: El cliente probablemente estará: {'Satisfecho' if raw_prediction[0] == 1 else 'No Satisfecho'}.")

        st.success("Datos procesados y predicción realizada.")

if __name__ == "__main__":
    main()
