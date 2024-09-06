import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# Añadir la carpeta 'scripts' al path de Python
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.append(scripts_dir)

# Importar las funciones de carga de modelos desde cada script
from XGBOOST1 import load_model as load_xgboost_model
from RandomForest1 import load_model as load_rf_model
from GradientBoosting1 import load_model as load_gb_model

def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model1.pkl'))
    xgb_scaler = joblib.load(os.path.join(models_dir, 'scaler_xgboost1.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'RandomForest_model1.pkl'))
    rf_scaler = joblib.load(os.path.join(models_dir, 'scaler_RandomForest1.pkl'))
    gb_model = joblib.load(os.path.join(models_dir, 'GradientBoosting_model1.pkl'))
    gb_scaler = joblib.load(os.path.join(models_dir, 'scaler_GradientBoosting1.pkl'))
    return {
        "XGBoost": (xgb_model, xgb_scaler),
        "Random Forest": (rf_model, rf_scaler),
        "Gradient Boosting": (gb_model, gb_scaler)
    }

def main():
    st.title("Predicción de Satisfacción de Pasajeros de Aerolíneas")

    # Cargar todos los modelos
    models = load_models()

    # Selector de modelo
    selected_model = st.selectbox("Seleccione el modelo a utilizar", list(models.keys()))

    # Recoger los datos de entrada del usuario
    gender = st.selectbox("Género", ["Female", "Male"])
    customer_type = st.selectbox("Tipo de Cliente", ["Loyal Customer", "disloyal Customer"])
    age = st.slider("Edad", 0, 100, 30)
    type_of_travel = st.selectbox("Tipo de Viaje", ["Personal Travel", "Business travel"])
    customer_class = st.selectbox("Clase", ["Eco", "Eco Plus", "Business"])
    flight_distance = st.number_input("Distancia de Vuelo", min_value=0)
    inflight_wifi_service = st.slider("Servicio WiFi a Bordo", 0, 5, 3)
    departure_arrival_time_convenient = st.slider("Conveniencia de Hora de Salida/Llegada", 0, 5, 3)
    ease_of_online_booking = st.slider("Facilidad de Reserva Online", 0, 5, 3)
    gate_location = st.slider("Ubicación de la Puerta", 0, 5, 3)
    food_and_drink = st.slider("Comida y Bebida", 0, 5, 3)
    online_boarding = st.slider("Embarque Online", 0, 5, 3)
    seat_comfort = st.slider("Comodidad del Asiento", 0, 5, 3)
    inflight_entertainment = st.slider("Entretenimiento a Bordo", 0, 5, 3)
    on_board_service = st.slider("Servicio a Bordo", 0, 5, 3)
    leg_room_service = st.slider("Espacio para Piernas", 0, 5, 3)
    baggage_handling = st.slider("Manejo de Equipaje", 0, 5, 3)
    checkin_service = st.slider("Servicio de Check-in", 0, 5, 3)
    inflight_service = st.slider("Servicio en Vuelo", 0, 5, 3)
    cleanliness = st.slider("Limpieza", 0, 5, 3)
    departure_delay_in_minutes = st.number_input("Retraso en la Salida (minutos)", min_value=0)
    arrival_delay_in_minutes = st.number_input("Retraso en la Llegada (minutos)", min_value=0)

    if st.button("Predecir"):
        # Crear un DataFrame con los datos de entrada
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Customer Type': [customer_type],
            'Age': [age],
            'Type of Travel': [type_of_travel],
            'Class': [customer_class],
            'Flight Distance': [flight_distance],
            'Inflight wifi service': [inflight_wifi_service],
            'Departure/Arrival time convenient': [departure_arrival_time_convenient],
            'Ease of Online booking': [ease_of_online_booking],
            'Gate location': [gate_location],
            'Food and drink': [food_and_drink],
            'Online boarding': [online_boarding],
            'Seat comfort': [seat_comfort],
            'Inflight entertainment': [inflight_entertainment],
            'On-board service': [on_board_service],
            'Leg room service': [leg_room_service],
            'Baggage handling': [baggage_handling],
            'Checkin service': [checkin_service],
            'Inflight service': [inflight_service],
            'Cleanliness': [cleanliness],
            'Departure Delay in Minutes': [departure_delay_in_minutes],
            'Arrival Delay in Minutes': [arrival_delay_in_minutes]
        })

        # Codificar variables categóricas
        categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col in categorical_columns:
            le = LabelEncoder()
            input_data[col] = le.fit_transform(input_data[col])

        # Obtener el modelo y scaler seleccionados
        model, scaler = models[selected_model]

        # Escalar los datos de entrada
        scaled_input = scaler.transform(input_data)

        # Hacer la predicción
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)

        # Mostrar el resultado
        st.write(f"Modelo utilizado: {selected_model}")
        st.write(f"Predicción: {'Satisfecho' if prediction[0] == 1 else 'Insatisfecho'}")
        st.write(f"Probabilidad de satisfacción: {probability[0][1]:.2f}")

if __name__ == "__main__":
    main()