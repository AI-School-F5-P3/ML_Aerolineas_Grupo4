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




def hash_string(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

def preprocess_data(df):
    # Codificar variables categóricas
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
        st.session_state.rf_scaler = joblib.load('models/rf_scaler.joblib')
        #st.write("Modelo RF cargado.")

def load_nn_model_and_scaler():
    if 'nn_model' not in st.session_state:
        st.session_state.nn_model = load_model('models/airline_satisfaction_model.keras')
        st.session_state.nn_scaler = joblib.load('models/nn_scaler.joblib')
        #st.write("Modelo NN cargado.")


Base = declarative_base()

class AirlineCustomers(Base):
    __tablename__ = 'customers'

    id = Column(Integer, primary_key=True)
    Gender = Column(String, name='Gender')
    Customer_Type = Column(String, name='Customer Type')
    Age = Column(Integer, name='Age')
    Type_of_Travel = Column(String, name='Type of Travel')
    Class = Column(String, name='Class')
    Flight_Distance = Column(Integer, name='Flight Distance')
    Inflight_wifi_service = Column(Integer, name='Inflight wifi service')
    Departure_Arrival_time_convenient = Column(Integer, name='Departure/Arrival time convenient')
    Ease_of_Online_booking = Column(Integer, name='Ease of Online booking')
    Gate_location = Column(Integer, name='Gate location')
    Food_and_drink = Column(Integer, name='Food and drink')
    Online_boarding = Column(Integer, name='Online boarding')
    Seat_comfort = Column(Integer, name='Seat comfort')
    Inflight_entertainment = Column(Integer, name='Inflight entertainment')
    On_board_service = Column(Integer, name='On-board service')
    Leg_room_service = Column(Integer, name='Leg room service')
    Baggage_handling = Column(Integer, name='Baggage handling')
    Checkin_service = Column(Integer, name='Checkin service')
    Inflight_service = Column(Integer, name='Inflight service')
    Cleanliness = Column(Integer, name='Cleanliness')
    Departure_Delay_in_Minutes = Column(Integer, name='Departure Delay in Minutes')
    Arrival_Delay_in_Minutes = Column(Integer, name='Arrival Delay in Minutes')
    Pred_satisfaction = Column(String, name='satisfaction')
    Real_satisfaction = Column(String, name='r_satisfaction')


def add_customer(session, customer_df):
    # Nos quedamos con la primera fila
    customer_data = customer_df.iloc[0]

    # Instancia de cliente
    new_customer = AirlineCustomers(
        Gender=customer_data['Gender'],
        Customer_Type=customer_data['Customer Type'],
        Age=int(customer_data['Age']),
        Type_of_Travel=customer_data['Type of Travel'],
        Class=customer_data['Class'],
        Flight_Distance=int(customer_data['Flight Distance']),
        Inflight_wifi_service=int(customer_data['Inflight wifi service']),
        Departure_Arrival_time_convenient=int(customer_data['Departure/Arrival time convenient']),
        Ease_of_Online_booking=int(customer_data['Ease of Online booking']),
        Gate_location=int(customer_data['Gate location']),
        Food_and_drink=int(customer_data['Food and drink']),
        Online_boarding=int(customer_data['Online boarding']),
        Seat_comfort=int(customer_data['Seat comfort']),
        Inflight_entertainment=int(customer_data['Inflight entertainment']),
        On_board_service=int(customer_data['On-board service']),
        Leg_room_service=int(customer_data['Leg room service']),
        Baggage_handling=int(customer_data['Baggage handling']),
        Checkin_service=int(customer_data['Checkin service']),
        Inflight_service=int(customer_data['Inflight service']),
        Cleanliness=int(customer_data['Cleanliness']),
        Departure_Delay_in_Minutes=int(customer_data['Departure Delay in Minutes']),
        Arrival_Delay_in_Minutes=int(customer_data['Arrival Delay in Minutes']),
        Pred_satisfaction=customer_data['satisfaction']
    )

    # Añadimos el nuevo cliente a la BBDD
    session.add(new_customer)
    session.commit()


def classify_values(values):
    classifications = []
    for value in values:
        if abs(value[0] - 0) <= 0.5:
            classifications.append('neutral or dissatisfied')
        else:
            classifications.append('satisfied')
        
    return classifications

def classify_value(value):
    classifications = []
    if abs(value) <= 0.5:
        classifications.append('neutral or dissatisfied')
    else:
        classifications.append('satisfied')
        
    return classifications


# Función para actualizar el feedback en la base de datos
def update_feedback(session, user_id, real_satisfaction):
    try:
        # Actualizar directamente en la tabla 'customers' usando SQL
        result = session.execute(
            text("UPDATE customers SET r_satisfaction = :satisfaction WHERE id = :id"),
            {"satisfaction": real_satisfaction, "id": user_id}
        )
        
        if result.rowcount == 0:
            raise ValueError(f"No se encontró un registro con ID {user_id}")
        
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise Exception(f"Error de base de datos: {str(e)}")
    except Exception as e:
        session.rollback()
        raise e

