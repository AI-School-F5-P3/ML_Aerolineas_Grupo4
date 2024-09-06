import joblib
import os

# Definir la ruta a la carpeta de modelos
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_model(model_name):
    """
    Carga un modelo y su scaler correspondiente.
    
    Args:
    model_name (str): Nombre del modelo a cargar ('xgboost', 'random_forest', o 'gradient_boosting')
    
    Returns:
    tuple: (modelo, scaler)
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model1.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{model_name}1.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo del modelo o scaler para {model_name}")
        return None, None

def load_all_models():
    """
    Carga todos los modelos y sus scalers correspondientes.
    
    Returns:
    dict: Un diccionario con los nombres de los modelos como claves y tuplas (modelo, scaler) como valores.
    """
    models = {}
    model_names = {
        "XGBoost": "xgboost",
        "Random Forest": "RandomForest",
        "Gradient Boosting": "GradientBoosting"
    }
    
    for display_name, file_name in model_names.items():
        model, scaler = load_model(file_name)
        if model is not None and scaler is not None:
            models[display_name] = (model, scaler)
    
    if not models:
        raise ValueError("No se pudo cargar ningún modelo.")
    
    return models

# Funciones individuales para cargar cada modelo si es necesario
def load_xgboost():
    return load_model("xgboost")

def load_random_forest():
    return load_model("RandomForest")

def load_gradient_boosting():
    return load_model("GradientBoosting")