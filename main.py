# Configurações para evitar warnings e logs desnecessários
import os
import absl.logging

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Força o uso de CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mostra apenas erros importantes
absl.logging.set_verbosity(absl.logging.ERROR)

# Importando bibliotecas
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf
from fastapi import FastAPI, HTTPException
import uvicorn
import time

# Coleta dos dados de ações
symbol = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-07-20'
df = yf.download(symbol, start=start_date, end=end_date)

# Preparação dos dados
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Função para criar sequências de dados
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustando os dados para o formato que o LSTM espera
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Construção do modelo LSTM
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Treinamento
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Avaliação do modelo
y_pred = model.predict(X_test)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Salvando o modelo
model.save('model_lstm.h5')

# API com FastAPI
app = FastAPI()

# Carregando o modelo treinado
model = tf.keras.models.load_model('model_lstm.h5')

@app.post("/predict/")
async def predict(data: dict):
    # Endpoint para previsão
    historical_data = data.get("historical_data")
    if not historical_data or len(historical_data) < seq_length:
        raise HTTPException(status_code=400, detail="Forneça pelo menos 60 valores históricos de preços.")

    # Prepara os dados recebidos
    normalized_data = scaler.transform(np.array(historical_data).reshape(-1, 1))
    input_data = np.array(normalized_data[-seq_length:]).reshape(1, seq_length, 1)

    # Faz a previsão
    start_time = time.time()
    prediction = model.predict(input_data)
    end_time = time.time()

    predicted_price = scaler.inverse_transform([[prediction[0][0]]])[0][0]
    response_time = end_time - start_time

    return {
        "predicted_price": predicted_price,
        "response_time_seconds": response_time
    }

@app.get("/monitor/")
async def monitor():
    # Endpoint de monitoramento
    return {
        "model_name": "LSTM Stock Price Predictor",
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        },
        "status": "API is running",
    }

# Iniciar a API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
