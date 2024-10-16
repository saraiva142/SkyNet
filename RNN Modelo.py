import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregando os dados
solar_wind = pd.read_csv('solar_wind.csv')
labels = pd.read_csv('labels.csv')

# Pré-processamento dos dados
merged_data = solar_wind.merge(labels, on=['period', 'timedelta'])

numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numeric_cols] = merged_data[numeric_cols].interpolate(method='linear', limit_direction='forward', axis=0)
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())
merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())

features = merged_data[["bt","temperature","bx_gse","by_gse","bz_gse","phi_gse","theta_gse","bx_gsm","by_gsm","bz_gsm","phi_gsm","theta_gsm","speed","density"]]
target = merged_data['dst']

# Padronização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Construindo a rede neural com ajustes
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)
])

# Compilando o modelo
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001), loss='mse')

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=128, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Plotando a curva de perda
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotando o gráfico de valores reais vs preditos
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Valores Reais', color='blue')
plt.plot(y_pred, label='Valores Preditos - Rede Neural', color='orange')
plt.title('Comparação de Valores Reais e Preditos')
plt.xlabel('Tempo')
plt.ylabel('Índice Dst')
plt.legend()
plt.show()
