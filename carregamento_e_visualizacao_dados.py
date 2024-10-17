import pandas as pd
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Carregar os CSVs usando Polars
labels = pl.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/labels.csv')
solar_wind = pl.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/solar_wind.csv')
satelite_pos = pl.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/satellite_pos.csv')
sunspot = pl.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/sunspots.csv')

# Visualizar os primeiros registros
print(labels.head())
print(solar_wind.head())
print(satelite_pos.head())
print(sunspot.head())

# Carregar os datasets de labels e solar_wind
labels_df = pd.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/labels.csv')
solar_wind_df = pd.read_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/solar_wind.csv')

# Converter timedelta para datetime se necessário (caso esteja como string)
labels_df['timedelta'] = pd.to_timedelta(labels_df['timedelta'])
solar_wind_df['timedelta'] = pd.to_timedelta(solar_wind_df['timedelta'])

# Filtrar os dados do labels apenas por hora
labels_df_hourly = labels_df[labels_df['timedelta'].dt.components.minutes == 0]

# Fazer a junção (merge) com base na coluna 'timedelta'
merged_df = pd.merge(solar_wind_df, labels_df_hourly[['timedelta', 'dst']], on='timedelta', how='left')

# Exibir o dataset final com a coluna 'dst' adicionada
print(merged_df.head())

# Se desejar, salvar o resultado em um novo arquivo CSV
merged_df.to_csv('/content/drive/MyDrive/IIV DESAFIO DE CIÊNCIA DE DADOS/modelos/dados-tratados/merged_df.csv', index=False)