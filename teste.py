import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Especificar o tamanho dos chunks
chunksize = 100000

# Ler o arquivo solar_wind.csv em chunks
chunk_list = []  # Lista para armazenar os pedaços

for chunk in pd.read_csv('solar_wind.csv', chunksize=chunksize):
    # Tratamento de dados em cada chunk
    chunk_numeric = chunk.select_dtypes(include=['float64', 'int64'])  # Selecionar apenas colunas numéricas
    chunk_numeric = chunk_numeric.interpolate(method='linear', limit_direction='forward', axis=0)
    chunk_numeric.fillna(chunk_numeric.mean(), inplace=True)
    
    # Reintegrar as colunas não numéricas
    chunk.update(chunk_numeric)
    chunk_list.append(chunk)

# Concatenar todos os chunks em um único DataFrame
solar_wind = pd.concat(chunk_list)

# Carregar outros arquivos de dados
satellite_pos = pd.read_csv('satellite_pos.csv')
sunspots = pd.read_csv('sunspots.csv')
labels = pd.read_csv('labels.csv')

# Visualização inicial
print(solar_wind.head())
print(satellite_pos.head())
print(sunspots.head())
print(labels.head())

# Estatísticas descritivas
print(solar_wind.describe())
print(satellite_pos.describe())
print(sunspots.describe())
print(labels.describe())

# Verificar dados faltantes
print(solar_wind.isnull().sum())
print(satellite_pos.isnull().sum())
print(sunspots.isnull().sum())
print(labels.isnull().sum())

# Visualização da distribuição do índice Dst
plt.figure(figsize=(10, 6))
sns.histplot(labels['dst'], kde=True)
plt.title('Distribuição do índice Dst')
plt.xlabel('Dst')
plt.ylabel('Frequência')
plt.show()

# Selecionar apenas colunas numéricas para a análise de correlação
numeric_cols = solar_wind.select_dtypes(include=['float64', 'int64']).columns
corr = solar_wind[numeric_cols].corr()

# Visualizar a matriz de correlação
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação das Variáveis do Vento Solar (Apenas Numéricas)')
plt.show()

# Análise de correlação com o índice Dst
merged_data = solar_wind.merge(labels, on=['period', 'timedelta'])
numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
corr_with_dst = merged_data[numeric_cols].corr()['dst'].drop('dst')

# Visualizar a correlação das variáveis com o índice Dst
plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_dst.index, y=corr_with_dst.values)
plt.title('Correlação das Variáveis com o Índice Dst')
plt.xticks(rotation=90)
plt.show()
