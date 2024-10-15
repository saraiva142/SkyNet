<h1 align="center">SKYNET - IIV DESAFIO DE CIÊNCIA DE DADOS 🤖 </h1>

## Equipe:

<div>
  
  [<img src="https://avatars.githubusercontent.com/u/64429904?v=4" alt="João Victor Saraiva" width="100">](https://github.com/saraiva142)
  [<img src = "https://avatars.githubusercontent.com/u/147951553?v=4" alt ="Ludmila" width = "100">](https://github.com/LudLES-t)
  
</div>

## Projeto:

### Sobre o desafio:

"O vento solar interagindo com a magnetosfera da Terra pode causar tempestades geomagnéticas, danificando tecnologias críticas. O índice de perturbação-tempestade-tempo (Dst) mede a gravidade da tempestade, impulsionando modelos de perturbação geomagnética. A previsão tradicional de Dst depende de observações do vento solar, mas modelos recentes de Aprendizado de Máquina (ML) mostram-se promissores". Para explorar soluções viáveis de ML, a Pontifícia Universidade Católica de Goiás organizou o VII Desafio de Ciência de Dados. Com o objetivo de modelos baseados em ML para prever Dst sejam criados e estudados. O documento resume os resultados e insights do problema proposto, enfatizando o potencial do ML para aprimorar a previsão de tempestades geomagnéticas para aplicações práticas.

### Motivo do desafio:

O desafio busca promover o trabalho em equipe, análise exploratória de dados, filtragem de dados, modelagem e algoritmos, aprendizado de linguagens de programação, machine learning e raciocínio lógico, com o objetivo de analisar os dados e fazer predições com séries temporais.

O escopo deste projeto envolve a criação de modelos de predição de tempestades geomagnéticas. Especificamente, o projeto visa:

1. **Análise Inicial dos Dados:** Descrever e explorar os dados brutos, identificar características importantes, e preparar os dados para o treinamento dos modelos.

2. **Pré-processamento de Dados:** Realizar a limpeza e pré-processamento dos dados, incluindo o tratamento de valores ausentes, a codificação de variáveis categóricas e a normalização, quando necessário.

3. **Desenvolvimento dos Modelos de Predição:** Selecionar e desenvolver um modelo de previsão apropriado para estimar DST, tendo em vista as 3 fases de uma tempestade geomagnética. Isso pode envolver o uso de técnicas de aprendizado de máquina, como regressão, séries temporais ou algoritmos de análise de texto, dependendo da natureza dos dados.

4. **Treinamento e Avaliação do Modelo:** Treinar o modelo com dados históricos e avaliá-lo usando métricas adequadas, como erro médio quadrático (RMSE).

5. **Análise de Resultados:** Comparar o desempenho dos modelos e entender seus pontos fortes e fracos.

6. **Conclusões**: Resumir os principais achados e próximos passos.

### Dados:

[![Static Badge](https://img.shields.io/badge/Dados%20brutos-Link-green?style=for-the-badge&logo=googlesheets)](https://drive.google.com/drive/folders/1S0WJQPltZ0j27zHfx-g2k7t6WiYeO8L-?usp=sharing)
[![Static Badge](https://img.shields.io/badge/Dados%20pre_processados-Link-green?style=for-the-badge&logo=googlesheets)](https://drive.google.com/drive/folders/1ug2s2DzdGoITI37N39TgTQlN0rowE6Uk?usp=drive_link)
[![Static Badge](https://img.shields.io/badge/Dicionário%20de%20Dados%20-%20PDF%20-%20red?style=for-the-badge&logo=files&logoColor=red
)](https://github.com/saraiva142/SkyNet/blob/main/dicion%C3%A1rio%20de%20dados%20-%20VII%20desafio%20CD%202024.pdf)  

### Codigo:
#ARRUMAR LINKS:
[![Static Badge](https://img.shields.io/badge/C%C3%B3digo%20do%20projeto-Link-orange?style=for-the-badge&logo=googlecolab)
](https://colab.research.google.com/drive/1uCbaxdK39zXcpc2FMXvMa06_0hzMAiBD?usp=sharing)  [![Static Badge](https://img.shields.io/badge/todos%20os%20algoritmos%20usados-Link-blue?style=for-the-badge)
](https://github.com/Joao-vpf/Vdesafiodedados/blob/main/files/Code/explicacao.md)

# 1 - **Análise Inicial dos Dados**

A análise inicial de dados é uma etapa onde buscamos compreender a estrutura dos dados e identificar quais variáveis têm maior impacto nas respostas que estamos tentando modelar. No nosso caso, o objetivo é entender como diferentes variáveis afetam o índice Dst, que está relacionado a tempestades geomagnéticas, e, a partir disso, descobrir os dados mais importantes para alimentar nossos modelos preditivos.

### 1.1 - Visualização e Estatísticas Descritivas

A seguir, foram geradas visualizações iniciais e estatísticas descritivas para obter uma visão geral da distribuição dos dados. A inspeção visual dos dados e o cálculo de estatísticas básicas como média, desvio padrão, valores mínimos e máximos também nos ajudaram a identificar possíveis padrões e anomalias nos dados, como variações inesperadas em determinadas variáveis que poderiam prejudicar a performance do modelo.

![image](https://github.com/user-attachments/assets/40d95a79-d2c7-4bc3-a3cf-405f8bf6a3e3)

![image](https://github.com/user-attachments/assets/d3ba8495-f717-42bb-8f7c-a6bc2dbf4e88)

A análise da distribuição do índice Dst, que é a variável alvo, revelou a concentração de valores em determinadas faixas e indicou a presença de outliers (eventos extremos). Esses outliers podem ser eventos extremos importantes para a modelagem de tempestades geomagnéticas.

![image](https://github.com/user-attachments/assets/a0d8ac24-2237-4d99-a7a2-1945dabbd381)

### 1.2 - Análise de Correlação

Uma das principais ferramentas na análise de dados é a correlação, que nos permite medir a força e a direção das relações entre as variáveis. Para isso, foi criada uma matriz de correlação para as colunas numéricas do conjunto de dados de vento solar (solar_wind.csv). Essa matriz é uma peça fundamental para identificar quais variáveis podem ser mais relevantes para a predição do índice Dst.

Além disso, foi feita uma análise específica da correlação entre as variáveis de vento solar e o índice Dst. Ao combinar as variáveis do vento solar com as informações do índice Dst, conseguimos ver diretamente quais delas têm uma relação mais forte com o Dst. Isso nos ajudará a focar nas features que mais impactam o fenômeno que estamos estudando — tempestades geomagnéticas.

![image](https://github.com/user-attachments/assets/98f28711-63b7-4e73-8888-4e4df6e26349)

Valores próximos de 1 ou -1 indicam uma correlação forte, onde 1 sugere uma relação positiva direta e -1 uma relação inversa. Ao identificar as variáveis cujos coeficientes de correlação com o Dst se aproximam de 1 ou -1, podemos determinar quais fatores do vento solar ou posição dos satélites têm maior influência sobre as tempestades geomagnéticas. Essas variáveis são consideradas as mais importantes para o nosso modelo de previsão, já que sua relação mais forte com o Dst sugere que podem ser preditores chave para a modelagem.

Para ilustrar melhor essas correlações, geramos o gráfico Correlação das variáveis com o índice Dst, que apresenta visualmente a força e o sentido da correlação de cada variável. Isso nos permite priorizar variáveis com maiores pesos na construção do modelo de predição, focando naquelas que têm impacto mais significativo sobre o índice Dst.

![image](https://github.com/user-attachments/assets/e0e36d92-05f2-4368-80a5-e6519f4626ca)


# 2. **Pré-processamento de Dados:**
