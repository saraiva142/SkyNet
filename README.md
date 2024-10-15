<h1 align="center">SKYNET - IIV DESAFIO DE CIÊNCIA DE DADOS 🤖 </h1>

## Equipe:
![image](https://github.com/user-attachments/assets/82ff4621-877c-4b38-aede-2cdea8b95dc0)

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


# 2. **Pré-processamento de Dados**

No processo de modelagem, o pré-processamento dos dados é uma etapa fundamental para garantir que o modelo receba entradas limpas e bem estruturadas. Ele envolve várias tarefas, como a limpeza de dados inconsistentes ou ausentes, a codificação de variáveis categóricas para permitir que modelos numéricos as interpretem, e a normalização das variáveis numéricas, ajustando-as para uma escala comum. Esse tratamento cuidadoso é crucial para melhorar a qualidade dos dados, reduzir viés e garantir que o modelo capture padrões relevantes de maneira eficaz.



#ESPERANDO RETORNO DA LUD



# 3. **Desenvolvimento dos Modelos de Predição**

Para o desenvolvimento dos modelos de predição do índice Dst, adotamos uma abordagem baseada em Redes Neurais Recorrentes (RNN) e Transformers, ambas técnicas avançadas e adequadas para modelagem de séries temporais. Como estamos lidando com dados temporais relacionados à atividade geomagnética, essas arquiteturas são particularmente eficazes para capturar a dinâmica complexa das tempestades geomagnéticas, divididas em três fases: início, pico, e recuperação.

As RNNs (Redes Neurais Recorrentes) são utilizadas para capturar as dependências temporais entre os eventos passados e presentes. Essa capacidade de "memória" permite que a rede aprenda padrões recorrentes nos dados de vento solar e atividade solar que influenciam o índice Dst ao longo do tempo.

O modelo de Transformer, por sua vez, supera algumas limitações das RNNs, como a dificuldade de capturar relações de longo prazo. Ele utiliza mecanismos de atenção, que identificam de forma mais eficiente quais partes da série temporal são mais relevantes para a predição em cada ponto. Isso é particularmente útil na modelagem de séries temporais complexas, onde o comportamento das variáveis pode mudar de forma imprevisível entre as fases da tempestade.

## 3.1 - Processamento e Arquitetura do Modelo RNN

O modelo RNN implementado neste projeto tem como objetivo prever o índice Dst a partir de dados relacionados ao vento solar. A arquitetura da rede neural é projetada para capturar padrões complexos nas séries temporais dos dados, utilizando uma série de camadas densas com funções de ativação e regularização.

Os dados de solar_wind e labels são carregados e mesclados com base nas colunas period e timedelta. As características relevantes (features) incluem colunas como bt, temperature, bx_gse, by_gse, entre outras, enquanto o alvo (target) é o índice dst. A normalização é aplicada aos dados de entrada usando o StandardScaler, o qual é uma técnica de normalização de dados utilizada no pré-processamento, onde transforma as características (features) de um conjunto de dados para que tenham média zero e desvio padrão um, garantindo que todos os recursos estejam na mesma escala. O cálculo da normalização desse é feito com cada dado subtraido da média de cada feature e dividindo pelo desvio padrão.

Tendo em vista a arquitetura do nosso RNN, esse modelo é construído utilizando a biblioteca TensorFlow e consiste em várias camadas densas, a primeira camada densa contém 512 neurônios com função de ativação ReLU e inclui regularização L2 para evitar overfitting, penalizando pesos grandes durante o treinamento, ao restringir a magnitude dos pesos, a regularização ajuda a generalizar melhor em novos dados. As camadas intermediárias Dense (256, 128, e 64 neurônios) (camadas adicionais) usam a função de ativação ReLU e seguem o mesmo padrão de regularização, elas ajudam a aprofundar a rede, permitindo que o modelo capture interações mais sutis entre as características, o aumento e a diminuição no número de neurônios também ajudam a criar um "funil", forçando a rede a extrair informações mais importantes à medida que avança nas camadas. As camadas Dropout ignora aleatoriamente 30% dos neurônios durante o treinamento, o que ajuda a reduzir o overfitting, e ao forçar a rede a não depender de neurônios específicos ela se torna mais robusta e capaz de generalizar melhor em dados não vistos. A camada final não tem função de ativação (ou pode ser uma função de ativação linear, dependendo do contexto), ela produz um único valor que representa a previsão do índice Dst.

O modelo é compilado utilizando o otimizador Nadam com uma taxa de aprendizado de 0.0001 e a função de perda mean squared error (RMSE), adequada para problemas. Os dados são divididos em conjuntos de treinamento e teste, com 80% dos dados utilizados para treinamento e 20% para teste. O treinamento do modelo é realizado por 100 épocas, utilizando um conjunto de validação que representa 20% dos dados de treinamento. O treinamento é interrompido antecipadamente se a perda de validação não melhorar por 10 épocas consecutivas. Durante o treinamento, a curva de perda é plotada para visualizar a convergência do modelo.

![image](https://github.com/user-attachments/assets/0743b4e5-c128-45f3-b746-402f698c7b6d)

Após o treinamento, o modelo é avaliado utilizando o conjunto de teste. O erro quadrático médio (RMSE) é calculado para quantificar a precisão das previsões. Um gráfico adicional compara os valores reais e preditos, permitindo uma visualização clara da eficácia do modelo na predição do índice Dst.

![image](https://github.com/user-attachments/assets/da91ab52-2576-48ba-8dbc-aec8460c27bb)



