# Project machine-learning - Processo de linguagem natural

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://as1.ftcdn.net/v2/jpg/04/78/94/28/1000_F_478942813_slamkjSCj9FVoglaBVGCOxvjJwKjEjJw.jpg)


## Projeto

Projetos como análise de sentimento PLN, criação de API twitter nesse projeto foi criado projetos de estudo aplicar na área.

- **Projeto de Análise de sentimento**

- Análise texto dados da covid19 - KMEANS
- Análise de sentimento nas frases de feedback
- Análise de sentimento com a base tweets
- Análise de sentimento tweets da companhia United airlines
- Análise de sentimento frases com TextBlob


## Stack utilizada

**Front-end:** React, Redux, TailwindCSS

**Programação** Python

**Machine learning**: Scikit-learn

**Leitura CSV**: Pandas

**Análise de dados**: Seaborn, Matplotlib

**Modelo machine learning - Processo de linguagem natural**: NLTK, TextBlob


## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```
    
## Demo modelo PLN - Análise de sentimento 

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Carregando o dataset
  data = pd.read_csv("data.csv")
  
  # Visualizando os 5 primeiros itens
  data.head()

  # visualizando linhas e colunas com shape
  data.shape

  # Informações das variaveis
  data.info()

  # Treino e teste da base de dados para x e y
  x = df_train.iloc[:, 0: 10]
  y = df_train.iloc[:, 10]

  # Visualizando o shape da variavel x
  x.shape

  # Visualizando o shape da variavel y
  y.shape

  # Treinando modelo de machine learning
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

  # Visualizando linhas e colunas do dado de treino x_train
  x_train.shape

  # Visualizando linhas e colunas do dado de treino y_train
  y_train.shape

  # Modelo AutoML - AutoKeras
  # Modelo classificação com autokeras

  # Importando a biblioteca 
  import autokeras as ak
  
  # Modelo de classificação - max_trials tempo que vai ser treinado
  model = ak.StructuredDataClassifier(max_trials = 10)
  
  # Modelo vai treino - epochs sera as epocas do modelo
  model_fit = model.fit(x = x_train, y = y_train, epochs = 100)
  
  # Valição do modelo foi preparado
  model_automl_eval = model.evaluate(x = x_test, y = y_test)

  # Previsão do modelo
  predict = model.predict(model_predict)
  predict
```

## Dataset

| Dataset               | Link                                                |
| ----------------- | ---------------------------------------------------------------- |
|       |  |
|       |  |
|       |  |
|       |  |


## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com

