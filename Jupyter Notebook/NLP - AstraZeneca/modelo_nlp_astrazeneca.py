# -*- coding: utf-8 -*-
"""Modelo NLP - AstraZeneca.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ov6R2LxNDnKx2TMCnMLtAwLYxWHMSwjp

# **PLN - Modelo de processo de linguagêm natural**

**Análise de sentimento tweets - Vacina AstraZeneca**
"""

# Carregando pacotes
!pip install watermark

# Versão do python
from platform import python_version

print('Versão Jupyter Notebook neste projeto:', python_version())

#Importação das bibliotecas

# Bibliotecas para NLTK
import nltk
import re
import wordcloud
import itertools
from wordcloud import WordCloud

import pandas as pd # Carregamento de arquivos de csv
import numpy as np # Carregamento cálculos em arrays multidimensionais

# Bibliotecas de visualização
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar as versões das bibliotecas
import watermark

# Warnings retirar alertas 
import warnings
warnings.filterwarnings("ignore")

# Baixando pacote do punkt

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Commented out IPython magic to ensure Python compatibility.
# Verficações da versões das bibliotecas

# %reload_ext watermark
# %watermark -a "Rafael Gallo" --iversions

# Configuração fundo dos gráficos e estilo, tamanho da fonte

sns.set_palette("Accent")
sns.set(style="whitegrid", color_codes=True, font_scale=1.3)
color = sns.color_palette()

"""# **Base dados**"""

# Carregando a base de dados
df = pd.read_csv("Vaccine Tweets-AstraZeneca.csv")

# Exebindo o 5 primeiro dados 
df.head(5)

# Exebindo o 5 últimos dados
df.tail(5)

# Número de linhas e colunas 
df.shape

# Exibido os tipos de dados
df.dtypes

# Informando as informações e das variaveis 
df.info()

# Total de colunas e linhas 

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Exibindo valores ausentes e Valores únicos

print("\nMissing values :  ", df.isnull().sum().values.sum())
print("\nUnique values :  \n",df.nunique())

# Polaridade do coluna 
df.Polarity

# Contando números de dados
df.Polarity.value_counts()

# Total de número duplicados
df.duplicated()

# Variação imparcial
df.var()

# Contagem de dados da coluna account_length

df.groupby(['Subjectivity'])['Polarity'].count()

# Renomeando as colunas do dataset

df.columns = ["Usuario",
              "Text",
              "Subjetividade",
              "Polaridade",
              "Sentimento"]
df.head()

# Contagem de dados da coluna na Sentimento
df.Sentimento.count()

# Contagem de dados da coluna na Subjetividade
df.Subjetividade.count()

# Contagem de dados da coluna na Polaridade
df.Polaridade.count()

# Contagem de dados da coluna na texto
df.Text.count()

# Textos duplicados total

df.drop_duplicates(["Text"], inplace = True)
df.Text.count()

"""# **Análise de dados**"""

# Gráfico barras de sentimento
plt.figure(figsize=(12.8,6))

ax = sns.countplot(df["Sentimento"])
plt.title("Análise de sentimento")
plt.xlabel("Sentimentos")
plt.ylabel("Total de sentimentos")
plt.show()

# Gráfico de scatterplot 
plt.figure(figsize=(12.8,6))

ax = sns.scatterplot(x="Subjetividade", y="Polaridade", data=df, hue="Sentimento")
plt.title("Polaridade das frases")
plt.ylabel("Total")
plt.xlabel("Polaridade e Subjetividade")

# Gráfico de boxplots - Verificando os dados no boxplot valor total verificando possíveis outliers
plt.figure(figsize=(12.8,6))

ax = sns.boxplot(x="Subjetividade", y="Sentimento", data = df)
plt.title("Gráfico sentimentos")
plt.xlabel("Sentimentos frases")
plt.ylabel("Total")

# Nuvem de palavras
words = ' '.join([tweet for tweet in df['Text']])
wordCloud = WordCloud(width=600, height=400).generate(words)

plt.figure(figsize=(18.8, 16))
plt.imshow(wordCloud)
plt.show()

"""# **Treino teste**
- Treino e teste da base de dados da colunas textos e sentimento
"""

train = df["Text"] # Variável para treino
test = df["Sentimento"] # Variável para teste

# Total de linhas e colunas dados variável x
train.shape

# Total de linhas e colunas dados variável y

test.shape

"""# **Pré-processamento**"""

# Dados de limpeza para modelo PLN

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Remove stop words: Removendo as stop words na base de dados
def remove_stop_words(instancia): # Removendo as stop words
    stopwords = set(nltk.corpus.stopwords.words("english"))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

# Palavras derivacionalmente relacionadas com significados semelhantes, palavras para retornar documentos que contenham outra palavra no conjunto.
def text_stemming(instancia):
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for w in instancia.split():
      palavras.append(stemmer.stem(w))
      return (" ".join(palavras))

# Limpeza na base de dados limpando dados de web com http e outros.
def dados_limpos(instancia): 
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instancia)

#Lemmatization: Em linguística é o processo de agrupar as formas flexionadas de uma palavra para que possam ser analisadas como um único item, identificado pelo lema da palavra , ou forma de dicionário.
def Lemmatization(instancia):
    palavras = []
    for w in instancia.split():
        palavras.append(wordnet_lemmatizer.lemmatize(w))
        return (" ".join(palavras))

# Preprocessing: Pré - processamento da base de dados que serão ser para análise de dados.
def Preprocessing(instancia):
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','').replace('"','')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

# Negações do texto
def neg(text):
    neg = ["não", "not"]
    neg_dect = False
    
    result = []
    pal = text.split()

    for x in pal:
        x = x.lower()
        if neg_dect == True:
            x = x + "_NEG"
        if x in neg:
            neg_dect = True
        result.append(x)

    return ("".join(result))

# Base dados limpo
train = [Preprocessing(i) for i in train]
train[:50]

# Tokenização as palavras precisam ser codificadas como inteiros, 
# Ou valores de ponto flutuante, para serem usadas como entradas para modelos machine learning.
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

tokenizer = TweetTokenizer()
vectorizer = CountVectorizer(analyzer="word", tokenizer = tokenizer.tokenize)
freq = vectorizer.fit_transform(train)
freq
freq.shape

"""# **Modelo machine learning**

- Modelo 01: Regressão logistica
"""

# Modelo de regressão logistica 

# Importação da biblioteca
from sklearn.linear_model import LogisticRegression

# Nome do algoritmo M.L
model_logistic = LogisticRegression() 

# Treinamento do modelo
model_logistic_fit = model_logistic.fit(vet_train, test)

# Score do modelo dados treino x
model_logistic_score = model_logistic.score(vet_train, test)

# Score do modelo dados treino y
print("Model - Logistic Regression: %.2f" % (model_logistic_score * 100))

# Previsão modelo com função predict de previsã das frases

model_logistic_pred = model_logistic.predict(vet_train)
model_logistic_pred

# Previsão modelo com função log_proba de probabilidades das frases

model_logistic_prob = model_logistic.predict_log_proba(vet_train)
model_logistic_prob

# Acúracia do modelo de Regressão logística
from sklearn import metrics
from sklearn.metrics import accuracy_score

accuracy_dt = accuracy_score(test, model_logistic_pred)
print("Acurácia - Regressão logística: %.2f" % (accuracy_dt * 100))

from sklearn.metrics import confusion_matrix

matrix_1 = confusion_matrix(model_logistic_pred, test)
matrix_1

# Classification report

from sklearn.metrics import classification_report

classification = classification_report(model_logistic_pred, test)
print("Modelo - Regressão logística")
print()
print(classification)

# Plot matriz de confusão
plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_1, annot=True, ax = ax, fmt = ".1f", cmap="Paired"); 
ax.set_title('Confusion Matrix - Regressão logística'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]);

"""# **Modelo 02 - Naive bayes**"""

# Modelo machine learning - Naive bayes

# Importação da biblioteca
from sklearn.naive_bayes import MultinomialNB

# Nome do algoritmo M.L
model_naive_bayes = MultinomialNB()

# Treinamento do modelo
model_naive_bayes_fit = model_naive_bayes.fit(vet_train, test)

# Score do modelo dados treino x
model_naive_bayes_scor = model_naive_bayes.score(vet_train, test)

 # Score do modelo dados treino y
print("Model - Naive Bayes: %.2f" % (model_naive_bayes_scor * 100))

# Previsão modelo com função predict de previsã das frases

model_naive_bayes_pred = model_naive_bayes.predict(vet_train)
model_naive_bayes_pred

# Previsão modelo com função log_proba de probabilidades das frases

model_naive_bayes_prob = model_naive_bayes.predict_proba(vet_train).round(2)
print(model_naive_bayes_prob)

# Acúracia do modelo de Naive bayes
accuracy_naive_bayes = metrics.accuracy_score(test, model_naive_bayes_pred)

print("Accuracy model Naive bayes: %.2f" % (accuracy_naive_bayes * 100))

# Confusion matrix
matrix_2 = confusion_matrix(model_naive_bayes_pred, test)
matrix_2

# Classification report
classification = classification_report(model_naive_bayes_pred, test)
print("Modelo - Naive bayes")
print()
print(classification)

# Plot confusion matrix
plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_2, annot=True, ax = ax, fmt = ".1f", cmap="Paired"); 
ax.set_title('Confusion Matrix - Naive bayes'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]);

"""# **Pipeline 1 - Regressão logística**"""

# Função para texto de negações
def marque_negacao(texto):
    
    # Negaçoes do texto mudando para not para "não"
    negacoes = ['não','not']
    negacao_detectada = False
    
    # Criando uma lista vazia 
    resultado = []
    palavras = texto.split()
    
    # For em palavras para os dados de negações 
    for p in palavras:
        p = p.lower()
        if negacao_detectada == True:
            p = p + '_NEG'
        if p in negacoes:
            negacao_detectada = True
        resultado.append(p)
    
    # Retornando a função
    return (" ".join(resultado))

# Importando bibliotecas do pipeline
from sklearn import svm
from sklearn.pipeline import Pipeline

# Pipeline modelo regressão logística
model_reg_log = Pipeline([
    ('counts', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# Treinamento do pipeline 
model_reg_log.fit(train, test)

# Pipeline simples 
model_reg_log_simples = Pipeline([
  ('counts', CountVectorizer()),
  ('classifier', LogisticRegression())
])

# Treinamento do pipeline
model_reg_log_simples.fit(train, test)

# Pipeline para negações
model_reg_log_negacoes = Pipeline([
  ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
  ('classifier', LogisticRegression())
])
# Treinamento do pipeline
model_reg_log_negacoes.fit(train, test)

# Validação cruzada do modelo
validacao_cruzada_Reg = cross_val_predict(model_reg_log, train, test)
validacao_cruzada_Reg

# Acúracia do modelo do pipeline regressão logística
accuracy_1_rg = metrics.accuracy_score(test, validacao_cruzada_Reg)

print("Accuracy pipeline 1 Logistic Regression: %.2f" % (accuracy_1_rg * 100))

# Classification report do pipeline 
classification = classification_report(validacao_cruzada_Reg, test)
print("Modelo - Pipeline 1 regressão logística")
print()
print(classification)

# Confusion matrix pipeline regressão logística
matrix_3 = confusion_matrix(validacao_cruzada_Reg, test)
matrix_3

# Matriz total de sentimentos

sentimento=['Positivo',
            'Negativo',
            'Neutro']

print(pd.crosstab(test, validacao_cruzada_Reg, rownames = ["Real"], colnames=["Predito"], margins = True))

# Plot confusion matrix
plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_3, annot=True, ax = ax, fmt = ".1f", cmap="Paired"); 
ax.set_title('Confusion Matrix - Pipeline 1 regressão logística'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]);

"""# **Pipeline 2 - Naive bayes**"""

# Pipeline simples naive bayes
model_pipeline_simples_2 = Pipeline([
  ('counts', CountVectorizer()),
  ('classifier', MultinomialNB())
])

# Treinamento do pipeline
model_pipeline_simples_2.fit(train, test)

# Pipeline negações
model_pipeline_negacoes_2 = Pipeline([
  ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
  ('classifier', MultinomialNB())
])

# Pipeline treinamento
model_pipeline_negacoes_2.fit(train, test)

# Pipeline SVM simples
model_pipeline_svm_simples_2 = Pipeline([
    ("counts", CountVectorizer()),
    ("classifier", svm.SVC(kernel = "linear"))
])
# Treinamento pipeline
model_pipeline_svm_simples_2.fit(train, test)

# Pipeline SVM para negacoes
model_pipeline_svm_negacoes_2 = Pipeline([
    ("counts", CountVectorizer(tokenizer = lambda text: marque_negacao(text))),
    ("classifier", svm.SVC(kernel = "linear"))
])

# Treinamento do pipeline
model_pipeline_svm_negacoes_2.fit(train, test)

# Validação cruzada pipeline naive bayes
validacao_cruzada_2 = cross_val_predict(model_pipeline_simples_2, train, test)
validacao_cruzada_2

# Acúracia do modelo do pipeline naive bayes
accuracy_pipeline_2_nb = metrics.accuracy_score(test, validacao_cruzada_2)
print("Accuracy pipeline 2 - Naive bayes: %.2f" % (accuracy_pipeline_2_nb * 100))

# Classification report do pipeline 2 
classification = classification_report(validacao_cruzada_2, test)
print("Modelo - Pipeline 2 naive bayes")
print()
print(classification)

# Matriz total de sentimentos
sentimento=['Positivo',
            'Negativo',
            'Neutro']

print(pd.crosstab(test, 
                  validacao_cruzada_2, 
                  rownames = ["Real"], 
                  colnames=["Predito"], 
                  margins = True))

# Confusion matrix pipeline 2 naive bayes
matrix_4 = confusion_matrix(validacao_cruzada_2, test)
matrix_4

# Plot confusion matrix
plt.figure(figsize=(15, 8))

ax = plt.subplot()
sns.heatmap(matrix_4, annot=True, ax = ax, fmt = ".1f", cmap="Paired"); 
ax.set_title('Confusion Matrix - Pipeline 1 regressão logística'); 
ax.xaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]); ax.yaxis.set_ticklabels(["Positivo", "Negativo", "Neutro"]);

# Métricas do modelos - Naive Nayes e regressão logística
def metricas_pipeline(model_naive_bayes, train, test):
    validacao_cruzada = cross_val_predict(model_naive_bayes, train, test, cv = 10)
    return "Acurácia do modelo: {}".format(metrics.accuracy_score(validacao_cruzada, test))

def metricas_pipeline(model_logistic, train, test):
    validacao_cruzada_Reg = cross_val_predict(model_logistic, train, test, cv = 10)
    return "Acurácia do modelo: {}".format(metrics.accuracy_score(validacao_cruzada_Reg, test))

print("Pipeline 1 - Naive Bayes")
print()
print("Model pipeline Naive Bayes Simples:", metricas_pipeline(model_pipeline_simples, train, test))
print("Model pipeline Naive Bayes negações:", metricas_pipeline(model_pipeline_negacoes, train, test))
print("Model pipeline SVM simples:", metricas_pipeline(model_pipeline_svm_simples, train, test))
print("Model pipeline SVM negacoes:", metricas_pipeline(model_pipeline_svm_negacoes, train, test))
print()
print("Pipeline 2 - Regressão Logística")
print()
print("Model pipeline Simples:", metricas_pipeline(model_reg_log, train, test))
print("Model pipeline negações:", metricas_pipeline(model_reg_log_simples, train, test))
print("Model pipeline SVM simples:", metricas_pipeline(model_reg_log_negacoes, train, test))

# Resultados - Modelos machine learning

modelos = pd.DataFrame({
    
    "Models" :["Pipeline 1 - Regressão logistica", 
               "Pipeline 2 - Naive Bayes"],

    "Acurácia" :[accuracy_pipeline_2_nb,
                 accuracy_1_rg]})

modelos.sort_values(by = "Acurácia", ascending = False)

## Salvando modelo M.L

import pickle
 
with open('model_logistic_pred.pkl', 'wb') as file:
    pickle.dump(model_logistic_pred, file)
    
with open('model_naive_bayes_pred.pkl', 'wb') as file:
    pickle.dump(model_naive_bayes_pred, file)

