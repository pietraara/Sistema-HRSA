import pandas as pd
import seaborn as sns
import re 
import os
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import numpy as np
import time
import requests
import json
from unidecode import unidecode
import unicodedata
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import ttest_ind, f_oneway
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import networkx as nx
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, BatchNormalization, Dropout
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Point

# Título do Aplicativo
st.title("Sistema HRSA - Ágio Interno e Empresa-Veículo")

# 1. Leitura da Base
st.header("1. Leitura da Base")
uploaded_file = st.file_uploader("Faça o upload do arquivo Excel com a base de dados:")
if uploaded_file:
    sheet_name = st.text_input("Digite o nome da planilha no arquivo:", value="Ágio")
    casos = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    st.dataframe(casos.head())

    # 2. Tratamento da Base
    st.header("2. Tratamento da Base")
    with st.spinner("Realizando tratamento de dados..."):
        # Conversões de formato datetime
        date_columns = ['Data Liminar/Tutela', 'Sentença', 'Acórdão']
        for col in date_columns:
            if col in casos.columns:
                casos[col] = pd.to_datetime(casos[col], errors='coerce')

        # Tratamento de valores inconsistentes e preenchimento de nulos
        valores_inconsistentes = ["_", "N/A", None, np.nan]
        casos.replace(valores_inconsistentes, np.nan, inplace=True)
        casos.fillna("Vazio", inplace=True)

        # Normalização de colunas
        if 'Número do Processo' in casos.columns:
            casos['Número do Processo'] = casos['Número do Processo'].str.replace(r'[.-]', '', regex=True)

        if 'Cidade' in casos.columns:
            casos['comarca'] = casos['Cidade'].str.replace(r' - \w{2}$', '', regex=True)

        st.success("Tratamento concluído!")
        st.dataframe(casos.head())

    # 3. Feature Engineering: Duração dos Processos
    st.header("3. Feature Engineering: Duração dos Processos")
    def extrair_ano(num_proc):
        try:
            ano = int(num_proc[9:13])
            return ano if 1900 <= ano <= 2100 else np.nan
        except (ValueError, TypeError):
            return np.nan

    if 'Número do Processo' in casos.columns:
        casos['ano_processo'] = casos['Número do Processo'].apply(extrair_ano)
        casos['data_presumida'] = pd.to_datetime(casos['ano_processo'], format='%Y', errors='coerce')

    if 'Data Liminar/Tutela' in casos.columns:
        casos['tempo_meses'] = (casos['Data Liminar/Tutela'] - casos['data_presumida']).dt.days / 30.44

    st.write("Duração dos processos calculada:")
    st.dataframe(casos[['Número do Processo', 'data_presumida', 'Data Liminar/Tutela', 'tempo_meses']].head())

    # 4. Base Externa: Novas Colunas
    st.header("4. Base Externa: Novas Colunas")
    url_raw = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json"
    comarcas = gpd.read_file(url_raw)
    comarcas = comarcas.rename(columns={"name": "comarca"})
    comarcas['latitude'] = comarcas['geometry'].centroid.y
    comarcas['longitude'] = comarcas['geometry'].centroid.x

    casos['comarca'] = casos['comarca'].apply(lambda x: unidecode(x).upper() if isinstance(x, str) else x)
    comarcas['comarca'] = comarcas['comarca'].apply(lambda x: unidecode(x).upper() if isinstance(x, str) else x)

    casos = casos.merge(comarcas, on="comarca", how="left")
    st.write("Base externa integrada:")
    st.dataframe(casos.head())

    # 5. Análises e Visualizações
    st.header("5. Análises e Visualizações")

    # Estatísticas Descritivas
    st.subheader("Estatísticas Descritivas")
    casos_cleaned = casos.replace(["Vazio", "VAZIO"], np.nan).dropna(how="all")

    def replace_nan_with_vazio(df):
        return df.fillna("Vazio")

    def map_categorical_to_numeric(data, column, mapping):
        if column in data.columns:
            data[column] = data[column].map(mapping)
        return data

    mapping = {"Favorável": 1, "Parcialmente": 0.5, "Desfavorável": 0}
    casos_cleaned = map_categorical_to_numeric(casos_cleaned, "Mérito Liminar/Tutela", mapping)
    casos_cleaned = map_categorical_to_numeric(casos_cleaned, "Mérito na Sentença", mapping)
    casos_cleaned = map_categorical_to_numeric(casos_cleaned, "Mérito no Acórdão", mapping)

    mapping = {"Sim": 1, "Não": 0}
    casos_cleaned = map_categorical_to_numeric(casos_cleaned, "Empresa-Veículo", mapping)

    # Gráficos de Barras
    st.subheader("Gráficos de Barras")
    contribuinte_full = replace_nan_with_vazio(
        casos_cleaned["Contribuinte"].value_counts().reset_index(name="Frequência").rename(columns={"index": "Contribuinte"})
    )

    top_contribuintes = contribuinte_full.sort_values(by="Frequência", ascending=False)
    fig, ax = plt.subplots()
    ax.bar(top_contribuintes["Contribuinte"], top_contribuintes["Frequência"], color="skyblue")
    ax.set_title("Top Contribuintes")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    natureza_full = replace_nan_with_vazio(
        casos_cleaned["Natureza"].value_counts().reset_index(name="Frequência").rename(columns={"index": "Natureza"})
    )

    top_naturezas = natureza_full.sort_values(by="Frequência", ascending=False)
    fig, ax = plt.subplots()
    ax.bar(top_naturezas["Natureza"], top_naturezas["Frequência"], color="skyblue")
    ax.set_title("Top Naturezas")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Gráficos de Pizza
    st.subheader("Gráficos de Pizza")
    for col, title in [
        ("Mérito Liminar/Tutela", "Distribuição do Mérito Liminar/Tutela"),
        ("Mérito na Sentença", "Distribuição do Mérito na Sentença"),
        ("Mérito no Acórdão", "Distribuição do Mérito no Acórdão"),
        ("Empresa-Veículo", "Distribuição de Empresa-Veículo"),
    ]:
        if col in casos_cleaned.columns:
            counts = casos_cleaned[col].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(title)
            st.pyplot(fig)

    # Gráfico de Linha
    st.subheader("Gráfico de Linha")
    tempo_por_comarca = casos.groupby("comarca")["tempo_meses"].mean().sort_values()
    fig, ax = plt.subplots()
    tempo_por_comarca.plot(kind="line", marker="o", ax=ax, color="skyblue")
    ax.set_title("Tempo Médio de Processos por Comarca")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Mapas Interativos
    st.subheader("Mapas Interativos Scatter Mapbox")
    for col, title in [
        ("Mérito Liminar/Tutela", "Distribuição de Casos por Mérito Liminar/Tutela"),
        ("Mérito na Sentença", "Distribuição de Casos por Mérito na Sentença"),
        ("Mérito no Acórdão", "Distribuição de Casos por Mérito no Acórdão"),
        ("Empresa-Veículo", "Distribuição de Casos por Empresa-Veículo"),
    ]:
        if col in casos.columns and 'latitude' in casos.columns and 'longitude' in casos.columns:
            fig = px.scatter_mapbox(
                casos,
                lat="latitude",
                lon="longitude",
                color=col,
                size="aparições_comarca" if 'aparições_comarca' in casos.columns else None,
                hover_name="comarca",
                title=title,
                mapbox_style="open-street-map",
                zoom=4
            )
            st.plotly_chart(fig)

    st.success("Análises concluídas e apresentadas com sucesso!")