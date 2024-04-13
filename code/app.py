import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, roc_auc_score, precision_score, recall_score

def load_predict_production_dataset():    
    url = "http://127.0.0.1:5001/invocations"
    headers = {'Content-Type': 'application/json'}
    data = pd.read_parquet('../data/processed/data_prod_ﬁltered.parquet')
    input_data = data.drop('shot_made_flag', axis=1).to_dict(orient='list')
    json_data = json.dumps({"inputs": input_data})
    response = requests.post(url, data=json_data, headers=headers)
    if response.status_code != 200:
        print(f"Erro na API: {response.status_code} - {response.text}")
        return None
    predictions = pd.DataFrame(json.loads(response.content))
    df_aux = pd.DataFrame()
    df_aux['prediction_score_0'] = predictions['predictions'].apply(lambda x: x[0])
    df_aux['prediction_score_1'] = predictions['predictions'].apply(lambda x: x[1])
    df_aux['prediction_label'] = predictions['predictions'].apply(lambda x: 1 if x[1] > 0.5 else 0)
    data.reset_index(drop=True, inplace=True)
    df_aux.reset_index(drop=True, inplace=True)
    return pd.concat([data, df_aux], axis=1)

def load_data():
    dev_file = '../data/processed/prediction_test.parquet'
    df_prod = load_predict_production_dataset()
    df_test = pd.read_parquet(dev_file)
    return df_test, df_prod

def calculate_metrics(df):
    y_true = df['shot_made_flag']
    y_pred = df['prediction_label']
    y_proba = df['prediction_score_1']
    loss = log_loss(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return pd.DataFrame({
        'Metric': ['Log Loss', 'F1 Score', 'AUC', 'Precision', 'Recall'],
        'Value': [loss, f1, auc, precision, recall]
    })

def plot_metric_comparison(metrics_test, metrics_prod):
    metrics = metrics_test.set_index('Metric').join(metrics_prod.set_index('Metric'), lsuffix='_test', rsuffix='_prod')
    metrics.plot(kind='bar', figsize=(10, 5))
    plt.title('Comparação entre métricas de Teste e Produção')
    plt.ylabel('Score')
    plt.grid(True)
    st.pyplot(plt)

def main():
    st.title("Controle de Desvio de Dados de Saída de Modelos")
    tab1, tab2 = st.tabs(["Visualização de Métricas", "Predição Interativa"])

    with tab1:
        df_test, df_prod = load_data()
        if st.button("Calcular e Comparar Métricas"):
            metrics_test = calculate_metrics(df_test)
            metrics_prod = calculate_metrics(df_prod)
            st.write("### Métricas de Teste")
            st.table(metrics_test)
            st.write("### Métricas de Produção")
            st.table(metrics_prod)
            plot_metric_comparison(metrics_test, metrics_prod)

    with tab2:
        st.subheader("Insira os valores para predição")
        lat = st.number_input("Latitude", value=339.53, min_value=332.53, max_value=340.88, step=0.01)
        lon = st.number_input("Longitude", value=-1182627, min_value=-1185198, max_value=-1180218, step=1)
        minutes_remaining = st.slider("Minutos Restantes", min_value=0, max_value=11, value=5)
        period = st.slider("Período", min_value=1, max_value=7, value=3)
        playoffs = st.selectbox("Playoffs", options=[0, 1])
        shot_distance = st.slider("Distância do Lançamento", min_value=0, max_value=79, value=15)

        if st.button("Realizar Predição"):
            input_features = {'lat': lat, 'lon': lon, 'minutes_remaining': minutes_remaining,
                              'period': period, 'playoffs': playoffs, 'shot_distance': shot_distance}
            response = requests.post("http://127.0.0.1:5001/invocations", 
                                     json={"inputs": input_features}, 
                                     headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                prediction = response.json()['predictions'][0]
                predicted_class = 1 if prediction[1] > 0.5 else 0
                result_message = "Acertou o arremesso!" if predicted_class == 1 else "Errou o arremesso."
                st.write(f"Resultado da Predição: {result_message}")
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
