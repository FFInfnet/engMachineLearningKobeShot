import pandas as pd
import requests
import json
from sklearn.metrics import log_loss, f1_score
from sklearn import metrics
import mlflow
import os
import personalTools as pTools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configurar MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Modifique conforme necessário
nome_experimento = "PipelineAplicacao"
mlflow.set_experiment(nome_experimento)

# Iniciar uma execução no MLflow
if not mlflow.get_experiment_by_name(nome_experimento):
    mlflow.create_experiment(nome_experimento)

with mlflow.start_run(run_name=nome_experimento):
    # Endpoint do modelo
    url = "http://127.0.0.1:5001/invocations"
    headers = {'Content-Type': 'application/json'}

    # Carregar dados
    data = pd.read_parquet('../data/processed/data_prod_ﬁltered.parquet')
    dir_prediction = '../data/prediction/'

    print('Informações sobre o dataset de producao')
    pTools.var_text_to_file(str(pTools.info_to_string(data)), f'{dir_prediction}dataset_info.txt')
    pTools.var_text_to_file(str(data.describe()), f'{dir_prediction}dataset_describe.txt')
    mlflow.log_artifact(f'{dir_prediction}dataset_info.txt')
    mlflow.log_artifact(f'{dir_prediction}dataset_describe.txt')

    # Fazer previsões
    input_data = data.drop('shot_made_flag', axis=1).to_dict(orient='list')
    json_data = json.dumps({"inputs": input_data})  # Convertendo em JSON
    
    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code != 200:
        print(f"Erro na API: {response.status_code} - {response.text}")
        exit()

    print("Previsões recebidas com sucesso.")
    predictions = pd.DataFrame(json.loads(response.text))

    predictions['probability'] = predictions['predictions'].apply(lambda x: x[1])
    predictions['prediction_score_0'] = predictions['predictions'].apply(lambda x: x[0])
    predictions['prediction_score_1'] = predictions['predictions'].apply(lambda x: x[1])
    predictions['shot_made_flag'] = data['shot_made_flag']

    # Calcular métricas
    loss = log_loss(data['shot_made_flag'], predictions['probability'])
    f1 = f1_score(data['shot_made_flag'], predictions['predictions'].apply(lambda x: 1 if x[1] > 0.5 else 0), average='binary')

    # Logar métricas
    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("f1_score", f1)

    # Salvar previsões como um artefato
    if not os.path.exists(dir_prediction):
        os.makedirs(dir_prediction)
    predictions.to_parquet(f'{dir_prediction}predictions.parquet')
    mlflow.log_artifact(f'{dir_prediction}predictions.parquet')
    predictions.to_csv(f'{dir_prediction}predictions.csv')
    mlflow.log_artifact(f'{dir_prediction}predictions.csv')

    dir_txt_prediction = '../output/prediction/'
    if not os.path.exists(dir_txt_prediction):
        os.makedirs(dir_txt_prediction)
    pTools.var_text_to_file(str(pTools.info_to_string(predictions)), f'{dir_txt_prediction}info.txt')
    mlflow.log_artifact(f'{dir_txt_prediction}info.txt')
    pTools.var_text_to_file(str(predictions.describe(include='all')), f'{dir_txt_prediction}describe.txt')
    mlflow.log_artifact(f'{dir_txt_prediction}describe.txt')
    
    # Verificar a saída dos arquivos e métricas
    print("Previsões e métricas registradas com sucesso.")

    # verificando o desempenho
    print("Verificando o desempenho")
    metricas = metrics.classification_report(data['shot_made_flag'], predictions['predictions'].apply(lambda x: 1 if x[1] > 0.5 else 0))
    print(metricas)
    pTools.var_text_to_file(metricas, f'{dir_txt_prediction}metrics.txt')
    mlflow.log_artifact(f'{dir_txt_prediction}metrics.txt')

    fig, ax = plt.subplots()
    cm = confusion_matrix(data['shot_made_flag'], predictions['predictions'].apply(lambda x: 1 if x[1] > 0.5 else 0))
    cmp = ConfusionMatrixDisplay(cm, display_labels=["Errou", "Acertou"])
    cmp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Matriz de Confusão Regressão Logística PRODUÇÃO')
    plt.xlabel('Classificação')
    plt.ylabel('Verdade')
    plt.savefig(f"{dir_txt_prediction}lr_producao_confusion_matrix.png")
    mlflow.log_artifact(f"{dir_txt_prediction}lr_producao_confusion_matrix.png")

    # Contagem de valores na coluna 'shot_made_flag'
    contagem_valores_shot_made_flag = data['shot_made_flag'].value_counts()

    # Exibir a contagem de valores
    print("\nContagem de valores na coluna 'Qualishot_made_flag':")
    print(contagem_valores_shot_made_flag)
