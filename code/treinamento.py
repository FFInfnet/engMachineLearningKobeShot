
from pycaret.classification import predict_model, get_config 
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import log_loss, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
import matplotlib.pyplot as plt
import personalTools as pTools
import pandas as pd
import pycaret.classification as pc 

dataset_path = f"../data/processed/"
datasedataset_name_train = "base_train.parquet"
dataset_train = f"{dataset_path}{datasedataset_name_train}" #f"{dataset_path}base_train_escalonado.parquet"
dataset_name_test = "base_test.parquet"
dataset_test = f"{dataset_path}{dataset_name_test}" #f"{dataset_path}base_test_escalonado.parquet"

df_treino = pd.read_parquet(dataset_train)
df_teste = pd.read_parquet(dataset_test)

mlflow.set_tracking_uri("http://localhost:5000")
nome_experimento = "Treinamento"
if not mlflow.get_experiment_by_name(nome_experimento):
    mlflow.create_experiment(nome_experimento)

# Configuração do PyCaret
setup_data = pc.setup(data=df_treino, test_data=df_teste, target='shot_made_flag', preprocess=False, normalize=False, session_id=123)
#list_models = setup_data.compare_models(['lr','dt'], n_select=2, sort='f1')
# Regressão Logística
lr_model = pc.create_model('lr')

# Árvore de Decisão
dt_model = pc.create_model('dt')

# Previsões e avaliações
lr_predictions = pc.predict_model(lr_model, data=df_teste)
lr_log_loss = log_loss(df_teste['shot_made_flag'], lr_predictions['prediction_score'])

dt_predictions = pc.predict_model(dt_model, data=df_teste)
dt_log_loss = log_loss(df_teste['shot_made_flag'], dt_predictions['prediction_score'])
dt_f1_score = f1_score(df_teste['shot_made_flag'], dt_predictions['prediction_label'].astype(int), average='binary')

# Iniciar MLflow e registrar métricas
mlflow.set_experiment(nome_experimento)
with mlflow.start_run():
    mlflow.log_metric("lr_log_loss", lr_log_loss)
    mlflow.log_metric("dt_log_loss", dt_log_loss)
    mlflow.log_metric("dt_f1_score", dt_f1_score)

    img_path = f"../output/img/"
    # criação da curva de validacao
    pTools.plot_parameter_validation_curve(df_treino.drop('shot_made_flag', axis=1),
                                        df_treino['shot_made_flag'],
                                        'C',
                                        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                                            lr_model, 'Regressão Logística', 'f1',
                                            True, f"{img_path}curva_LR.png")
    mlflow.log_artifact(f"{img_path}curva_LR.png")
    
    pTools.plot_parameter_validation_curve(df_treino.drop('shot_made_flag', axis=1),
                                        df_treino['shot_made_flag'],
                                        'max_depth', {'max_depth': [2, 3, 4, 5, 6, 7, 8]},
                                            dt_model, 'Árvore Decisão', 'f1', False,
                                            f"{img_path}curva_DT.png")
    mlflow.log_artifact(f"{img_path}curva_DT.png")



    img_confusion_path = f"{img_path}matriz_confusao_"
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Matriz de confusão para o modelo de Regressão Logística
    fig, ax = plt.subplots()
    cm = confusion_matrix(df_teste['shot_made_flag'], lr_predictions['prediction_label'])
    cmp = ConfusionMatrixDisplay(cm, display_labels=["Errou", "Acertou"])
    cmp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Matriz de Confusão Regressão Logística')
    plt.xlabel('Classificação')
    plt.ylabel('Verdade')
    plt.savefig(f"{img_confusion_path}lr.png")
    mlflow.log_artifact(f"{img_confusion_path}lr.png")

    # Matriz de confusão para o modelo de Árvore de Decisão
    fig, ax = plt.subplots()
    cm = confusion_matrix(df_teste['shot_made_flag'], dt_predictions['prediction_label'])
    cmp = ConfusionMatrixDisplay(cm, display_labels=["Errou", "Acertou"])
    cmp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Matriz de Confusão Árvore de Decisão')
    plt.xlabel('Classificação')
    plt.ylabel('Verdade')
    plt.savefig(f"{img_confusion_path}dt.png")
    mlflow.log_artifact(f"{img_confusion_path}dt.png")

    escolha_do_modelo = '''
    A escolha deve ser baseada em qual modelo apresenta o melhor equilíbrio entre essas métricas, priorizando um menor log_loss e um F1_score mais alto para um melhor desempenho geral.\n
    Se a Regressão Logística apresentar um log_loss significativamente menor, ela pode ser preferível pela sua capacidade de generalização. Por outro lado, se a Árvore de Decisão mostrar um F1_score muito superior e um log_loss competitivo, ela pode ser a melhor escolha devido à sua capacidade de lidar melhor com a complexidade dos dados.
    A decisão final deve ser tomada considerando o objetivo do modelo e o trade-off entre precisão e generalização.
    '''
    #---------------------------
    # curva ROC
    # Aplica os modelos no conjunto de teste para obter as probabilidades preditas
    lr_predicoes = predict_model(lr_model, data=df_teste.drop('shot_made_flag', axis=1))
    dt_predicoes = predict_model(dt_model, data=df_teste.drop('shot_made_flag', axis=1))

    # Extrai as probabilidades de ser a classe positiva (geralmente na coluna Label ou Score)
    lr_probs = lr_predicoes['prediction_score']
    dt_probs = dt_predicoes['prediction_score']

    # Extrai as verdadeiras etiquetas do conjunto de teste
    y_test = get_config('y_test')

    # Calcula a curva ROC e a área sob a curva (AUC)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_roc_auc = auc(lr_fpr, lr_tpr)

    tree_fpr, tree_tpr, _ = roc_curve(y_test, dt_probs)
    tree_roc_auc = auc(tree_fpr, tree_tpr)

    # Plotando a curva ROC
    plt.figure()
    lw = 2
    plt.plot(tree_fpr, tree_tpr, color='orange', lw=lw, label='Árvore de Decisão (AUC = %0.2f)' % tree_roc_auc)
    plt.plot(lr_fpr, lr_tpr, color='blue', lw=lw, label='Regressão Logística (AUC = %0.2f)' % lr_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Característica de Operação do Receptor (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('../output/img/teste_curva_roc.png', bbox_inches='tight')
    plt.close()

   # ---------------------------
   # FINALIZACAO MELHOR MODELO
    registered_model_name = 'model_kobe_shots'
    nexamples = 5
    model_version = -1
    tune_model = setup_data.tune_model(lr_model,
                                optimize = 'f1',
                                search_library = 'scikit-learn',
                                search_algorithm = 'random',
                                n_iter = 4)
    
    yhat_test = setup_data.predict_model(tune_model, raw_score=True)
    pd.set_option('display.max_columns', None)
    print(yhat_test.head())
    mlflow.log_metrics({
        'final_model_log_loss': log_loss(yhat_test.shot_made_flag, yhat_test.prediction_label),
        'final_model_f1': f1_score(yhat_test.shot_made_flag, yhat_test.prediction_label),
    })

    yhat_test.to_parquet('../data/processed/prediction_test.parquet')
    mlflow.log_artifact('../data/processed/prediction_test.parquet')

    final_model = setup_data.finalize_model(tune_model)

    # EXPORTACAO PARA LOG E REGISTRO DO MODELO
    setup_data.save_model(final_model, f'./{registered_model_name}') 
    # Carrega novamente o pipeline + bestmodel
    model_pipe = setup_data.load_model(f'./{registered_model_name}')
    # Assinatura do Modelo Inferida pelo MLFlow
    model_features = list(df_treino.drop('shot_made_flag', axis=1).columns)
    inf_signature = infer_signature(df_treino[model_features], 
                                    model_pipe.predict_proba(df_treino.drop('shot_made_flag', axis=1)))
    # Exemplo de entrada para o MLmodel
    input_example = {x: df_treino[x].values[:nexamples] for x in model_features}
    # Log do pipeline de modelagem do sklearn e registrar como uma nova versao
    mlflow.sklearn.log_model(
        sk_model=model_pipe,
        artifact_path="sklearn-model",
        registered_model_name=registered_model_name,
        signature = inf_signature,
        input_example = input_example,
        pyfunc_predict_fn='predict_proba'
    )
    # Criacao do cliente do servico MLFlow e atualizacao versao modelo
    client = MlflowClient()
    if model_version == -1:
        model_version = client.get_latest_versions(registered_model_name)[-1].version
    # Registrar o modelo como staging
    client.set_registered_model_alias(
        name    = registered_model_name, 
        alias   = "staging", 
        version = model_version
        )
