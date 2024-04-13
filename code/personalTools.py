import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import validation_curve
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow

# define função de plotagem de valores minimos e maximos
# pode ser usada para comparação da amplitude dos dados e dados escalonados
def plotagemMaxMin(df_original, titulo, output_img_file):
    valores_min = df_original.min()
    valores_max = df_original.max()

    ls_min_max = [{x : [valores_min.loc[x], valores_max.loc[x]]} for x in df_original.columns]
    df_min_max = pd.DataFrame({list(item.keys())[0]: list(item.values())[0] for item in ls_min_max})
    print(df_min_max.head())
    df_min_max.plot(kind='bar', figsize=(10, 6))
    plt.title(titulo)
    plt.ylabel('Valores (faixa dinâmica)')
    plt.xlabel('Colunas (variáveis analisadas)')
    plt.savefig(output_img_file, bbox_inches='tight')
  
def save_info_plot(df, file_path):
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    info_lines = info_str.split('\n')
    info_str = '\n'.join(reversed(info_lines))

    plt.figure(figsize=(10, 8))
    plt.text(0.01, 0.99, info_str, fontfamily='monospace', fontsize=10, verticalalignment='top', horizontalalignment='left')
    plt.axis('off')

    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def save_describe_plot(df, file_path):
    info_str = df.describe().to_string() 

    plt.figure(figsize=(10, 8))
    plt.text(0.01, 0.99, info_str, fontfamily='monospace', fontsize=10, verticalalignment='top', horizontalalignment='left')
    plt.axis('off')

    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def analyze_dataframe(df, tipo, file_path):
    with open(file_path, 'w') as f:
        impressao = f"Descrição do Dataset ***({tipo})***:"
        print(impressao)
        f.write(impressao+"\n")
        impressao = str(df.describe(include='all'))
        print(impressao)
        f.write(impressao+"\n\n")
        
        impressao = f"Total de linhas: {df.shape[0]}"
        print(impressao)
        f.write(impressao+"\n")
        impressao = f"Total de colunas: {df.shape[1]}"
        print(impressao)
        f.write(impressao+"\n\n")
        
        missing_data = df.isnull().sum()
        impressao = "Dados Faltantes por Coluna:"
        print(impressao)
        f.write(impressao+"\n")
        impressao = str(missing_data[missing_data > 0])
        print(impressao)
        f.write(impressao+"\n\n")
        
        impressao = "Recomendação de Codificação para Variáveis Categóricas e Valores Únicos:"
        print(impressao)
        f.write(impressao+"\n")
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            unique_values = df[col].unique()
            encoding_type = "One-Hot Encoding" if len(unique_values) <= 10 else "Label Encoding"
            impressao = f"Coluna: {col} - Codificação Recomendada: {encoding_type}"
            f.write(impressao+"\n")
            impressao = f"Valores Únicos: {unique_values}"
            f.write(impressao+"\n\n")

def var_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        print(text)
        f.write(text+"\n")

def std_scaler_dataset(df, drop_columns):
    df_escalonado = df.drop(drop_columns, axis=1)
    from sklearn.preprocessing import StandardScaler
    escalonando = StandardScaler()
    escalonado = escalonando.fit_transform(df_escalonado)
    df_escalonado = pd.DataFrame(escalonado, columns=df_escalonado.columns, index=df.index)
    return df_escalonado

# plotar a curva de validação
def plot_parameter_validation_curve(X, Y, param_name, grid_search,
                                    model, model_name, scoring,
                                    logx, file_path):
    print('Parameter:', param_name)
    print('GridSearch:', grid_search[param_name])
    print('Scoring:', scoring)
    plt.figure(figsize=(6,4))
    train_scores, test_scores = validation_curve(model,
                                                 X = X, 
                                                 y = Y, 
                                                 param_name=param_name, 
                                                 param_range= grid_search[param_name],
                                                 scoring=scoring,
                                                 cv=10,
                                                 n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Curva Validação Modelo " + model_name)
    plt.xlabel(param_name)
    plt.ylabel("Score ("+scoring+")")
    if logx:
        plt.semilogx(grid_search[param_name], train_scores_mean,'-o', label="Treino",
                     color="darkorange", lw=2)
        plt.semilogx(grid_search[param_name], test_scores_mean,'-o', label="Validação-Cruzada",
                     color="navy", lw=2)
    else:
        plt.plot(grid_search[param_name], train_scores_mean,'-o', label="Treino",
                     color="darkorange", lw=2)
        plt.plot(grid_search[param_name], test_scores_mean,'-o', label="Validação-Cruzada",
                 color="navy", lw=2)
    plt.fill_between(grid_search[param_name], train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)
    plt.fill_between(grid_search[param_name], test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

# Função para registrar ou atualizar o modelo no MLflow Model Registry
def register_or_update_model(model, model_name):
    model_uri = f"runs:/{run.info.run_id}/{model_name}"
    client = MlflowClient()
    try:
        model_version = client.create_model_version(name=model_name, source=model_uri, run_id=run.info.run_id)
        client.transition_model_version_stage(name=model_name, version=model_version.version, stage="Staging")
        print(f"Modelo {model_name} registrado/atualizado com sucesso. Versão {model_version.version} em 'Staging'.")
    except Exception as e:
        print(f"Erro ao registrar/atualizar o modelo {model_name}: {str(e)}")


def info_to_string(df):
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    info_lines = info_str.split('\n')
    info_str = '\n'.join(reversed(info_lines))
    return info_str

def teste():
    # vamos verificar o balanceamento de classes
    plt.bar(x=df_wine['rotulo'].value_counts(normalize=True).index,
            height=df_wine['rotulo'].value_counts(normalize=True).values)
    plt.xticks(ticks=[0, 1], labels=['REGULAR', 'OTIMO']) # nomeia os eixos
    plt.show()