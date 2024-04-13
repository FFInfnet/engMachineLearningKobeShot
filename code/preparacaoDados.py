import pandas as pd
import seaborn as sns
import mlflow.tracking 
import mlflow, os
import random, string, mlflow, os
import personalTools as ptools
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

nome_run = 'Fabio'

mlflow.set_tracking_uri("http://localhost:5000")
nome_experimento = "PreparacaoDados"
if not mlflow.get_experiment_by_name(nome_experimento):
    mlflow.create_experiment(nome_experimento)

# Colunas relevantes
colunas_relevantes = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']

# Configuração do MLflow
mlflow.set_experiment(nome_experimento)
    # Preparando os datasets de treino e teste.
    # serão salvos 4 datasets:
    # treino e teste para desenvolvimento e para produção

dir_data_processed = '../data/processed'
dir_img = '../output/img'    

for tipo in ['dev', 'prod']:
    nome_run_sufix = lambda: '-'.join(random.choice(string.ascii_uppercase)) + ''.join(random.choice(string.digits) for _ in range(3))
    with mlflow.start_run(run_name=nome_run + nome_run_sufix()):    
        dataset_path = f"../data/raw/dataset_kobe_{tipo}.parquet"
        df = pd.read_parquet(dataset_path)

        # removendo linhas com dados faltantes (se houver)
        df_filtered = df[colunas_relevantes].dropna()

        # levantamento dos dados dos datasets
        file_analise_txt = f"../output/txt/dataset_kobe_{tipo}_analise.txt"
        ptools.analyze_dataframe(df_filtered, tipo, file_analise_txt)
        mlflow.log_artifact(file_analise_txt)

        if (tipo == 'prod'):
            # salva o .parquet dos dados já filtrados de produca0
            df_filtered_file = '../data/processed/data_prod_ﬁltered.'
            df_filtered.to_parquet(df_filtered_file+'parquet')
            mlflow.log_artifact(df_filtered_file+'parquet')
            df_filtered.to_csv(df_filtered_file+'csv')
            mlflow.log_artifact(df_filtered_file+'csv')


        if (tipo == 'dev'):
            # salva o .parquet dos dados já filtrados de desenvolvimento
            df_filtered_file = '../data/processed/data_ﬁltered.parquet'
            df_filtered.to_parquet(df_filtered_file)
            mlflow.log_artifact(df_filtered_file)
            
            # Separação dos dados 80% / 20% de maneira aleatoria e estratificada
            random_st = 42
            percent_teste = 0.2
            df_train, df_test = train_test_split(df_filtered,
                                                test_size=percent_teste,
                                                random_state=random_st,
                                                stratify=df_filtered['shot_made_flag'])
            
            # log para responder 6.D
            mlflow.log_param("porcentagem teste", percent_teste * 100)
            if not os.path.exists(dir_data_processed):
                os.makedirs(dir_data_processed)

            if not os.path.exists(dir_img):
                os.makedirs(dir_img)

            for tipo_base in ['train', 'test']:
                df_aux = 'df_' + tipo_base
                df_file = f"{dir_data_processed}/base_{tipo_base}"
                df_file_img_info = f"{dir_img}/base_{tipo_base}_{tipo}_info.png"
                df_file_img_describe = f"{dir_img}/base_{tipo_base}_{tipo}_describe.png"

                file_ext = '.parquet'
                vars()[df_aux].to_parquet(df_file + file_ext)
                mlflow.log_artifact(df_file + file_ext)
                file_ext = '.csv'
                vars()[df_aux].to_csv(df_file + file_ext)
                mlflow.log_artifact(df_file + file_ext)

                ptools.save_info_plot(vars()[df_aux], df_file_img_info)
                mlflow.log_artifact(df_file_img_info)
            
                ptools.save_describe_plot(vars()[df_aux], df_file_img_describe)
                mlflow.log_artifact(df_file_img_describe)
                mlflow.log_metric(f"{tipo_base}_qtd_linhas", vars()[df_aux].shape[0])
    
            dataset_path = f"../data/processed/"
            df_treino = pd.read_parquet(f"{dataset_path}base_train.parquet")
            df_teste = pd.read_parquet(f"{dataset_path}base_test.parquet")
            img_path = f"../output/img/"

            # Visualizando a relação entre as variáveis
            pd.plotting.scatter_matrix(df_treino, diagonal='kde', figsize=(15,15), alpha=0.2)
            plt.savefig(f"{img_path}base_train_relacao_variaveis.png", bbox_inches='tight')

            # Visualizando matriz de correlação das variáveis
            # calculando e plotando a correlação
            df_data = df_treino.copy()
            df_data.drop(['shot_made_flag'], axis=1, inplace=True)
            df_wine_corr = df_data.corr()
            plt.figure(figsize=(9, 9))
            sns.heatmap(df_wine_corr, annot=True, linewidth=.5, cmap=sns.light_palette("seagreen", as_cmap=True))
            plt.savefig(f"{img_path}base_train_matrix_correlacao.png", bbox_inches='tight')

            # Visualizando a distribuição dos valores das variaveis e os outliers.
            fig, axs = plt.subplots(1, 6, figsize=(15, 5))
            for i, coluna in enumerate(df_data):
                df_data[[coluna]].boxplot(ax=axs[i])
                axs[i].set_xticklabels([coluna], rotation=75, ha='right')
            plt.savefig(f"{img_path}base_train_distr_outliers.png", bbox_inches='tight')

            # Visualizando a amplitude dos valores contidos nas variaveis
            ptools.plotagemMaxMin(df_treino,
                                'Comparação de amplitude das variáveis (original)',
                                f"{img_path}base_train_amplitude_variaveis.png")

            # criando um dataset escalonado de treino
            df_treino_escalonado = ptools.std_scaler_dataset(df_treino, ['shot_made_flag'])
            df_treino_escalonado['shot_made_flag'] = df_treino['shot_made_flag']

            # salvando o dataset de treinamento escalonado
            df_treino_escalonado.to_parquet(f"{dataset_path}base_train_escalonado.parquet")
            df_treino_escalonado.to_csv(f"{dataset_path}base_train_escalonado.csv")

            # visualizando a amplitude das variáveis depois do escalonamento
            ptools.plotagemMaxMin(df_treino_escalonado.drop(['shot_made_flag'], axis=1),
                                'Comparação de amplitude das variáveis (escalonadas)',
                                f"{img_path}base_train_amplitude_variaveis_escalonadas.png")

            # criando um dataset escalonado de teste
            df_test_escalonado = ptools.std_scaler_dataset(df_teste, ['shot_made_flag'])
            df_test_escalonado['shot_made_flag'] = df_teste['shot_made_flag']

            # salvando o dataset de treinamento escalonado
            df_test_escalonado.to_parquet(f"{dataset_path}base_test_escalonado.parquet")
            df_test_escalonado.to_csv(f"{dataset_path}base_test_escalonado.csv")

        # plotagem dos histogramas de densidade das variáveis
        for tipo_base in ['train', 'test']:
            file_path = f'{dir_data_processed}/base_{tipo_base}.parquet'
            data = pd.read_parquet(file_path)
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.drop('shot_made_flag')

            for column in numerical_columns:
                plt.figure(figsize=(10,5))
                sns.histplot(data[data['shot_made_flag'] == 1][column], label='Arremessos convertidos (1)', kde=True, element='step', stat='density', common_norm=False)
                sns.histplot(data[data['shot_made_flag'] == 0][column], label='Arremeços errados (0)', kde=True, element='step', stat='density', common_norm=False)
                plt.legend()
                plt.title(f'Distribuição de {column} por Resultado do Arremesso ({tipo_base})')
                plt.xlabel(column)
                plt.ylabel('Densidade')
                plt.savefig(f'{dir_img}/{tipo_base}_hist_density_{column}.png', bbox_inches='tight')
                plt.close()

        mlflow.end_run()