# engMachineLearningKobeShot
Projeto Final da Disciplina Engenharia de Machine Learning da Pós-Graduação em IA e ML (Infnet)

![Alt text](image.png)

START MLFflow:
Configuração de iniciação do serviço do MlFlow.
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
-> INFO:waitress:Serving on http://127.0.0.1:5000


set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve -m "models:/model_kobe_shots/Production" --no-conda -p 5001
 -> INFO:waitress:Serving on http://127.0.0.1:5001

