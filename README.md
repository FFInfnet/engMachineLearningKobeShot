# engMachineLearningKobeShot
# Projeto Final da Disciplina Engenharia de Machine Learning da Pós-Graduação em IA e ML (Infnet)

# Análise dos Arremessos de Kobe Bryant com Machine Learning

# Este projeto utiliza técnicas de machine learning para analisar e prever os resultados dos arremessos # de Kobe Bryant ao longo de sua carreira na NBA.

## Instalação
Clone o projeto usando:
git clone https://github.com/FFInfnet/engMachineLearningKobeShot.git

Instale as dependências necessárias com:
pip install -r requirements.txt

## Uso

    # 1 - START MLFflow:
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
        # -> INFO:waitress:Serving on http://127.0.0.1:5000

    # 2 - Execute o arquivo: preparacaoDados.py
    # 3 - Execute o arquivo: treinamento.py
    # 4 - Registre o modelo "model_kobe_shots" no mlFlow
    # 5 - Inicie o serviço do modelo no mlflow: 
        set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
        mlflow models serve -m "models:/model_kobe_shots/Production" --no-conda -p 5001
            # -> INFO:waitress:Serving on http://127.0.0.1:5001
    # 6 - Execute o arquivo aplicacao.py
    # 7 - Execute o serviço do streamlit: streamlit run app.py
    

## Contribuição

Contribuições são sempre bem-vindas! Veja como você pode contribuir:

- Faça fork do projeto.
- Crie uma nova branch (`git checkout -b feature/nomeDaFeature`).
- Faça commit das suas alterações (`git commit -am 'Adicionar alguma funcionalidade'`).
- Faça push para a branch (`git push origin feature/nomeDaFeature`).
- Abra um Pull Request.

## Licença

Este projeto está licenciado sob a Licença GNU GENERAL PUBLIC LICENSE.

## Contato

Se você tiver alguma dúvida ou sugestão, pode abrir uma issue aqui no GitHub ou me enviar uma mensagem diretamente.


