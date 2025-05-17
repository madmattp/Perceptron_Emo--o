# Análise de Sentimentos com Perceptron

Este projeto implementa uma análise de sentimentos utilizando **aprendizado de máquina** com o classificador **Perceptron**. O objetivo é treinar um modelo capaz de identificar se um texto (ex: uma resenha de filme) possui **sentimento positivo ou negativo**.

## 📚 Descrição
- O modelo é treinado com um arquivo `.csv` contendo textos em português e seus respectivos sentimentos (`"pos"` ou `"neg"`).
- Usa o `TfidfVectorizer` para transformar o texto em vetores numéricos (TF-IDF).
- Utiliza o classificador `Perceptron` da biblioteca `scikit-learn`.
- Pode avaliar novos textos e salvar/carregar o modelo treinado usando `joblib`.
- Foi desenvolvido com foco na simplicidade e clareza para fins educacionais.

## 📂 Dataset
O arquivo CSV utilizado foi obtido do Kaggle, disponível em:

🔗 [IMDB Reviews em Português - Kaggle](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr)

Formato esperado do CSV:
- **Coluna 2**: o texto da resenha.
- **Coluna 3**: o sentimento da resenha (`"pos"` ou `"neg"`).

Certifique-se de manter o cabeçalho original do CSV e que ele esteja codificado em UTF-8.

## ⚙️ Requisitos
- Python 3.12 ou superior
- Bibliotecas necessárias:
  - `scikit-learn`
  - `nltk`
  - `joblib`

Instale as dependências com:
```bash
pip install -r requirements.txt
```

## 🚀 Como usar

### 1. Treinar o modelo
```python
>>> from emotion_eval import *
[nltk_data] Downloading package stopwords to
[nltk_data]     /home/usuario/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
>>> perceptron, vectorizer = train(dataset_path="imdb-reviews-pt-br.csv")
[INFO] Carregando o dataset...
[INFO] Total de exemplos carregados: 49459
[INFO] Dividindo dados em treino e teste...
[INFO] Total treino: 39567 | Total teste: 9892
[INFO] Criando vetor de stopwords em português...
[INFO] Vetorizando textos com TF-IDF...
[INFO] Iniciando treinamento com Perceptron...
[INFO] Treinamento finalizado em 0.12 segundos.
[INFO] Avaliando modelo...
[RESULTADO] Precisão do modelo: 0.8449
```

### 2. Avaliar novos textos
```python
>>> text = "Uma obra que apenas o grande Zack Snider poderia ter feito. Filme terrível, simplesmente terrível."
>>> avaliacao = eval_text(text=text, vectorizer=vectorizer, perceptron=perceptron)
[INFO] Avaliando novo texto...
>>> print(f"[SAÍDA] Sentimento avaliado: {avaliacao}")
[SAÍDA] Sentimento avaliado: neg
```

### 3. Salvar e carregar modelo
```python
>>> save_model(perceptron, vectorizer)
[INFO] Modelo salvo em 'model.joblib' e 'vectorizer.joblib'.

>>> perceptron, vectorizer = load_model()
[INFO] Modelo e vetorizador carregados com sucesso.
```

## 📝 Observações
- Certifique-se de que o arquivo `imdb-reviews-pt-br.csv` esteja no diretório correto ou forneça o caminho completo.
- O script faz uso da base de stopwords da `nltk` para o português. Ela será baixada automaticamente se necessário.
- Para melhorar a acurácia, recomenda-se fazer limpeza textual mais agressiva ou experimentar outros vetorizadores.

## 📄 Licença

Este projeto é open source e pode ser utilizado livremente para fins educacionais e acadêmicos.
