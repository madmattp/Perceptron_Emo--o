from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import csv
import nltk
from nltk.corpus import stopwords
import joblib

# Baixa a lista de stopwords
nltk.download('stopwords')

# Carrega o dataset
def load_dataset(file_path: str) -> tuple[list[str], list[str]]:
    """
    Carrega um dataset de sentimentos a partir de um arquivo CSV

    O CSV deve conter colunas onde:
    - A coluna índice 2 possui o texto da avaliação.
    - A coluna índice 3 possui o sentimento ("pos" ou "neg").

    Args:
        file_path (str): Caminho para o arquivo CSV.

    Returns:
        tuple[list[str], list[str]]: Uma tupla contendo duas listas:
            - Lista de textos (str)
            - Lista de sentimentos (str)
    """
    
    textos = []
    sentimentos = []
    
    with open(file=file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # pula o cabeçalho
        
        for row in csv_reader:
            # row[2] = texto da review, row[3] = sentimento ("pos" ou "neg")
            textos.append(row[2])
            sentimentos.append(row[3])

    return textos, sentimentos

# Treina e retorna o mmodelo Perceptron        
def train(dataset_path: str)-> tuple[Perceptron, TfidfVectorizer]:

    textos, sentimentos = load_dataset(file_path=dataset_path)
    
    # DIVIDINDO OS DADOS EM CONJUNTOS DE TREINAMENTO E TESTE
        # X_train	Lista com os textos de treino (80% do total)
        # X_test	Lista com os textos de teste (20% do total)
        # y_train	Lista com os rótulos ("pos"/"neg") de treino correspondentes a X_train
        # y_test	Lista com os rótulos ("pos"/"neg") de teste correspondentes a X_test

    X_train, X_test, y_train, y_test = train_test_split(
        textos, sentimentos, test_size=0.2, random_state=42
    )

    # Lista de stopwords em português
    stopwords_pt = stopwords.words('portuguese')

    # Vetoriza os parâmetros com TfidVectorizer    (TF-IDF = Term Frequency - Inverse Document Frequency)
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Limita o vocabulário às 5000 palavras mais frequentes
        stop_words=stopwords_pt  # Remove palavras muito comuns em português (ex: "o", "e", "mas")
    )
    X_train_vec = vectorizer.fit_transform(X_train)  # Aprende o vocabulário com os textos de treino e transforma em vetores
    X_test_vec = vectorizer.transform(X_test)        # Transforma os textos de teste usando o mesmo vocabulário


    # Cria o classificador Perceptron
    perceptron = Perceptron(max_iter=1000, eta0=0.01, random_state=42)

    # Treina o modelo
    perceptron.fit(X_train_vec, y_train)

    # Faz previsões no conjunto de teste
    y_pred = perceptron.predict(X_test_vec)

    # Avalia a confiança (precisão)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisão: {acc:.4f}")


    # Retorna o modelo Perceptron e o vetorizador
    return perceptron, vectorizer

# Avalia o texto
def eval_text(text: str, vectorizer: TfidfVectorizer, perceptron: Perceptron) -> str:
    vec = vectorizer.transform([text])
    return perceptron.predict(vec)[0]
    
# Salva o modelo em formato de arquivo
def save_model(vectorizer: TfidfVectorizer, perceptron: Perceptron) -> None:
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(perceptron, 'perceptron_model.joblib')


if __name__ == "__main__":
    perceptron, vectorizer = train(dataset_path="imdb-reviews-pt-br.csv")

    text = "Uma obra que apenas o grande Zack Snider poderia ter cagado. Filme terrível, simplesmente terrível."
    print(f"Texto avaliado: \n {text}")

    avaliacao = eval_text(text=text, vectorizer=vectorizer, perceptron=perceptron)
    print(f"Eval: {avaliacao}")