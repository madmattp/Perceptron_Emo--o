from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import csv
import nltk
from nltk.corpus import stopwords
import joblib
import time

# Baixa a lista de stopwords
nltk.download('stopwords')

# Carrega o dataset
def load_dataset(file_path: str) -> tuple[list[str], list[str]]:
    print("[INFO] Carregando o dataset...")
    textos = []
    sentimentos = []
    
    with open(file=file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # pula o cabeçalho
        
        for row in csv_reader:
            textos.append(row[2])
            sentimentos.append(row[3])
    
    print(f"[INFO] Total de exemplos carregados: {len(textos)}")
    return textos, sentimentos

# Treina e retorna o modelo Perceptron
def train(dataset_path: str) -> tuple[Perceptron, TfidfVectorizer]:
    textos, sentimentos = load_dataset(file_path=dataset_path)

    print("[INFO] Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        textos, sentimentos, test_size=0.2, random_state=42
    )
    print(f"[INFO] Total treino: {len(X_train)} | Total teste: {len(X_test)}")

    print("[INFO] Criando vetor de stopwords em português...")
    stopwords_pt = stopwords.words('portuguese')

    print("[INFO] Vetorizando textos com TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=stopwords_pt
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[INFO] Iniciando treinamento com Perceptron...")
    start_time = time.time()
    perceptron = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
    perceptron.fit(X_train_vec, y_train)
    end_time = time.time()

    duration = end_time - start_time
    print(f"[INFO] Treinamento finalizado em {duration:.2f} segundos.")

    print("[INFO] Avaliando modelo...")
    y_pred = perceptron.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULTADO] Precisão do modelo: {acc:.4f}")

    return perceptron, vectorizer

# Avalia o texto
def eval_text(text: str, vectorizer: TfidfVectorizer, perceptron: Perceptron) -> str:
    print("[INFO] Avaliando novo texto...")
    vec = vectorizer.transform([text])
    return perceptron.predict(vec)[0]
    
# Salva o modelo
def save_model(vectorizer: TfidfVectorizer, perceptron: Perceptron) -> None:
    print("[INFO] Salvando modelo...")
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(perceptron, 'perceptron_model.joblib')
    print("[INFO] Modelo salvo com sucesso.")

# Carrega o modelo
def load_model(vect_path: str = 'vectorizer.joblib', model_path: str = 'perceptron_model.joblib') -> tuple[TfidfVectorizer, Perceptron]:
    print("[INFO] Carregando modelo salvo...")
    vectorizer = joblib.load(vect_path)
    perceptron = joblib.load(model_path)
    print("[INFO] Modelo carregado com sucesso.")
    return vectorizer, perceptron

if __name__ == "__main__":
    perceptron, vectorizer = train(dataset_path="imdb-reviews-pt-br.csv")

    text = "Uma obra que apenas o grande Zack Snider poderia ter feito. Filme terrível, simplesmente terrível."
    print(f"\n[ENTRADA] Texto avaliado:\n {text}")

    avaliacao = eval_text(text=text, vectorizer=vectorizer, perceptron=perceptron)
    print(f"[SAÍDA] Sentimento avaliado: {avaliacao}")
