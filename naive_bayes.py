# Importe a biblioteca pandas
import pandas as pd

# Carregue o conjunto de dados
df = pd.read_csv('movies_metadata.csv')

# Verifique a estrutura dos dados
print(df.head(10))

# Função para atribuir rótulos com base nos gêneros
def assign_genre_label(genres):
    target_genres = ['Action', 'Comedy', 'History', 'Family', 'Drama', 'Adventure', 'Romance', 'Crime', 'Thriller', 'Fantasy', 'Science Fiction', 'Mystery', 'Documentary', 'Horror']
    
    genre_list = eval(genres)  # Converte a string de gêneros em uma lista de dicionários
    
    for genre in genre_list:
        if genre['name'] in target_genres:
            return genre['name']
    
    return 'Other'

# Atribuir rótulos aos filmes com base em gêneros
df['label'] = df['genres'].apply(assign_genre_label)

# Verifique os rótulos atribuídos
print(df[['title', 'genres', 'label']])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Criar uma lista de todos os gêneros
genre_keywords = ['Action', 'Comedy', 'History', 'Family', 'Drama', 'Adventure', 'Romance', 'Crime', 'Thriller', 'Fantasy', 'Science Fiction', 'Mystery', 'Documentary', 'Horror']

# Criar um vetorizador de gêneros
vectorizer = CountVectorizer(vocabulary=genre_keywords)

# Transformar os gêneros em recursos
X = vectorizer.transform(df['label'])

# Dividir os Dados em Conjunto de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Criar e treinar o modelo Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Acurácia do Modelo
accuracy = nb_classifier.score(X_test, y_test)
print(f'Acurácia do Modelo: {accuracy}')

# Fazer uma previsão
sample_genre_vector = vectorizer.transform(['Action Adventure'])
predicted_genre = nb_classifier.predict(sample_genre_vector)
print(f'Gênero Previsto: {predicted_genre[0]}')

# Recomendar filmes com base no gênero previsto
recommended_movies = df[df['label'] == predicted_genre[0]]
print('Filmes Recomendados:')
print(recommended_movies[['title', 'genres']])

# Fornecer os gêneros do filme para o modelo
sample_genre_vector = vectorizer.transform(['Action Adventure'])

# Obter a categoria de gênero prevista
predicted_genre = nb_classifier.predict(sample_genre_vector)

# Exibir a categoria de gênero prevista
print(f'Gênero Previsto: {predicted_genre[0]}')

#
#
##
# Fazer previsões
y_pred = nb_classifier.predict(X_test)

# Calcular precisão, revocação e F1 para cada classe
for genre in genre_keywords:
    y_true = (y_test == genre).astype(int)
    y_pred_class = (y_pred == genre).astype(int)

    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)

    print(f'Gênero: {genre}')
    print(f'Precisão: {precision}')
    print(f'Revocação: {recall}')
    print(f'Pontuação F1: {f1}')