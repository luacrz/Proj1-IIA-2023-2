import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Carregar os dados
movies_metadata = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings_small.csv')

# Engenharia de Recursos: Calcula a média das classificações de usuários por filme
# e a quantidade de usuarios que avaliaram cada filme
average_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
average_ratings.rename(columns={'mean': 'average_rating', 'count': 'num_ratings'}, inplace=True)

# Filtrar os filmes com pelo menos 20 avaliações
average_ratings = average_ratings[average_ratings['num_ratings'] >= 20]

# Remover entradas que não são convertíveis em inteiros
movies_metadata = movies_metadata[movies_metadata['id'].str.isnumeric()]

# Converter a coluna 'id' para int64
movies_metadata['id'] = movies_metadata['id'].astype('int')

# Combinar os dados de filmes com as médias das avaliações
movies_metadata = movies_metadata.merge(average_ratings, left_on='id', right_on='movieId', how='inner')


# Atribuir rótulos aos filmes com base nos gêneros
target_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Family', 'Fantasy', 
                 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
                 'Thriller', 'War', 'Western', 'NewGenre']

def assign_genre_label(genres):
    genre_list = eval(genres)  # Converte a string de gêneros em uma lista de dicionários
    
    for genre in genre_list:
        if genre['name'] in target_genres:
            return genre['name']
    
    return 'Other'

# print(f'Antes\n {movies_metadata.head(10)}')
movies_metadata['label'] = movies_metadata['genres'].apply(assign_genre_label)
# print(f'Depois\n {movies_metadata.head(10)}')

# Criar o vetorizador de gêneros
vectorizer = CountVectorizer(vocabulary=target_genres)
X = vectorizer.fit_transform(movies_metadata['label']).toarray()

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, movies_metadata['label'], test_size=0.2, random_state=42)

# Treinar o modelo Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Avaliar o modelo
y_pred = naive_bayes.predict(X_test)

# Calcular métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

for genre in target_genres:
    y_true = (y_test == genre).astype(int)
    y_pred_class = (y_pred == genre).astype(int)

    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)

    print(f'Gênero: {genre}')
    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'Pontuação F1: {f1:.2f}')

# Calcular a curva ROC para o gênero 'Action'
y_true_action = (y_test == 'Action').astype(int)
y_prob_action = naive_bayes.predict_proba(X_test)[:, target_genres.index('Action')]

fpr, tpr, thresholds = roc_curve(y_true_action, y_prob_action)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para a classe "Action"')
plt.legend(loc='lower right')
plt.show()
