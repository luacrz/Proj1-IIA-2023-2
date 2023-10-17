import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer

# Carregue o conjunto de dados
df = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings_small.csv')

# Engenharia de Recursos: Calcula a média das classificações de usuários por filme
average_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
average_ratings.rename(columns={'mean': 'average_rating', 'count': 'num_ratings'}, inplace=True)

# Filtrar os filmes com pelo menos 20 avaliações
average_ratings = average_ratings[average_ratings['num_ratings'] >= 20]

# Remover entradas que não são convertíveis em inteiros
df = df[df['id'].str.isnumeric()]

# Converter a coluna 'id' para int64
df['id'] = df['id'].astype('int64')

# Combinar os dados de filmes com as médias das avaliações
df = df.merge(average_ratings, left_on='id', right_on='movieId', how='inner')

# Definir os gêneros desejados
genre_keywords = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama']

# Função para atribuir rótulos com base nos gêneros
def assign_genre_label(genres):
    target_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama']
    
    genre_list = eval(genres)  # Converte a string de gêneros em uma lista de dicionários
    
    assigned_genres = [genre['name'] for genre in genre_list if genre['name'] in target_genres]
    
    if assigned_genres:
        return assigned_genres
    else:
        return ['Other']

# Atribuir rótulos aos filmes com base em gêneros
df['label'] = df['genres'].apply(assign_genre_label)

# Usar MultiLabelBinarizer para binarizar os rótulos
mlb = MultiLabelBinarizer()
labels_binarized = mlb.fit_transform(df['label'])

# Criar um DataFrame com os rótulos binarizados
labels_df = pd.DataFrame(labels_binarized, columns=mlb.classes_)

# Concatenar os DataFrames
X_combined = pd.concat([df[genre_keywords], df['average_rating'], labels_df], axis=1)

# Dividir os Dados em Conjunto de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X_combined, labels_df, df['average_rating'], test_size=0.2, random_state=42)

# Substitua valores ausentes com a média
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Criar e treinar o modelo HistGradientBoostingClassifier
hgbc_classifier = HistGradientBoostingClassifier()
hgbc_classifier.fit(X_train_imputed, y_train)

# Acurácia do Modelo
accuracy = hgbc_classifier.score(X_test_imputed, y_test)
print(f'Acurácia do Modelo: {accuracy:.2f}')

# Fazer uma previsão
sample_genre_vector = pd.DataFrame(0, index=[0], columns=genre_keywords)  # Inicializa com zeros
sample_genre_vector['Drama'] = 1  # Adiciona 1 para o gênero 'Drama'
sample_genre_vector['average_rating'] = df['average_rating'].mean()  # Adiciona a média das avaliações
predicted_ratings = hgbc_classifier.predict(sample_genre_vector)
print('Gêneros Previstos:', mlb.inverse_transform(predicted_ratings))

# Recomendar filmes com base nos gêneros previstos
predicted_movies = labels_df[mlb.inverse_transform(predicted_ratings)[0]]
recommended_movies = df.iloc[predicted_movies.index][['title', 'genres']]
print('Filmes Recomendados:')
print(recommended_movies.head(10))

# Fazer previsões
y_pred = hgbc_classifier.predict(X_test_imputed)

# Calcular precisão, revocação e F1 para cada gênero
for i, genre in enumerate(mlb.classes_):
    y_true = y_test[:, i]
    y_pred_class = y_pred[:, i]

    precision = precision_score(y_true, y_pred_class, zero_division=1)
    recall = recall_score(y_true, y_pred_class, zero_division=1)
    f1 = f1_score(y_true, y_pred_class, zero_division=1)

    print(f'Gênero: {genre}')
    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'Pontuação F1: {f1:.2f}')
