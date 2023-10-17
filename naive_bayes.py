import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.impute import SimpleImputer

# Carregue o conjunto de dados
df = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings_small.csv')

# Verifique a estrutura dos dados
#print(df.head(10))
#print(ratings_df.head(10))

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


# Função para atribuir rótulos com base nos gêneros
def assign_genre_label(genres):
    target_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Family', 
    'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
    'Thriller', 'War', 'Western', 'NewGenre']
    
    genre_list = eval(genres)  # Converte a string de gêneros em uma lista de dicionários
    
    for genre in genre_list:
        if genre['name'] in target_genres:
            return genre['name']
    
    return 'Other'

# Atribuir rótulos aos filmes com base em gêneros
df['label'] = df['genres'].apply(assign_genre_label)

# Criar uma lista de todos os gêneros
genre_keywords = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Family', 
'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
'Thriller', 'War', 'Western', 'NewGenre']

# Criar um vetorizador de gêneros
vectorizer = CountVectorizer(vocabulary=genre_keywords)

# Transformar os gêneros e as médias das classificações em recursos
X = vectorizer.transform(df['label'])  # Recursos de gênero

# Converter a matriz esparsa em um DataFrame
X_df = pd.DataFrame(X.toarray())

#print(X_df.head(10))

# Definir os nomes das colunas no DataFrame
X_df.columns = genre_keywords

# Concatenar os DataFrames
X_combined = pd.concat([X_df, df['average_rating']], axis=1)
print(X_combined.head(10))

# Dividir os Dados em Conjunto de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X_combined, df['label'], test_size=0.2, random_state=42)

# print(train_test_split(X_combined, df['label'], test_size=0.2, random_state=42))

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
sample_genre_vector = vectorizer.transform(['Action Adventure Western NewGenre'])
sample_genre_vector = pd.concat([pd.DataFrame(sample_genre_vector.toarray()), pd.DataFrame([df['average_rating'].mean()])], axis=1)
predicted_genre = hgbc_classifier.predict(sample_genre_vector)
print(f'Gênero Previsto: {predicted_genre[0]}')

# Recomendar filmes com base no gênero previsto
recommended_movies = df[df['label'] == predicted_genre[0]]
print('Filmes Recomendados:')
print(recommended_movies[['title', 'genres']].head(10))

# Fazer previsões
y_pred = hgbc_classifier.predict(X_test)

# Calcular precisão, revocação e F1 para cada classe
for genre in genre_keywords:
    y_true = (y_test == genre).astype(int)
    y_pred_class = (y_pred == genre).astype(int)
    '''
    print(f'Y TRUE\n{y_true}')
    print(f'Y test\n{y_test}')
    print(f'Y genre\n{genre}')
    print(f'Y COMPARAÇÃO\n{(y_test == genre).astype(int)}')
    #print(f'Y pred\n{y_pred}')
    #print(f'Y_pred_class\n{y_pred_class}')
    '''
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)

    print(f'Gênero: {genre}')
    print(f'Precisão: {precision:.2f}')
    print(f'Revocação: {recall:.2f}')
    print(f'Pontuação F1: {f1:.2f}')

# Curva ROC (apenas para uma classe)
y_true = (y_test == 'Drama').astype(int)
y_prob = hgbc_classifier.predict_proba(X_test)[:, genre_keywords.index('Drama')]

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para a classe "Drama" ')
plt.legend(loc='lower right')
plt.show()
