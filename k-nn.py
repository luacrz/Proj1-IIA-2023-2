import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


# Carregue o conjunto de dados
df = pd.read_csv('movies_metadata.csv')
print(df.head(10))


# Função para atribuir rótulos com base nos gêneros
def assign_genre_label(genres):
    target_genres = ['Action', 'Comedy', 'History', 'Family', 'Drama', 'Adventure', 'Romance', 'Crime', 'Thriller', 'Fantasy', 'Science Fiction', 'Mystery', 'Documentary', 'Horror']
    
    genre_list = eval(genres)
    
    for genre in genre_list:
        if genre['name'] in target_genres:
            return genre['name']
    
    return 'Other'

df['label'] = df['genres'].apply(assign_genre_label)
print(df[['title', 'genres', 'label']])

# Dividir os dados em conjuntos de treinamento e teste
genre_keywords = ['Action', 'Comedy', 'History', 'Family', 'Drama', 'Adventure', 'Romance', 'Crime', 'Thriller', 'Fantasy', 'Science Fiction', 'Mystery', 'Documentary', 'Horror']
vectorizer = CountVectorizer(vocabulary=genre_keywords)
X = vectorizer.transform(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Criação e treinamento do modelo K-NN
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Escolha o número de vizinhos desejado
knn_classifier.fit(X_train, y_train)

# Avaliar o desempenho do modelo
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do Modelo K-NN: {accuracy}')

# Fazer previsões e recomendar filmes com base no gênero previsto
sample_genre_vector = vectorizer.transform(['Action Adventure'])
predicted_genre = knn_classifier.predict(sample_genre_vector)
print(f'Gênero Previsto com K-NN: {predicted_genre[0]}')

recommended_movies = df[df['label'] == predicted_genre[0]]
print('Filmes Recomendados:')
print(recommended_movies[['title', 'genres']])
#
#
#
##
# Avaliar o desempenho do modelo K-NN
y_pred = knn_classifier.predict(X_test)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do Modelo K-NN: {accuracy}')

# Calcular precisão, revocação e pontuação F1 para cada classe
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

# Curva ROC (apenas para uma classe)
y_true = (y_test == 'Action').astype(int) ##exemplo genero ação
y_prob = knn_classifier.predict_proba(X_test)[:, genre_keywords.index('Action')]

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

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