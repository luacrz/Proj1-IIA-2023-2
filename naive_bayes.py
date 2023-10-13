import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

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

# Curva ROC (apenas para uma classe)
y_true = (y_test == 'Action').astype(int)
y_prob = nb_classifier.predict_proba(X_test)[:, genre_keywords.index('Action')]

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para a classe "Action" (Naive Bayes)')
plt.legend(loc='lower right')
plt.show()
