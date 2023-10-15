# Criar um vetorizador de gêneros
vectorizer = CountVectorizer(vocabulary=genre_keywords)

# Transformar os gêneros e as médias das classificações em recursos
X = vectorizer.transform(df['label'])  # Recursos de gênero
X = pd.concat([X, df['average_rating']], axis=1)  # Recursos combinados

# Converter a matriz esparsa em um DataFrame
X_df = pd.DataFrame(X.toarray())

# Definir os nomes das colunas no DataFrame
X_df.columns = genre_keywords

# Concatenar os DataFrames
X_combined = pd.concat([X_df, df['average_rating']], axis=1)

# Dividir os Dados em Conjunto de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)
