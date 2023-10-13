import pandas as pd

## Carregamento dos dados com o Panda
# Caminho do arquivo CSV
file_path = r'C:\Users\Luana\Documents\UnB\semestre 7\IIA\movies_metadata.csv'


# O pandas vai ler o arquivo CSV
df = pd.read_csv(file_path)

# verificação 
print(df.head())

## Pré-processamento dos Dados
# para lidar com valores ausentes
#df = df.fillna(df.mean())

# selecionar elementos relevantes (possibilidade de alteração, vamos discutir)
#relevant_columns = ['title', 'genres', 'cast', 'director', 'vote_average']
#df = df[relevant_columns]

# converter em lista de generos
#df['genres'] = df['genres'].apply(lambda x: x.split('|'))
