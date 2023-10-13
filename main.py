import pandas as pd

# Caminho do arquivo CSV
file_path = r'C:\Users\Luana\Documents\UnB\semestre 7\IIA\movies_metadata.csv'


# O pandas vai ler o arquivo CSV
df = pd.read_csv(file_path)

# Só é uma verificação das primeiras linhas do DataFrame para garantir que os dados foram carregados corretamente
print(df.head())
