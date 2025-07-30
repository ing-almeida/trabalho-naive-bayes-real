import pandas as pd

df = pd.read_csv('data/top_movies_real.csv')

print("Colunas disponíveis:", df.columns.tolist())
print(df.head())

df['rating_label'] = df['rating_imdb'].apply(lambda x: 'Alta' if x >= 8.5 else 'Baixa')

from sklearn.preprocessing import LabelEncoder
le_duration = LabelEncoder()
le_rating_mpa = LabelEncoder()

df['duration_encoded'] = le_duration.fit_transform(df['duration'].astype(str))
df['rating_mpa_encoded'] = le_rating_mpa.fit_transform(df['rating_mpa'].astype(str))

X = df[['year', 'duration_encoded', 'rating_mpa_encoded']]
y = df['rating_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

modelo = MultinomialNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

import pickle

with open('modelo_naive_bayes.pkl', 'wb') as f:
    pickle.dump(modelo, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'duration': le_duration,
        'rating_mpa': le_rating_mpa
    }, f)

print("\nModelo e encoders salvos com sucesso!")
