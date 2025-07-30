from flask import Flask, render_template, request
import pickle
import pandas as pd
import re

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo e os encoders
with open('modelo_naive_bayes.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Carrega os dados reais para popular os selects sem valores inválidos
df = pd.read_csv('data/top_movies_real.csv')

# Regex para capturar padrões plausíveis de duração como "1h", "2h 30min", "1h 15m"
def is_valid_duration(d):
    return isinstance(d, str) and re.match(r'^\d{1,2}h(\s?\d{1,2}m(in)?)?$', d.strip().lower())

durations = sorted(set(d for d in df['duration'].dropna() if is_valid_duration(d)))

# Filtra ratings válidos (remove NaN e strings vazias ou 'nan')
ratings = sorted(df['rating_mpa'].dropna().unique())
ratings = [r for r in ratings if str(r).strip().lower() != 'nan' and str(r).strip() != '']

# Rota inicial - formulário
@app.route('/')
def index():
    return render_template('index.html', durations=durations, ratings=ratings)

# Rota de previsão
@app.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    duration = request.form['duration']
    rating_mpa = request.form['rating_mpa']

    # ⚠️ Validação do ano
    if year < 1960 or year > 2024:
        return render_template('result.html', erro=True)

    # Transforma os dados usando os encoders
    duration_encoded = encoders['duration'].transform([duration])[0]
    rating_mpa_encoded = encoders['rating_mpa'].transform([rating_mpa])[0]

    # Cria o dataframe com os dados formatados
    dados = pd.DataFrame([[year, duration_encoded, rating_mpa_encoded]],
                         columns=['year', 'duration_encoded', 'rating_mpa_encoded'])

    # Faz a previsão
    resultado = modelo.predict(dados)[0]

    return render_template('result.html', resultado=resultado)

# Executa a aplicação
if __name__ == '__main__':
    app.run(debug=True)
