# 🎬 Classificador de Filmes com Naive Bayes

Este repositório contém uma aplicação completa que utiliza **machine learning** com **Flask** para prever a classificação de sucesso de um filme com base em dados reais do IMDb.

O sistema foi desenvolvido como parte do trabalho final da disciplina de **Probabilidade e Estatística**, unindo aprendizado supervisionado com aplicação prática em uma interface web simples e intuitiva.

---

## 📊 Sobre o Projeto

A aplicação permite prever se um filme é **altamente avaliado** ou não, a partir dos seguintes dados:

- 🎞️ Ano de lançamento  
- ⏱️ Duração do filme (ex: 2h 10min)  
- 🔞 Classificação indicativa (MPA Rating, ex: PG-13, R)

O modelo utilizado é um **Naive Bayes Multinomial**, treinado com uma base real contendo os melhores filmes por ano, extraída da [Base dos Dados](https://basedosdados.org/).

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Flask** – servidor web
- **Pandas** – manipulação de dados
- **Scikit-learn** – machine learning (treino do modelo)
- **HTML + CSS** – interface e estilização
- **Jinja2** – renderização de templates

