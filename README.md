# Классификация токсичных комментариев

Multi-label классификация комментариев по типу токсичности. 

## Задача

По мотивам [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).
Нужно разметить комментарии с Wikipedia по шести независимым типам токсичности — у одного
комментария может быть несколько меток сразу или ни одной:

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`


**Метрика — mean column-wise ROC-AUC**: ROC-AUC считается по каждой из 6 меток отдельно и
усредняется. 

## Данные

| Выборка | Комментариев |
|---------|-------------:|
| Train | 159 571 |
| Test  | 153 164 |


## Подходы и результаты

| Модель | Mean ROC-AUC |
|--------|:------------:|
| BiLSTM + GloVe | 0.97023 |
| DistilBERT     | 0.98524 |


## Структура проекта

```
toxic-comment-classification/
├── data/                  # CSV с Kaggle + GloVe (в .gitignore)
│   ├── train.csv
│   ├── test.csv
│   └── glove/glove.6B.100d.txt
├── notebooks/
│   ├── 01-eda.ipynb       # анализ + чистка текста -> *_clean.csv
│   ├── 02-bilstm.ipynb    # BiLSTM + предобученные GloVe (с нуля)
│   └── 03-bert.ipynb      # дообучение DistilBERT
├── checkpoints/           # веса лучших моделей (в .gitignore)
├── submissions/           # сабмиты в формате Kaggle
├── src/                   # вспомогательные модули
└── requirements.txt
```

