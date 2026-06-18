# 🏢💀 Bankrupt-AI — przewidywanie bankructwa polskich firm

Projekt zaliczeniowy z przedmiotu **Uczenie maszynowe w Python — laboratorium** (CDV, grupa 4).

> Model uczenia maszynowego, który na podstawie wskaźników finansowych firmy przewiduje,
> czy grozi jej bankructwo — wraz z interaktywną stroną www i analizą kosztową dla banku/inwestora.

---

## 👤 Skład zespołu

- **Wojciech Płonka** — projekt indywidualny (solo)

---

## 🎯 Koncepcja projektu

Celem jest zbudowanie i porównanie kilku modeli klasyfikacyjnych, które na podstawie
**64 wskaźników finansowych** przedsiębiorstwa (płynność, zadłużenie, rentowność, rotacja)
przewidują, czy firma **zbankrutuje** w nadchodzącym okresie.

Projekt nie kończy się na samym modelu — powstaje też **interaktywna strona internetowa**,
na której można wpisać dane firmy i otrzymać werdykt wraz z wyjaśnieniem (które wskaźniki
najbardziej wpłynęły na decyzję) oraz **scenariuszem kosztowym** (ile traci instytucja
finansowa na błędnej decyzji kredytowej).

### Dane

- **Źródło:** [Polish Companies Bankruptcy Data — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- **Opis:** dane ~10 000 polskich firm zebrane przez Emerging Markets Information Service,
  64 wskaźniki finansowe (Attr1–Attr64) + etykieta `class` (0 = firma przetrwała, 1 = bankructwo).
- **Charakterystyka:** dane silnie **niezbalansowane** (bankrutów jest ~2–7%) oraz z **brakami danych** —
  co czyni je realistycznym, „brudnym" zbiorem wymagającym czyszczenia i odpowiednich technik.

### Planowane modele

| Model | Po co |
|---|---|
| **K-Nearest Neighbors (KNN)** | baza odniesienia, wymaga skalowania danych |
| **Decision Tree** | interpretowalny, daje regułki decyzyjne |
| **Random Forest** | mocniejszy zespół drzew, zwykle najlepszy wynik |
| **Gradient Boosting / XGBoost** | model premium do porównania |

Dodatkowo: porównanie **różnych konfiguracji** (np. liczba sąsiadów w KNN, głębokość drzewa),
techniki radzenia sobie z niezbalansowaniem (**SMOTE / class_weight**) oraz **SHAP** do wyjaśniania decyzji.

### Wizualizacje

- rozkład bankrutów vs zdrowych firm, mapa ciepła korelacji wskaźników
- porównanie skuteczności modeli (accuracy / precision / recall / F1)
- macierz pomyłek, krzywa precision-recall
- **interaktywna strona www** (Next.js + wykresy) — formularz „sprawdź firmę"
- **scenariusz kosztowy** — koszt błędnych decyzji w zł

---

## 🗂️ Struktura repozytorium

```
projekt-ml-bankructwa/
├── notebook/   # Jupyter/Colab — cała część ML (dane → trening → wyniki)
├── data/       # zbiór danych (pobierany w notebooku)
├── model/      # wytrenowany model wyeksportowany do formatu ONNX
├── web/        # strona Next.js (interaktywne demo, deploy na Vercel)
└── README.md
```

## 🛠️ Stack technologiczny

- **ML:** Python, pandas, scikit-learn, imbalanced-learn, SHAP, matplotlib/seaborn (Google Colab)
- **Web:** Next.js, TypeScript, onnxruntime-web (model w przeglądarce), wykresy
- **Hosting:** Vercel | **Kod:** GitHub

---

## 📊 Wyniki

_(uzupełnione po wytrenowaniu modeli)_

## 🧠 Wnioski

_(uzupełnione na końcu projektu)_
