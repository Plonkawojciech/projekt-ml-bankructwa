# Przewidywanie bankructwa polskich firm

Projekt z uczenia maszynowego. Na podstawie 64 wskaźników finansowych modele klasyfikacyjne
przewidują, czy firmie grozi bankructwo. Repozytorium zawiera analizę danych, trening i porównanie
czterech modeli, wyjaśnialność (SHAP), analizę kosztową oraz stronę prezentującą wyniki.

Demo: https://predykcja-bankructwa.vercel.app

Autor: Wojciech Płonka

## Opis

Celem jest zbudowanie i porównanie kilku modeli, które na podstawie wskaźników finansowych
(rentowność, zadłużenie, płynność, rotacja) klasyfikują firmę jako zagrożoną bankructwem lub nie.
Ze względu na rzadkość bankructw nacisk położono na właściwą ocenę modeli (czułość i precyzja,
a nie sama trafność) oraz na praktyczne znaczenie błędów (analiza kosztowa).

## Dane

- Źródło: [Polish Companies Bankruptcy Data](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
  (UCI Machine Learning Repository, zbiór 365).
- Około 43 000 rekordów opisujących polskie firmy, każdy z 64 wskaźnikami finansowymi (Attr1–Attr64)
  oraz etykietą `class` (0 — firma przetrwała, 1 — bankructwo).
- Dane są zanonimizowane (bez nazw firm), niezbalansowane (bankruci to ok. 4,8% zbioru) i zawierają
  braki, które uzupełniono medianą wyznaczoną na zbiorze treningowym.

## Modele i wyniki

Cztery modele wytrenowano na 80% danych i przetestowano na pozostałych 20%. Trzy z nich uczono
z ważeniem klas, aby nie ignorowały rzadkich bankrutów.

| Model | Trafność | Czułość | Precyzja | F1 | AUC |
|-------|----------|---------|----------|------|------|
| KNN | 0,952 | 0,002 | 0,500 | 0,005 | 0,722 |
| Decision Tree | 0,807 | 0,723 | 0,163 | 0,266 | 0,854 |
| Random Forest | 0,963 | 0,344 | 0,762 | 0,474 | 0,945 |
| Gradient Boosting | 0,875 | 0,818 | 0,254 | 0,387 | 0,937 |

![Porównanie modeli](artefakty/porownanie_modeli.png)

![Macierze pomyłek](artefakty/macierze_pomylek.png)

Najważniejsze wskaźniki według lasu losowego to miary zadłużenia, rentowności i zdolności do
obsługi zobowiązań:

![Ważność cech](artefakty/waznosc_cech.png)

## Analiza kosztowa

Dwa typy błędów kosztują różnie. Przeoczony bankrut (kredyt dla firmy, która upadnie) jest zwykle
dużo droższy niż fałszywy alarm. Przy założeniu, że przeoczony bankrut to 200 000 zł, a fałszywy
alarm 5 000 zł, najtańszy okazuje się Gradient Boosting (model o najwyższej czułości), a nie ten
o najwyższej ogólnej trafności. Dobór modelu powinien więc wynikać z kosztu błędów, a nie
z pojedynczej metryki.

## Struktura projektu

```
notebook/
  01_dane_i_eksploracja.ipynb   pobranie i czyszczenie danych, eksploracja, wizualizacje
  02_modele.ipynb               trening, ocena, SHAP, analiza kosztowa, wnioski
scripts/
  train_and_export.py           trening modeli i eksport modelu dla strony
  build_notebook_02.py          generator notebooka 02
artefakty/                      wykresy wyników i metryki (JSON)
web/                            strona (Next.js) prezentująca wyniki
```

## Uruchomienie

Notebooki najłatwiej otworzyć w Google Colab (Plik > Otwórz notatnik > GitHub) lub lokalnie w
Jupyterze. Pobierają dane i instalują zależności samodzielnie.

Strona:

```
cd web
npm install
npm run dev
```

## Technologie

- Analiza i modele: Python, pandas, scikit-learn, SHAP, matplotlib, seaborn.
- Strona: Next.js, TypeScript. Model do demonstracji liczony jest po stronie przeglądarki.
