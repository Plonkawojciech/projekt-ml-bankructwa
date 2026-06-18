# 📈🇵🇱 GPW Liga Radar — predykcja awansów i spadków w indeksach giełdowych

Projekt zaliczeniowy z przedmiotu **Uczenie maszynowe w Python — laboratorium** (CDV, grupa 4).

> Model uczenia maszynowego, który na podstawie cech rynkowych spółki przewiduje, do której „ligi"
> Giełdy Papierów Wartościowych należy (**WIG20 / mWIG40 / sWIG80**), i wskazuje kandydatów do
> awansu lub spadku przy najbliższej kwartalnej rewizji indeksów — wraz z analizą wpływu na fundusze ETF.

---

## 👤 Skład zespołu

- **Wojciech Płonka** — projekt indywidualny (solo)

---

## 🎯 Koncepcja projektu

Główne indeksy warszawskiej giełdy tworzą hierarchię według wielkości spółek:

| Indeks | Spółki | Charakter |
|---|---|---|
| **WIG20** | 20 największych | blue chips (np. Orlen, KGHM, PKO, Allegro) |
| **mWIG40** | 40 średnich | mid-cap |
| **sWIG80** | 80 mniejszych | small-cap |

Skład tych indeksów jest **rewidowany kwartalnie** — spółki awansują i spadają między ligami w zależności
od rankingu opartego na **kapitalizacji free-float** i **wartości obrotów**. To realne, mierzalne kryteria,
a więc problem nadaje się do modelowania.

**Pomysł:** zbudować klasyfikator, który na podstawie cech rynkowych spółki przypisuje ją do właściwej ligi.
Spółki, które model „widzi" w innej lidze niż obecna (lub leżą blisko granicy decyzyjnej), to **kandydaci do
awansu lub spadku** przy następnej rewizji.

### Wątek ETF

Na GPW notowane są fundusze ETF replikujące te indeksy (m.in. **Beta ETF WIG20TR**, **Beta ETF mWIG40TR**).
Gdy spółka wchodzi do indeksu, fundusz **musi** ją dokupić, a gdy wypada — sprzedać. Te wymuszone transakcje
wpływają na kurs. Projekt pokazuje, które spółki są najbliżej takiej zmiany i jaki może mieć ona wpływ na ETF.

### Dane

W pełni automatyczny, reprodukowalny pipeline (bez ręcznie wpisywanych danych):

1. **Skład indeksów** — pobierany z Wikipedii (`pandas.read_html`): nazwy spółek w każdej lidze.
2. **Mapowanie na symbole giełdowe** — wyszukiwarka Yahoo Finance (nazwa → ticker `.WA`).
3. **Dane rynkowe** — `yfinance`: historia notowań (ceny, wolumen), kapitalizacja, sektor.

### Cechy (features)

Wyliczane z danych rynkowych, zbieżne z oficjalnymi kryteriami rankingu indeksów:

- **kapitalizacja** rynkowa,
- **średni obrót** dzienny (cena × wolumen) — miara płynności,
- **zmienność** (odchylenie standardowe stóp zwrotu),
- **momentum** (stopa zwrotu za 3/6/12 miesięcy),
- poziom ceny, sektor.

### Planowane modele

| Model | Rola |
|---|---|
| **K-Nearest Neighbors** | baza, wymaga skalowania |
| **Decision Tree** | interpretowalny, reguły przypisania do ligi |
| **Random Forest** | główny model, zespół drzew |
| **Gradient Boosting** | model premium do porównania |

Dodatkowo: porównanie konfiguracji, analiza ważności cech, **SHAP** (dlaczego spółka trafia do danej ligi).

### Wizualizacje

- rozkłady cech w podziale na ligi, mapa ciepła korelacji,
- porównanie skuteczności modeli (macierz pomyłek, metryki),
- **ranking kandydatów do awansu/spadku** (spółki najbliżej granicy),
- **interaktywna strona www** (Next.js + Vercel) z wątkiem ETF.

---

## 🗂️ Struktura repozytorium

```
.
├── notebook/                    # część ML (Jupyter/Colab)
│   ├── 01_dane.ipynb            # pobranie i przygotowanie danych GPW
│   ├── 02_modele.ipynb          # trening i porównanie modeli (wkrótce)
│   └── archiwum_bankructwa/     # poprzednia wersja tematu (archiwum)
├── data/                        # zapisane dane (CSV)
├── model/                       # model wyeksportowany do ONNX
├── web/                         # strona Next.js (deploy na Vercel)
└── README.md
```

## 🛠️ Stack technologiczny

- **ML:** Python, pandas, scikit-learn, yfinance, SHAP, matplotlib/seaborn (Google Colab)
- **Web:** Next.js, TypeScript, onnxruntime-web (model w przeglądarce), wykresy
- **Hosting:** Vercel | **Kod:** GitHub

---

## 📊 Wyniki

_(uzupełnione po wytrenowaniu modeli)_

## 🧠 Wnioski

_(uzupełnione na końcu projektu)_
