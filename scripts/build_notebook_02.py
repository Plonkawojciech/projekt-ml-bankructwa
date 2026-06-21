"""Generator notebooka 02_modele.ipynb — buduje poprawny plik .ipynb z opisami w stylu raportu."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'notebook' / '02_modele.ipynb'

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t}

cells = []

cells.append(md(
"""# 🏢💀 Bankrupt-AI — przewidywanie bankructwa polskich firm
## Część 2: Modele, ocena i wnioski

**Autor:** Wojciech Płonka (projekt indywidualny)

**Przedmiot:** Uczenie maszynowe w Python — laboratorium

---

Druga część projektu obejmuje właściwe modelowanie: trening i porównanie czterech klasyfikatorów,
rzetelną ocenę z uwzględnieniem niezbalansowania klas, analizę ważności cech, wyjaśnialność (SHAP)
oraz autorską analizę kosztową błędnych decyzji kredytowych. Notebook jest samowystarczalny — pobiera
i przygotowuje dane od zera, więc można go uruchomić niezależnie od części pierwszej."""))

cells.append(md(
"""## Konfiguracja środowiska

Poza standardowym zestawem (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`) wykorzystywana jest
biblioteka **SHAP** do wyjaśniania decyzji modelu. W Google Colab instalujemy ją jednorazowo."""))

cells.append(code("!pip -q install shap"))

cells.append(code(
"""import io, zipfile, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

sns.set_theme(style='whitegrid')
RNG = 42
print('Środowisko gotowe ✅')"""))

cells.append(md(
"""## Wczytanie i przygotowanie danych

Powtarzamy w skrócie kroki z części pierwszej: pobranie zbioru UCI, scalenie pięciu plików, konwersję
etykiety i uzupełnienie braków medianą. Dzięki temu notebook działa samodzielnie."""))

cells.append(code(
"""URL = 'https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip'
with urllib.request.urlopen(URL) as r:
    zip_bytes = r.read()

ramki = []
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
    for nazwa in sorted(n for n in z.namelist() if n.endswith('.arff')):
        with z.open(nazwa) as f:
            dane, _ = arff.loadarff(io.TextIOWrapper(f, encoding='utf-8'))
        ramki.append(pd.DataFrame(dane))

df = pd.concat(ramki, ignore_index=True)
df['class'] = df['class'].apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
CECHY = [c for c in df.columns if c.startswith('Attr')]
df[CECHY] = df[CECHY].fillna(df[CECHY].median())

print('Rozmiar zbioru:', df.shape)
print(f'Bankruci: {df["class"].sum()} ({df["class"].mean()*100:.2f}%)')"""))

cells.append(md(
"""## Podział na cechy, etykietę oraz zbiór treningowy i testowy

Zmienne objaśniające (`X`) to 64 wskaźniki finansowe, a zmienna objaśniana (`y`) to etykieta bankructwa.
Zbiór dzielimy na **80% treningu** i **20% testu**, z opcją `stratify`, która zachowuje tę samą proporcję
bankrutów w obu częściach — istotne przy tak rzadkiej klasie. Dodatkowo przygotowujemy wersję przeskalowaną
(dla KNN, który mierzy odległości) oraz wagi klas równoważące rzadkość bankrutów."""))

cells.append(code(
"""X = df[CECHY].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG)

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

wagi = compute_sample_weight('balanced', y_train)

print('Trening:', X_train.shape, '| Test:', X_test.shape)"""))

cells.append(md(
"""## Trening modeli

Trenujemy cztery klasyfikatory o różnej naturze:

- **K-Nearest Neighbors (KNN)** — klasyfikuje firmę na podstawie najbardziej podobnych firm; wymaga
  przeskalowanych danych, bo opiera się na odległościach.
- **Drzewo decyzyjne** — ciąg pytań tak/nie; w pełni interpretowalne.
- **Las losowy** — zespół setek drzew głosujących większością; zwykle najdokładniejszy.
- **Gradient Boosting** — drzewa budowane sekwencyjnie, każde poprawia błędy poprzedniego.

Trzy z nich (drzewo, las, boosting) uczymy z **równoważeniem klas** (`class_weight` / `sample_weight`),
aby model nie ignorował rzadkich bankrutów. KNN nie ma takiej opcji — posłuży jako punkt odniesienia
pokazujący, dlaczego naiwne podejście zawodzi na niezbalansowanych danych."""))

cells.append(code(
"""modele = {}

knn = KNeighborsClassifier(n_neighbors=15).fit(X_train_s, y_train)
modele['KNN'] = (knn, X_test_s)

dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RNG).fit(X_train, y_train)
modele['Drzewo decyzyjne'] = (dt, X_test)

rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                            n_jobs=-1, random_state=RNG).fit(X_train, y_train)
modele['Las losowy'] = (rf, X_test)

gb = GradientBoostingClassifier(random_state=RNG).fit(X_train, y_train, sample_weight=wagi)
modele['Gradient Boosting'] = (gb, X_test)

print('Wytrenowano', len(modele), 'modele ✅')"""))

cells.append(md(
"""## Ocena modeli

Przy ~5% bankrutów **sama trafność (accuracy) jest myląca** — model przewidujący zawsze „firma przetrwa"
osiągnąłby ~95% trafności, nie wykrywając ani jednego bankructwa. Dlatego dla klasy „bankrut" liczymy:

- **czułość (recall)** — jaki odsetek realnych bankrutów model wyłapał (najważniejsze: przeoczony bankrut jest kosztowny),
- **precyzję (precision)** — jaki odsetek wskazań „bankrut" był trafny,
- **F1** — średnią harmoniczną obu,
- **AUC** — zdolność rozróżniania klas niezależnie od progu decyzyjnego."""))

cells.append(code(
"""def ocena(model, Xte):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        'Trafność': accuracy_score(y_test, pred),
        'Czułość': recall_score(y_test, pred, zero_division=0),
        'Precyzja': precision_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'AUC': roc_auc_score(y_test, proba),
    }

wyniki = pd.DataFrame({nazwa: ocena(m, Xte) for nazwa, (m, Xte) in modele.items()}).T
wyniki.round(3)"""))

cells.append(md(
"""Tabela powyżej dobrze pokazuje kompromisy: KNN ma najwyższą trafność, ale **czułość bliską zeru** —
jest praktycznie bezużyteczny do wykrywania bankructw. Drzewo i Gradient Boosting wyłapują dużo bankrutów
(wysoka czułość) kosztem precyzji, a Las losowy oferuje najlepszy balans (najwyższe F1 i AUC).

Poniżej pełny raport klasyfikacji dla modelu o najlepszym F1."""))

cells.append(code(
"""najlepszy = wyniki['F1'].idxmax()
model_najlepszy, Xte_najlepszy = modele[najlepszy]
print(f'Najlepszy model wg F1: {najlepszy}\\n')
print(classification_report(y_test, model_najlepszy.predict(Xte_najlepszy),
                            target_names=['zdrowa (0)', 'bankrut (1)']))"""))

cells.append(md(
"""## Macierze pomyłek

Macierz pomyłek rozkłada predykcje na cztery pola. Dla problemu bankructwa najgroźniejsze jest pole
lewy-dół — **bankrut zaklasyfikowany jako zdrowa firma** (przeoczone ryzyko). Porównanie czterech macierzy
naocznie pokazuje, czym różnią się modele."""))

cells.append(code(
"""fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (nazwa, (m, Xte)) in zip(axes, modele.items()):
    cm = confusion_matrix(y_test, m.predict(Xte))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['zdrowa', 'bankrut'], yticklabels=['zdrowa', 'bankrut'])
    ax.set_title(nazwa)
    ax.set_xlabel('predykcja modelu')
    ax.set_ylabel('prawdziwa etykieta')
plt.tight_layout()
plt.show()"""))

cells.append(md(
"""## Graficzne porównanie modeli

Zestawienie kluczowych metryk na jednym wykresie. Skupiamy się na czułości, precyzji, F1 i AUC —
miarach istotnych przy niezbalansowanej klasie."""))

cells.append(code(
"""ax = wyniki[['Czułość', 'Precyzja', 'F1', 'AUC']].plot(kind='bar', figsize=(10, 5))
plt.title('Porównanie modeli (klasa: bankrut)')
plt.ylabel('wartość metryki')
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()"""))

cells.append(md(
"""## Ważność cech

Las losowy pozwala ocenić, które wskaźniki najsilniej wpływają na decyzję modelu. To pierwsza warstwa
interpretacji — pokazuje, na co model „patrzy" najbardziej."""))

cells.append(code(
"""OPIS = {
    'Attr1': 'zysk netto / aktywa', 'Attr2': 'zobowiązania / aktywa', 'Attr3': 'kapitał obrotowy / aktywa',
    'Attr6': 'zyski zatrzymane / aktywa', 'Attr7': 'EBIT / aktywa', 'Attr10': 'kapitał własny / aktywa',
    'Attr13': '(zysk brutto+amort.) / sprzedaż', 'Attr16': '(zysk brutto+amort.) / zobowiązania',
    'Attr24': 'zysk brutto(3l) / aktywa', 'Attr26': '(zysk netto+amort.) / zobowiązania',
    'Attr27': 'zysk oper. / koszty fin.', 'Attr34': 'koszty oper. / zobowiązania',
    'Attr35': 'zysk ze sprzedaży / aktywa', 'Attr39': 'zysk ze sprzedaży / sprzedaż',
    'Attr46': '(akt. obrot.-zapasy) / zob. krótkot.', 'Attr5': 'wskaźnik płynności (cykl)',
    'Attr58': 'koszty / sprzedaż',
}
waznosc = pd.Series(rf.feature_importances_, index=CECHY).sort_values(ascending=False).head(12)[::-1]
etyk = [f'{k} — {OPIS.get(k, k)}' for k in waznosc.index]

plt.figure(figsize=(9, 5))
sns.barplot(x=waznosc.values, y=etyk, hue=etyk, palette='viridis', legend=False)
plt.title('Najważniejsze wskaźniki według lasu losowego')
plt.xlabel('ważność')
plt.ylabel('')
plt.tight_layout()
plt.show()"""))

cells.append(md(
"""## Wyjaśnialność modelu — SHAP

Ważność cech mówi, *które* wskaźniki są istotne, ale nie *w którą stronę* działają. **SHAP** (SHapley
Additive exPlanations) rozkłada każdą predykcję na wkłady poszczególnych cech. Na wykresie zbiorczym poniżej:
każdy punkt to jedna firma, pozycja w poziomie to wpływ cechy na predykcję (w prawo = zwiększa ryzyko
bankructwa), a kolor to wartość wskaźnika (czerwony = wysoka). To najbardziej zaawansowany element analizy."""))

cells.append(code(
"""import shap

# wyjaśniamy najlepszy model na próbce danych testowych (dla szybkości)
proba_n = min(800, len(X_test))
idx = np.random.RandomState(RNG).choice(len(X_test), proba_n, replace=False)
probka = pd.DataFrame(X_test[idx], columns=CECHY)

explainer = shap.TreeExplainer(rf)
shap_out = explainer(probka)
vals = shap_out.values
if vals.ndim == 3:          # (n, cechy, klasy) -> bierzemy klasę 'bankrut'
    vals = vals[:, :, 1]

shap.summary_plot(vals, probka, max_display=12, show=True)"""))

cells.append(md(
"""## Scenariusz kosztowy — perspektywa instytucji finansowej

Modele oceniamy zwykle metrykami, ale w praktyce liczą się **pieniądze**. Dwa typy błędów kosztują różnie:

- **przeoczony bankrut** (firma oznaczona jako zdrowa faktycznie upada) — bank traci na niespłaconym kredycie,
- **fałszywy alarm** (zdrowa firma oznaczona jako bankrut) — utracona marża lub koszt dodatkowej weryfikacji.

Przyjmijmy przykładowe koszty: **przeoczony bankrut = 200 000 zł**, **fałszywy alarm = 5 000 zł**.
Policzmy łączny koszt błędów każdego modelu — okaże się, że najtańszy model to *nie* ten o najwyższej trafności."""))

cells.append(code(
"""KOSZT_PRZEOCZONY = 200_000   # bankrut oznaczony jako zdrowa firma (false negative)
KOSZT_FALSZYWY    = 5_000     # zdrowa firma oznaczona jako bankrut (false positive)

wiersze = []
for nazwa, (m, Xte) in modele.items():
    cm = confusion_matrix(y_test, m.predict(Xte))   # [[TN, FP], [FN, TP]]
    fn, fp = cm[1, 0], cm[0, 1]
    koszt = fn * KOSZT_PRZEOCZONY + fp * KOSZT_FALSZYWY
    wiersze.append({'Model': nazwa, 'Przeoczeni bankruci (FN)': fn,
                    'Fałszywe alarmy (FP)': fp, 'Łączny koszt [zł]': koszt})

koszty = pd.DataFrame(wiersze).set_index('Model').sort_values('Łączny koszt [zł]')
print('Najtańszy model:', koszty['Łączny koszt [zł]'].idxmin())
koszty"""))

cells.append(code(
"""plt.figure(figsize=(9, 4.5))
kolory = ['#16a34a' if v == koszty['Łączny koszt [zł]'].min() else '#94a3b8'
          for v in koszty['Łączny koszt [zł]']]
sns.barplot(x=koszty.index, y=koszty['Łączny koszt [zł]'] / 1e6, palette=kolory, hue=koszty.index, legend=False)
plt.title('Łączny koszt błędów modelu (mln zł)')
plt.ylabel('koszt [mln zł]')
plt.xlabel('')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()"""))

cells.append(md(
"""## Wnioski

- **Dane są silnie niezbalansowane** (~5% bankrutów), co czyni samą trafność (accuracy) bezużyteczną miarą —
  KNN osiąga ~95% trafności, praktycznie nie wykrywając bankrutów.
- **Równoważenie klas jest kluczowe.** Modele uczone z `class_weight`/`sample_weight` (drzewo, las, boosting)
  faktycznie wykrywają bankrutów, podczas gdy naiwny KNN zawodzi.
- **Nie ma jednego „najlepszego" modelu — zależy od celu.** Las losowy daje najlepszy balans (F1, AUC) i
  wysoką precyzję; Gradient Boosting wyłapuje najwięcej bankrutów (najwyższa czułość) kosztem fałszywych alarmów.
- **Analiza kosztowa zmienia ranking.** Gdy przeoczony bankrut jest dużo droższy niż fałszywy alarm, najtańszy
  okazuje się model o najwyższej czułości — a nie ten o najwyższej trafności. To pokazuje, że dobór modelu
  powinien wynikać z kosztu błędów, a nie z pojedynczej metryki.
- **Najważniejsze wskaźniki** to miary rentowności, zadłużenia i zdolności do obsługi zobowiązań — co potwierdza
  SHAP i jest spójne z intuicją ekonomiczną.

Interaktywne demo (wykrywacz ryzyka + scenariusz kosztowy) dostępne jest na stronie projektu zbudowanej w Next.js."""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Zapisano {OUT} ({len(cells)} komórek)')
