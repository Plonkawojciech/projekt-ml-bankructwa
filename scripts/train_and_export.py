"""
Trening i eksport modeli — silnik projektu Bankrupt-AI.

Pobiera dane UCI (polskie firmy), trenuje 4 modele klasyfikacyjne + interpretowalną
regresję logistyczną dla strony www, liczy realne metryki i eksportuje artefakty:

  - web/public/model.json        -> model + metryki dla strony (prawdziwe liczby)
  - artefakty/metryki.json       -> pełne metryki do README/notebooka
  - artefakty/*.png              -> wykresy wyników (macierze pomyłek, porównanie, ważność cech)

Uruchamiane lokalnie (venv .mlenv). Notebook 02_modele.ipynb robi to samo z opisami — to jest "silnik".
"""
import json, io, zipfile, urllib.request, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artefakty'
WEBPUB = ROOT / 'web' / 'public'
ART.mkdir(exist_ok=True)
WEBPUB.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')
RNG = 42

OPIS = {
    'Attr1': 'zysk netto / aktywa', 'Attr2': 'zobowiązania / aktywa', 'Attr3': 'kapitał obrotowy / aktywa',
    'Attr4': 'aktywa obrotowe / zob. krótkot.', 'Attr5': 'wskaźnik płynności (cykl)', 'Attr6': 'zyski zatrzymane / aktywa',
    'Attr7': 'EBIT / aktywa', 'Attr8': 'kapitał własny / zobowiązania', 'Attr9': 'sprzedaż / aktywa',
    'Attr10': 'kapitał własny / aktywa', 'Attr11': '(zysk brutto+koszty fin.) / aktywa', 'Attr12': 'zysk brutto / zob. krótkot.',
    'Attr13': '(zysk brutto+amort.) / sprzedaż', 'Attr14': '(zysk brutto+odsetki) / aktywa', 'Attr15': 'cykl zobowiązań',
    'Attr16': '(zysk brutto+amort.) / zobowiązania', 'Attr17': 'aktywa / zobowiązania', 'Attr18': 'zysk brutto / aktywa',
    'Attr19': 'zysk brutto / sprzedaż', 'Attr20': '(zapasy*365) / sprzedaż', 'Attr21': 'dynamika sprzedaży',
    'Attr22': 'zysk operacyjny / aktywa', 'Attr23': 'zysk netto / sprzedaż', 'Attr24': 'zysk brutto(3l) / aktywa',
    'Attr25': '(kap. własny-akcyjny) / aktywa', 'Attr26': '(zysk netto+amort.) / zobowiązania', 'Attr27': 'zysk oper. / koszty fin.',
    'Attr28': 'kapitał obrotowy / aktywa trwałe', 'Attr29': 'log(aktywa)', 'Attr30': '(zobow.-gotówka) / sprzedaż',
    'Attr31': '(zysk brutto+odsetki) / sprzedaż', 'Attr32': 'cykl zob. krótkot.', 'Attr33': 'koszty oper. / zob. krótkot.',
    'Attr34': 'koszty oper. / zobowiązania', 'Attr35': 'zysk ze sprzedaży / aktywa', 'Attr36': 'sprzedaż / aktywa',
    'Attr37': '(akt. obrot.-zapasy) / zob. długot.', 'Attr38': 'kapitał stały / aktywa', 'Attr39': 'zysk ze sprzedaży / sprzedaż',
    'Attr40': 'kapitał pracujący / zob. krótkot.', 'Attr41': 'zdolność obsługi długu', 'Attr42': 'zysk operacyjny / sprzedaż',
    'Attr43': 'rotacja należności+zapasów (dni)', 'Attr44': '(należności*365) / sprzedaż', 'Attr45': 'zysk netto / zapasy',
    'Attr46': '(akt. obrot.-zapasy) / zob. krótkot.', 'Attr47': '(zapasy*365) / koszt sprzedaży', 'Attr48': 'EBITDA / aktywa',
    'Attr49': 'EBITDA / sprzedaż', 'Attr50': 'aktywa obrotowe / zobowiązania', 'Attr51': 'zob. krótkot. / aktywa',
    'Attr52': 'cykl zob. krótkot. (koszt)', 'Attr53': 'kapitał własny / aktywa trwałe', 'Attr54': 'kapitał stały / aktywa trwałe',
    'Attr55': 'kapitał obrotowy', 'Attr56': 'marża pokrycia', 'Attr57': 'kapitał pracujący / sprzedaż netto',
    'Attr58': 'koszty / sprzedaż', 'Attr59': 'zob. długot. / kapitał własny', 'Attr60': 'sprzedaż / zapasy',
    'Attr61': 'sprzedaż / należności', 'Attr62': '(zob. krótkot.*365) / sprzedaż', 'Attr63': 'sprzedaż / zob. krótkot.',
    'Attr64': 'sprzedaż / aktywa trwałe',
}

# ---------------------------------------------------------------- 1. DANE
print('== Pobieranie danych UCI ==')
URL = 'https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip'
with urllib.request.urlopen(URL) as r:
    zb = r.read()
ramki = []
with zipfile.ZipFile(io.BytesIO(zb)) as z:
    for nazwa in sorted(n for n in z.namelist() if n.endswith('.arff')):
        with z.open(nazwa) as f:
            dane, _ = arff.loadarff(io.TextIOWrapper(f, encoding='utf-8'))
        rk = pd.DataFrame(dane)
        rk['forecast_year'] = int(nazwa.replace('year.arff', '').split('/')[-1])
        ramki.append(rk)
df = pd.concat(ramki, ignore_index=True)
df['class'] = df['class'].apply(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))
CECHY = [c for c in df.columns if c.startswith('Attr')]
df[CECHY] = df[CECHY].fillna(df[CECHY].median())
print(f'   wierszy: {len(df)}, bankrutów: {df["class"].sum()} ({df["class"].mean()*100:.2f}%)')

X = df[CECHY].values
y = df['class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)

scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
sw = compute_sample_weight('balanced', y_train)

# ---------------------------------------------------------------- 2. MODELE
print('== Trening modeli ==')
modele = {}

knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr_s, y_train)
modele['KNN'] = (knn, Xte_s)

dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=RNG).fit(X_train, y_train)
modele['Drzewo decyzyjne'] = (dt, X_test)

rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=RNG).fit(X_train, y_train)
modele['Las losowy'] = (rf, X_test)

gb = GradientBoostingClassifier(random_state=RNG).fit(X_train, y_train, sample_weight=sw)
modele['Gradient Boosting'] = (gb, X_test)

def ocena(model, Xte):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred, zero_division=0),
        'f1': f1_score(y_test, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, proba),
        'cm': confusion_matrix(y_test, pred).tolist(),
    }

wyniki = {nazwa: ocena(m, Xte) for nazwa, (m, Xte) in modele.items()}
for nazwa, w in wyniki.items():
    print(f'   {nazwa:18} acc={w["accuracy"]:.3f}  recall={w["recall"]:.3f}  precision={w["precision"]:.3f}  f1={w["f1"]:.3f}  auc={w["roc_auc"]:.3f}')

# najlepszy wg F1 (balans recall/precyzja przy niezbalansowaniu)
najlepszy = max(wyniki, key=lambda k: wyniki[k]['f1'])
print(f'   => najlepszy model (F1): {najlepszy}')

# ---------------------------------------------------------------- 3. WAŻNOŚĆ CECH (Las losowy)
waznosc = pd.Series(rf.feature_importances_, index=CECHY).sort_values(ascending=False)
top_waznosc = [(k, OPIS.get(k, k), float(v)) for k, v in waznosc.head(12).items()]

# ---------------------------------------------------------------- 4. MODEL DLA STRONY (Gradient Boosting -> drzewa do JSON)
# Eksportujemy mocny model (GB, AUC ~0.94) jako strukturę drzew. Predykcja w przeglądarce liczona w czystym JS:
#   F = F0 + learning_rate * suma(wartości liści) ; prawdopodobieństwo = sigmoid(F).
# Poniżej weryfikujemy, że taka rekonstrukcja daje identyczny wynik jak sklearn.

def eksportuj_drzewo(reg):
    t = reg.tree_
    return {
        'f': t.feature.tolist(),                     # indeks cechy (-2 dla liścia)
        't': [round(float(x), 6) for x in t.threshold],
        'l': t.children_left.tolist(),
        'r': t.children_right.tolist(),
        'v': [round(float(t.value[i].ravel()[0]), 6) for i in range(t.node_count)],
    }

# Dla strony: osobny GB BEZ ważenia klas -> skalibrowane prawdopodobieństwo (typowa firma ~ poziom bazowy).
# Ważone modele zostają w tabeli porównawczej (tam istotna jest czułość).
gb_web = GradientBoostingClassifier(random_state=RNG).fit(X_train, y_train)
gb_web_auc = roc_auc_score(y_test, gb_web.predict_proba(X_test)[:, 1])
drzewa = [eksportuj_drzewo(est[0]) for est in gb_web.estimators_]

def licz_drzewo(tr, x):
    n = 0
    while tr['l'][n] != -1:
        n = tr['l'][n] if x[tr['f'][n]] <= tr['t'][n] else tr['r'][n]
    return tr['v'][n]

# F0 wyznaczamy empirycznie: F0 = decision_function - wkład drzew (stała, niezależna od próbki)
probka = X_test[:300]
wklad = np.array([gb_web.learning_rate * sum(licz_drzewo(tr, x) for tr in drzewa) for x in probka])
ref = gb_web.decision_function(probka).ravel()
F0 = float(np.mean(ref - wklad))
# --- weryfikacja: rekonstrukcja musi zgadzać się z gb_web.decision_function ---
recon = F0 + wklad
max_blad = float(np.max(np.abs(recon - ref)))
prob_mediana = 1 / (1 + np.exp(-(F0 + gb_web.learning_rate * sum(
    licz_drzewo(tr, df[CECHY].median().values) for tr in drzewa))))
print(f'   weryfikacja GB-web (JS vs sklearn): max błąd = {max_blad:.2e}  -> {"OK" if max_blad < 1e-4 else "UWAGA"}')
print(f'   GB-web AUC={gb_web_auc:.3f} | prawdop. dla typowej (median) firmy = {prob_mediana*100:.1f}%')

# 8 najważniejszych wskaźników -> suwaki na stronie (reszta domyślnie = mediana)
top8 = list(waznosc.head(8).index)
zdrowa_med = df[df['class'] == 0][CECHY].median()
bankrut_med = df[df['class'] == 1][CECHY].median()

model_web = {
    'opis': 'Gradient Boosting (drzewa) na 64 wskaźnikach. Prawdopodobieństwo bankructwa liczone w przeglądarce.',
    'predyktor': {
        'typ': 'gradient_boosting',
        'F0': F0,
        'learning_rate': float(gb_web.learning_rate),
        'kolejnosc_cech': CECHY,                     # kolejność 64 cech (indeksy w drzewach)
        'drzewa': drzewa,
        'kalibrowany': True,
        'metryki': {'roc_auc': round(float(gb_web_auc), 4)},
    },
    # 8 cech pokazywanych jako suwaki na stronie
    'cechy_glowne': [
        {'key': k, 'label': OPIS.get(k, k), 'median': float(df[k].median()),
         'p05': float(df[k].quantile(0.05)), 'p95': float(df[k].quantile(0.95))}
        for k in top8
    ],
    'mediany_all': {k: float(df[k].median()) for k in CECHY},   # neutralna "typowa firma" (64 cechy)
    'presety': {
        'zdrowa': {k: float(zdrowa_med[k]) for k in CECHY},
        'bankrut': {k: float(bankrut_med[k]) for k in CECHY},
    },
    'modele': {nazwa: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                       for kk, vv in w.items() if kk != 'cm'} for nazwa, w in wyniki.items()},
    'macierze': {nazwa: w['cm'] for nazwa, w in wyniki.items()},
    'najlepszy': najlepszy,
    'top_waznosc': [{'key': k, 'label': lab, 'importance': round(v, 4)} for k, lab, v in top_waznosc],
    'dane': {'firmy': int(len(df)), 'bankruci': int(df['class'].sum()),
             'procent_bankrutow': round(df['class'].mean() * 100, 2), 'liczba_cech': len(CECHY)},
}
(WEBPUB / 'model.json').write_text(json.dumps(model_web, ensure_ascii=False, indent=2), encoding='utf-8')
(ART / 'metryki.json').write_text(json.dumps(model_web, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'   zapisano web/public/model.json oraz artefakty/metryki.json')

# ---------------------------------------------------------------- 5. WYKRESY WYNIKÓW
# 5a. porównanie modeli
met_df = pd.DataFrame({n: {k: w[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
                       for n, w in wyniki.items()}).T
ax = met_df[['recall', 'precision', 'f1', 'roc_auc']].plot(kind='bar', figsize=(10, 5))
plt.title('Porównanie modeli (klasa: bankrut)')
plt.ylabel('wartość metryki'); plt.xticks(rotation=15); plt.ylim(0, 1); plt.legend(loc='lower right')
plt.tight_layout(); plt.savefig(ART / 'porownanie_modeli.png', dpi=130); plt.close()

# 5b. macierze pomyłek
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (nazwa, w) in zip(axes, wyniki.items()):
    sns.heatmap(w['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['zdrowa', 'bankrut'], yticklabels=['zdrowa', 'bankrut'])
    ax.set_title(nazwa); ax.set_xlabel('predykcja'); ax.set_ylabel('prawda')
plt.tight_layout(); plt.savefig(ART / 'macierze_pomylek.png', dpi=130); plt.close()

# 5c. ważność cech
plt.figure(figsize=(9, 5))
top12 = waznosc.head(12)[::-1]
etyk12 = [f'{k} — {OPIS.get(k, k)[:28]}' for k in top12.index]
sns.barplot(x=top12.values, y=etyk12, hue=etyk12, palette='viridis', legend=False)
plt.title('Najważniejsze wskaźniki wg lasu losowego'); plt.xlabel('ważność'); plt.ylabel('')
plt.tight_layout(); plt.savefig(ART / 'waznosc_cech.png', dpi=130); plt.close()
print('   zapisano wykresy PNG do artefakty/')

# kopia wykresów dla strony
import shutil
for png in ['porownanie_modeli.png', 'macierze_pomylek.png', 'waznosc_cech.png']:
    shutil.copy(ART / png, WEBPUB / png)

print('\n== GOTOWE ==')
print(json.dumps(model_web['modele'], ensure_ascii=False, indent=2))
print('Najlepszy:', najlepszy)
