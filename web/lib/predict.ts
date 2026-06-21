// Logika predykcji — odpowiednik sklearn Gradient Boosting, liczona w przeglądarce.
// Zweryfikowane: wynik zgodny z sklearn z dokładnością ~1e-6 (patrz scripts/train_and_export.py).

export interface Tree {
  f: number[]; // indeks cechy w węźle (-2 dla liścia)
  t: number[]; // próg podziału
  l: number[]; // lewe dziecko (-1 dla liścia)
  r: number[]; // prawe dziecko
  v: number[]; // wartość liścia (log-odds wkład)
}

export interface Predyktor {
  typ: string;
  F0: number;
  learning_rate: number;
  kolejnosc_cech: string[];
  drzewa: Tree[];
  metryki: Record<string, number>;
}

export interface Cecha {
  key: string;
  label: string;
  median: number;
  p05: number;
  p95: number;
}

export interface Metryki {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface ModelData {
  opis: string;
  predyktor: Predyktor;
  cechy_glowne: Cecha[];
  mediany_all: Record<string, number>;
  presety: { zdrowa: Record<string, number>; bankrut: Record<string, number> };
  modele: Record<string, Metryki>;
  macierze: Record<string, number[][]>;
  najlepszy: string;
  top_waznosc: { key: string; label: string; importance: number }[];
  dane: { firmy: number; bankruci: number; procent_bankrutow: number; liczba_cech: number };
}

function wartoscLiscia(tree: Tree, x: number[]): number {
  let n = 0;
  while (tree.l[n] !== -1) {
    n = x[tree.f[n]] <= tree.t[n] ? tree.l[n] : tree.r[n];
  }
  return tree.v[n];
}

// Prawdopodobieństwo bankructwa (0..1) dla pełnego wektora cech (klucz -> wartość).
export function prawdopodobienstwo(model: ModelData, wartosci: Record<string, number>): number {
  const { kolejnosc_cech, drzewa, F0, learning_rate } = model.predyktor;
  const x = kolejnosc_cech.map((k) => (k in wartosci ? wartosci[k] : model.mediany_all[k]));
  let F = F0;
  for (const tree of drzewa) F += learning_rate * wartoscLiscia(tree, x);
  return 1 / (1 + Math.exp(-F));
}
