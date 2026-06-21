"use client";

import { useEffect, useMemo, useState } from "react";
import { ModelData, prawdopodobienstwo } from "@/lib/predict";

const fmtPct = (x: number) => (x * 100).toFixed(1) + "%";
const fmtInt = (x: number) => x.toLocaleString("pl-PL");
const fmtZl = (x: number) => x.toLocaleString("pl-PL") + " zł";

const NAZWY_METRYK: [keyof ModelData["modele"][string], string][] = [
  ["recall", "Czułość"],
  ["precision", "Precyzja"],
  ["f1", "F1"],
  ["roc_auc", "AUC"],
  ["accuracy", "Trafność"],
];

export default function Strona() {
  const [model, setModel] = useState<ModelData | null>(null);
  const [wartosci, setWartosci] = useState<Record<string, number>>({});
  const [kosztFN, setKosztFN] = useState("200000");
  const [kosztFP, setKosztFP] = useState("5000");

  useEffect(() => {
    fetch("/model.json")
      .then((r) => r.json())
      .then((m: ModelData) => {
        setModel(m);
        setWartosci({ ...m.mediany_all });
      });
  }, []);

  const prob = useMemo(
    () => (model ? prawdopodobienstwo(model, wartosci) : 0),
    [model, wartosci]
  );

  // wkład każdego z 8 wskaźników: o ile zmienia ryzyko względem swojej mediany
  const wklady = useMemo(() => {
    if (!model) return [];
    return model.cechy_glowne
      .map((c) => {
        const probMediana = prawdopodobienstwo(model, { ...wartosci, [c.key]: c.median });
        return { label: c.label, key: c.key, efekt: prob - probMediana };
      })
      .filter((w) => Math.abs(w.efekt) > 0.002)
      .sort((a, b) => Math.abs(b.efekt) - Math.abs(a.efekt))
      .slice(0, 6);
  }, [model, wartosci, prob]);

  if (!model) return <div className="loading">Ładowanie modelu…</div>;

  const ustaw = (key: string, v: number) => setWartosci((w) => ({ ...w, [key]: v }));
  const preset = (p: Record<string, number>) => setWartosci({ ...p });

  // progi ustawione względem bazowej częstości bankructw (~5%): model zwraca skalibrowane
  // prawdopodobieństwo, więc 3× powyżej bazy to już ryzyko podwyższone, a ~3× więcej — wysokie.
  const band =
    prob < 0.06
      ? { kolor: "var(--green)", txt: "Niskie ryzyko" }
      : prob < 0.15
      ? { kolor: "var(--amber)", txt: "Podwyższone ryzyko" }
      : { kolor: "var(--red)", txt: "Wysokie ryzyko" };

  const maxWklad = Math.max(0.001, ...wklady.map((w) => Math.abs(w.efekt)));

  // scenariusz kosztowy z macierzy pomyłek: cm = [[TN, FP], [FN, TP]]
  const fnNum = Number(kosztFN) || 0;
  const fpNum = Number(kosztFP) || 0;
  const koszty = Object.entries(model.macierze).map(([nazwa, cm]) => ({
    nazwa,
    fn: cm[1][0],
    fp: cm[0][1],
    koszt: cm[1][0] * fnNum + cm[0][1] * fpNum,
  }));
  const minKoszt = Math.min(...koszty.map((k) => k.koszt));

  return (
    <main>
      {/* HERO */}
      <header className="hero">
        <div className="wrap">
          <span className="eyebrow">Projekt ML · Uczenie maszynowe w Python</span>
          <h1>Czy ta firma zbankrutuje?</h1>
          <p className="lead">
            Model uczenia maszynowego przewidujący bankructwo polskich firm na podstawie 64 wskaźników
            finansowych. Porównanie czterech modeli, analiza kosztowa i interaktywny wykrywacz ryzyka.
          </p>
          <div className="stats">
            <div className="stat">
              <div className="num">{fmtInt(model.dane.firmy)}</div>
              <div className="lab">analizowanych firm</div>
            </div>
            <div className="stat">
              <div className="num">{model.dane.procent_bankrutow}%</div>
              <div className="lab">to bankruci</div>
            </div>
            <div className="stat">
              <div className="num">{model.dane.liczba_cech}</div>
              <div className="lab">wskaźników finansowych</div>
            </div>
            <div className="stat">
              <div className="num">{model.modele[model.najlepszy].roc_auc.toFixed(2)}</div>
              <div className="lab">AUC najlepszego modelu</div>
            </div>
          </div>
        </div>
      </header>

      {/* WYKRYWACZ */}
      <section>
        <div className="wrap">
          <p className="kicker">Interaktywny wykrywacz</p>
          <h2>Sprawdź ryzyko firmy</h2>
          <p className="sub">
            Ustaw osiem najważniejszych wskaźników, a model (Gradient Boosting, liczony w przeglądarce)
            na żywo oszacuje prawdopodobieństwo bankructwa. Skorzystaj z gotowych profili, by zobaczyć kontrast.
          </p>

          <div className="predictor">
            <div className="card sliders">
              {model.cechy_glowne.map((c) => {
                const v = wartosci[c.key] ?? c.median;
                const step = (c.p95 - c.p05) / 100 || 0.01;
                return (
                  <div className="row" key={c.key}>
                    <div className="slabel">
                      <span className="name">{c.label}</span>
                      <span className="val">{v.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min={c.p05}
                      max={c.p95}
                      step={step}
                      value={v}
                      aria-label={c.label}
                      aria-valuetext={v.toFixed(2)}
                      onChange={(e) => ustaw(c.key, parseFloat(e.target.value))}
                    />
                  </div>
                );
              })}
              <div className="presets">
                <button className="btn" onClick={() => preset(model.presety.zdrowa)}>
                  ↩ Typowa zdrowa firma
                </button>
                <button className="btn" onClick={() => preset(model.presety.bankrut)}>
                  ⚠ Typowy bankrut
                </button>
                <button className="btn" onClick={() => preset(model.mediany_all)}>
                  Reset
                </button>
              </div>
            </div>

            <div className="card verdict">
              <div className="bigpct" style={{ color: band.kolor }}>
                {fmtPct(prob)}
              </div>
              <div className="tag" style={{ background: band.kolor, color: "#fff" }}>
                {band.txt}
              </div>
              <div className="track">
                {/* skala 0–40%: większość firm mieści się nisko, więc rozciągamy zakres dla czytelności */}
                <div className="marker" style={{ left: `${Math.min(100, prob * 250)}%` }} />
              </div>
              <p className="baserate">Bazowa częstość bankructw w zbiorze: ~5%</p>

              <div className="contrib">
                <h4>Co najbardziej wpływa na ten wynik</h4>
                {wklady.length > 0 ? (
                  wklady.map((w) => (
                    <div className="crow" key={w.key}>
                      <span className="cn">{w.label}</span>
                      <div
                        className="cbar"
                        style={{
                          width: `${(Math.abs(w.efekt) / maxWklad) * 70 + 8}px`,
                          background: w.efekt > 0 ? "var(--red)" : "var(--green)",
                        }}
                      />
                      <span style={{ color: w.efekt > 0 ? "var(--red)" : "var(--green)", fontWeight: 600 }}>
                        {w.efekt > 0 ? "↑" : "↓"}
                      </span>
                    </div>
                  ))
                ) : (
                  <p style={{ fontSize: "12.5px", color: "var(--muted)", margin: 0 }}>
                    Wszystkie wskaźniki są blisko typowych wartości — brak wyróżniających się czynników ryzyka.
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* PORÓWNANIE MODELI */}
      <section>
        <div className="wrap">
          <p className="kicker">Wyniki</p>
          <h2>Porównanie czterech modeli</h2>
          <p className="sub">
            Każdy model trenowano na 80% danych i testowano na pozostałych 20%. Przy ~5% bankrutów sama
            trafność (accuracy) myli — kluczowa jest <strong>czułość</strong> (ilu realnych bankrutów model
            wyłapuje) oraz <strong>precyzja</strong>.
          </p>

          <div className="card" style={{ overflowX: "auto" }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  {NAZWY_METRYK.map(([, label]) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(model.modele).map(([nazwa, m]) => (
                  <tr key={nazwa} className={nazwa === model.najlepszy ? "best" : ""}>
                    <td>
                      {nazwa}
                      {nazwa === model.najlepszy && <span className="pill">najlepszy F1</span>}
                    </td>
                    {NAZWY_METRYK.map(([k]) => (
                      <td key={k}>{m[k].toFixed(3)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="figure">
            <img src="/porownanie_modeli.png" alt="Porównanie metryk modeli" />
            <p className="figcap">
              Porównanie metryk dla klasy „bankrut”. Las losowy daje najlepszy balans, Gradient Boosting
              wyłapuje najwięcej bankrutów (najwyższa czułość).
            </p>
          </div>

          <div className="figure">
            <img src="/macierze_pomylek.png" alt="Macierze pomyłek modeli" />
            <p className="figcap">
              Macierze pomyłek. Widać, dlaczego KNN zawodzi przy niezbalansowanych danych — niemal nie
              wykrywa bankrutów, mimo wysokiej ogólnej trafności.
            </p>
          </div>
        </div>
      </section>

      {/* WAŻNOŚĆ CECH */}
      <section>
        <div className="wrap">
          <p className="kicker">Wyjaśnialność</p>
          <h2>Co decyduje o bankructwie</h2>
          <p className="sub">
            Las losowy ocenia, które wskaźniki najsilniej wpływają na predykcję. Dominują miary
            zadłużenia, rentowności i zdolności do obsługi zobowiązań.
          </p>
          <div className="figure">
            <img src="/waznosc_cech.png" alt="Ważność wskaźników finansowych" />
            <p className="figcap">Najważniejsze wskaźniki według lasu losowego.</p>
          </div>
        </div>
      </section>

      {/* SCENARIUSZ KOSZTOWY */}
      <section>
        <div className="wrap">
          <p className="kicker">Analiza biznesowa</p>
          <h2>Scenariusz kosztowy</h2>
          <p className="sub">
            Nie każdy błąd kosztuje tyle samo. Przeoczony bankrut (udzielony kredyt firmie, która upadnie)
            jest zwykle znacznie droższy niż fałszywy alarm. Ustaw koszty i zobacz, który model jest
            najtańszy dla instytucji finansowej — to często <em>nie</em> ten o najwyższej trafności.
          </p>

          <div className="card">
            <div className="costgrid">
              <div className="field">
                <label>Koszt przeoczonego bankruta (fałszywie „zdrowa”)</label>
                <input
                  type="number"
                  value={kosztFN}
                  min={0}
                  step={10000}
                  inputMode="numeric"
                  onChange={(e) => setKosztFN(e.target.value)}
                />
                <div className="hint">strata na niespłaconym kredycie</div>
              </div>
              <div className="field">
                <label>Koszt fałszywego alarmu (zdrowa oznaczona „bankrut”)</label>
                <input
                  type="number"
                  value={kosztFP}
                  min={0}
                  step={1000}
                  inputMode="numeric"
                  onChange={(e) => setKosztFP(e.target.value)}
                />
                <div className="hint">utracona marża / koszt weryfikacji</div>
              </div>
            </div>

            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Przeoczeni bankruci</th>
                  <th>Fałszywe alarmy</th>
                  <th>Łączny koszt</th>
                </tr>
              </thead>
              <tbody>
                {koszty.map((k) => (
                  <tr key={k.nazwa} className={k.koszt === minKoszt ? "best" : ""}>
                    <td>
                      {k.nazwa}
                      {k.koszt === minKoszt && <span className="pill">najtańszy</span>}
                    </td>
                    <td>{fmtInt(k.fn)}</td>
                    <td>{fmtInt(k.fp)}</td>
                    <td>{fmtZl(k.koszt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer>
        <div className="wrap">
          <strong>Bankrupt-AI</strong> — projekt zaliczeniowy z przedmiotu „Uczenie maszynowe w Python”.
          <br />
          Dane:{" "}
          <a href="https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data" target="_blank" rel="noreferrer">
            Polish Companies Bankruptcy Data (UCI)
          </a>{" "}
          · Kod:{" "}
          <a href="https://github.com/Plonkawojciech/projekt-ml-bankructwa" target="_blank" rel="noreferrer">
            GitHub
          </a>
          <div className="disclaimer">
            Narzędzie demonstracyjne i edukacyjne. Predykcje opierają się na danych historycznych
            zanonimizowanych spółek i nie stanowią porady finansowej ani oceny zdolności kredytowej.
          </div>
        </div>
      </footer>
    </main>
  );
}
