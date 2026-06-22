"use client";

import { useEffect, useState } from "react";
import { ModelData, prawdopodobienstwo } from "@/lib/predict";
import { VizKNN, VizTree, VizForest, VizBoosting } from "./ModelViz";

const JAK_DZIALAJA = [
  {
    nazwa: "K-Nearest Neighbors (KNN)",
    opis: "Szuka najbardziej podobnych firm w danych i sprawdza, ile z nich zbankrutowało — większość sąsiadów decyduje o werdykcie.",
    Viz: VizKNN,
  },
  {
    nazwa: "Decision Tree (drzewo decyzyjne)",
    opis: "Zadaje ciąg pytań tak/nie o wskaźniki firmy i schodzi gałęziami aż do końcowej decyzji: zdrowa albo bankrut.",
    Viz: VizTree,
  },
  {
    nazwa: "Random Forest (las losowy)",
    opis: "Buduje setki różnych drzew, każde głosuje, a wynikiem jest decyzja większości — pojedyncze błędy się znoszą.",
    Viz: VizForest,
  },
  {
    nazwa: "Gradient Boosting",
    opis: "Buduje drzewa po kolei — każde kolejne naprawia błędy poprzedniego, więc model stopniowo coraz lepiej rozpoznaje trudne przypadki.",
    Viz: VizBoosting,
  },
];

const fmtPct = (x: number) => (x * 100).toFixed(1) + "%";
const fmtInt = (x: number) => x.toLocaleString("pl-PL");
const fmtZl = (x: number) => x.toLocaleString("pl-PL") + " zł";

const REPO = "https://github.com/Plonkawojciech/projekt-ml-bankructwa";

const NAZWY_METRYK: [keyof ModelData["modele"][string], string][] = [
  ["recall", "Czułość"],
  ["precision", "Precyzja"],
  ["f1", "F1"],
  ["roc_auc", "AUC"],
  ["accuracy", "Trafność"],
];

export default function Strona() {
  const [model, setModel] = useState<ModelData | null>(null);
  const [kosztFN, setKosztFN] = useState("200000");
  const [kosztFP, setKosztFP] = useState("5000");

  useEffect(() => {
    fetch("/model.json")
      .then((r) => r.json())
      .then((m: ModelData) => setModel(m));
  }, []);

  if (!model) return <div className="loading">Ładowanie modelu…</div>;

  const bandFor = (p: number) =>
    p < 0.06
      ? { kolor: "var(--green)", txt: "Niskie ryzyko" }
      : p < 0.15
      ? { kolor: "var(--amber)", txt: "Podwyższone ryzyko" }
      : { kolor: "var(--red)", txt: "Wysokie ryzyko" };

  const przyklady = [
    { nazwa: "Typowa zdrowa firma", p: prawdopodobienstwo(model, model.presety.zdrowa) },
    { nazwa: "Firma zagrożona", p: prawdopodobienstwo(model, model.presety.bankrut) },
  ];

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
      {/* TOP BAR */}
      <div className="topbar">
        <div className="topbar-inner">
          <span className="name">Przewidywanie bankructwa firm</span>
          <a href={REPO} target="_blank" rel="noreferrer">
            Kod na GitHub ↗
          </a>
        </div>
      </div>

      {/* INTRO */}
      <header className="intro">
        <div className="wrap">
          <p className="label">Projekt — uczenie maszynowe</p>
          <h1>Przewidywanie bankructwa polskich firm</h1>
          <p className="lead">
            Analiza danych finansowych 43 405 spółek i porównanie czterech modeli klasyfikacyjnych, które na
            podstawie 64 wskaźników przewidują, czy firmie grozi bankructwo.
          </p>
          <div className="facts">
            <div className="f">
              <b>{fmtInt(model.dane.firmy)}</b>
              <span>analizowanych firm</span>
            </div>
            <div className="f">
              <b>{model.dane.procent_bankrutow}%</b>
              <span>to bankruci</span>
            </div>
            <div className="f">
              <b>{model.dane.liczba_cech}</b>
              <span>wskaźników</span>
            </div>
            <div className="f">
              <b>4</b>
              <span>modele ML</span>
            </div>
          </div>
        </div>
      </header>

      {/* 01 DANE */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">01</span>
            <h2>Dane</h2>
          </div>
          <p className="sub">
            Zbiór <em>Polish Companies Bankruptcy Data</em> z repozytorium UCI — zanonimizowane sprawozdania
            finansowe polskich firm. Każdy rekord to 64 wskaźniki (rentowność, zadłużenie, płynność) oraz
            informacja, czy firma zbankrutowała. Zbiór jest silnie <strong>niezbalansowany</strong>: bankruci
            stanowią zaledwie {model.dane.procent_bankrutow}% — co bezpośrednio wpływa na dobór i ocenę modeli.
          </p>
        </div>
      </section>

      {/* 02 MODELE */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">02</span>
            <h2>Modele i wyniki</h2>
          </div>
          <p className="sub">
            Każdy model trenowano na 80% danych i testowano na pozostałych 20%. Przy ~5% bankrutów sama
            trafność (accuracy) myli — kluczowa jest <strong>czułość</strong> (ilu realnych bankrutów model
            wyłapuje) oraz <strong>precyzja</strong>.
          </p>

          <div className="card tablecard">
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
              Porównanie metryk dla klasy „bankrut”. Random Forest daje najlepszy balans, Gradient Boosting
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

      {/* 03 CO DECYDUJE */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">03</span>
            <h2>Co decyduje o bankructwie</h2>
          </div>
          <p className="sub">
            Random Forest ocenia, które wskaźniki najsilniej wpływają na predykcję. Dominują miary zadłużenia,
            rentowności i zdolności do obsługi zobowiązań.
          </p>
          <div className="figure">
            <img src="/waznosc_cech.png" alt="Ważność wskaźników finansowych" />
            <p className="figcap">Najważniejsze wskaźniki według modelu Random Forest.</p>
          </div>
        </div>
      </section>

      {/* 04 DEMO */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">04</span>
            <h2>Model w działaniu</h2>
          </div>
          <p className="sub">
            Tak model (Gradient Boosting) ocenia dwa typowe profile firm na podstawie ich wskaźników
            finansowych — gotowe prawdopodobieństwo bankructwa dla każdego z nich.
          </p>

          <div className="examples">
            {przyklady.map((ex) => {
              const b = bandFor(ex.p);
              const frac = Math.min(1, ex.p / 0.4);
              return (
                <div className="excard" key={ex.nazwa}>
                  <div className="exname">{ex.nazwa}</div>
                  <div className="expct" style={{ color: b.kolor }}>
                    {fmtPct(ex.p)}
                  </div>
                  <div className="extag" style={{ background: b.kolor }}>
                    {b.txt}
                  </div>
                  <div className="exbar">
                    <div className="exbar-fill" style={{ width: `${frac * 100}%`, background: b.kolor }} />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="figcap" style={{ textAlign: "center" }}>
            Firmie zagrożonej model przypisuje kilkukrotnie wyższe ryzyko niż zdrowej (bazowa częstość
            bankructw w zbiorze: ~5%).
          </p>
        </div>
      </section>

      {/* 05 KOSZT */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">05</span>
            <h2>Analiza kosztowa</h2>
          </div>
          <p className="sub">
            Nie każdy błąd kosztuje tyle samo. Przeoczony bankrut (kredyt dla firmy, która upadnie) jest
            zwykle dużo droższy niż fałszywy alarm. Ustaw koszty i zobacz, który model jest najtańszy — to
            często <em>nie</em> ten o najwyższej trafności.
          </p>

          <div className="card">
            <div className="costgrid">
              <div className="field">
                <label>Koszt przeoczonego bankruta (fałszywie „zdrowa”)</label>
                <div className="inputwrap">
                  <input
                    type="number"
                    value={kosztFN}
                    min={0}
                    step={10000}
                    inputMode="numeric"
                    onChange={(e) => setKosztFN(e.target.value)}
                  />
                  <span className="unit">zł</span>
                </div>
                <div className="hint">strata na niespłaconym kredycie</div>
              </div>
              <div className="field">
                <label>Koszt fałszywego alarmu (zdrowa oznaczona „bankrut”)</label>
                <div className="inputwrap">
                  <input
                    type="number"
                    value={kosztFP}
                    min={0}
                    step={1000}
                    inputMode="numeric"
                    onChange={(e) => setKosztFP(e.target.value)}
                  />
                  <span className="unit">zł</span>
                </div>
                <div className="hint">utracona marża / koszt weryfikacji</div>
              </div>
            </div>

            <div className="tablecard">
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
        </div>
      </section>

      {/* 06 WNIOSKI */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">06</span>
            <h2>Wnioski</h2>
          </div>
          <ul className="wnioski">
            <li>
              <b>Trafność (accuracy) myli</b> przy ~5% bankrutów — KNN osiąga 95% trafności, wykrywając
              praktycznie zero bankrutów.
            </li>
            <li>
              <b>Równoważenie klas jest kluczowe</b> — dopiero modele uczone z wagami klas zaczynają wykrywać
              bankrutów.
            </li>
            <li>
              <b>Nie ma jednego „najlepszego” modelu</b> — Random Forest daje najlepszy balans (F1, AUC),
              a Gradient Boosting wyłapuje najwięcej bankrutów.
            </li>
            <li>
              <b>Analiza kosztowa zmienia ranking</b> — gdy przeoczony bankrut jest drogi, najtańszy okazuje
              się model o najwyższej czułości, a nie o najwyższej trafności.
            </li>
          </ul>
        </div>
      </section>

      {/* 07 JAK DZIAŁAJĄ MODELE */}
      <section>
        <div className="wrap">
          <div className="sec-head">
            <span className="sec-num">07</span>
            <h2>Jak działają modele</h2>
          </div>
          <p className="sub">
            Cztery modele klasyfikują firmę na różne sposoby. Poniżej uproszczone schematy pokazujące, jak
            każdy z nich podejmuje decyzję.
          </p>
          <div className="vizgrid">
            {JAK_DZIALAJA.map(({ nazwa, opis, Viz }) => (
              <div className="vizcard" key={nazwa}>
                <div className="vizimg">
                  <Viz />
                </div>
                <h3>{nazwa}</h3>
                <p>{opis}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer>
        <div className="wrap">
          Dane:{" "}
          <a href="https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data" target="_blank" rel="noreferrer">
            Polish Companies Bankruptcy Data (UCI)
          </a>{" "}
          · Kod:{" "}
          <a href={REPO} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <div className="disclaimer">
            Narzędzie demonstracyjne i edukacyjne. Predykcje opierają się na danych historycznych
            zanonimizowanych spółek i nie stanowią porady finansowej.
          </div>
        </div>
      </footer>
    </main>
  );
}
