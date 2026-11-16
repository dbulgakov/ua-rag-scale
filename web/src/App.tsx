import { type FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";

type Hit = { id: string; title: string; url: string; text: string };
type Answer = { text: string; citations: string[]; error: string | null };
type Check = { answers: string[]; passages: Hit[] };
type LinkGraph = { nodes: { id: number; title: string; url: string; seed: boolean }[]; edges: { source: number; target: number }[] };
type Result = { question: string; size: number; hits: Hit[]; answer: Answer; check: Check | null; graph: LinkGraph | null };
type Point = { size: number; recall: number };
type Series = { method: string; label: string; points: Point[]; language?: "uk" | "en"; dashed?: boolean };
type Recommendation = { size: string; use: string; avoid: string };
type Research = { curves: Series[]; english: Series[]; tokens: Record<"uk" | "en", Record<string, number>>; recommendations: Recommendation[]; router: { share: number; recall: number; random: number }[] | null };
type Meta = { sizes: number[]; examples: { id: string; question: string }[]; research: Research | null };
type Trace = { index_size: number; matches_hnswlib: boolean; query: number[]; nodes: { id: number; x: number; y: number; found: boolean }[]; steps: { level: number; from: number; to: number }[]; levels: number };

const COLORS = ["#15803d", "#ea580c", "#2563eb", "#7c3aed", "#db2777", "#0891b2"];
const sizeLabel = (size: number) => (size >= 1_000_000 ? `${size / 1_000_000} млн` : `${size / 1_000} тис.`);
const percent = (value: number) => `${Math.round(value * 100)}%`;
const millions = (tokens: Record<string, number>) => Math.round(Math.max(...Object.values(tokens)) / 1e6);

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(response.status === 502 ? "Сервер пошуку не запущено." : `Сервер повернув помилку ${response.status}.`);
  return response.json() as Promise<T>;
}

function Toggle({ label, value, onChange, on, off }: { label: string; value: boolean; onChange: (value: boolean) => void; on: string; off: string }) {
  return (
    <div className="mode" role="radiogroup" aria-label={label}>
      <button type="button" role="radio" aria-checked={value} className={value ? "active" : ""} onClick={() => onChange(true)}>{on}</button>
      <button type="button" role="radio" aria-checked={!value} className={!value ? "active" : ""} onClick={() => onChange(false)}>{off}</button>
    </div>
  );
}

function Step({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <li className="step">
      <div className="step-head"><span className="step-number">{number}</span><h3>{title}</h3></div>
      <div className="step-body">{children}</div>
    </li>
  );
}

function Passage({ hit, label, cited }: { hit: Hit; label?: string; cited?: boolean }) {
  return (
    <article className={`passage${cited ? " cited" : ""}`} id={label}>
      <header>{label && <span className="label">{label}</span>}<strong>{hit.title}</strong></header>
      <p>{hit.text.length > 420 ? `${hit.text.slice(0, 420)}…` : hit.text}</p>
      <a href={hit.url} target="_blank" rel="noreferrer">Відкрити статтю</a>
    </article>
  );
}

function LineChart({ series, selected, tokens }: { series: Series[]; selected: number; tokens?: Research["tokens"] }) {
  const width = 640, height = 300, left = 44, right = 16, top = 16, bottom = 36;
  const value = (item: Series, size: number) => (tokens ? tokens[item.language ?? "uk"][String(size)] : size);
  const xs = series.flatMap((item) => item.points.map((point) => value(item, point.size)));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const x = (raw: number) => left + ((Math.log10(raw) - Math.log10(minX)) / (Math.log10(maxX) - Math.log10(minX))) * (width - left - right);
  const ticks = tokens ? [1e6, 1e7, 1e8].filter((tick) => tick >= minX && tick <= maxX) : Array.from(new Set(series.flatMap((item) => item.points.map((point) => point.size)))).sort((a, b) => a - b);
  const values = series.flatMap((item) => item.points.map((point) => point.recall));
  const low = Math.floor(Math.min(...values) * 10) / 10, high = Math.ceil(Math.max(...values) * 10) / 10;
  const y = (value: number) => top + (1 - (value - low) / (high - low)) * (height - top - bottom);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label="Криві якості пошуку залежно від розміру бази">
      {[low, (low + high) / 2, high].map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="grid" /><text x={left - 6} y={y(tick) + 4} className="tick" textAnchor="end">{percent(tick)}</text></g>)}
      {ticks.map((tick) => <text key={tick} x={x(tick)} y={height - 12} className="tick" textAnchor="middle">{tokens ? `${sizeLabel(tick)} токенів` : sizeLabel(tick)}</text>)}
      {!tokens && <line x1={x(selected)} x2={x(selected)} y1={top} y2={height - bottom} className="cursor" />}
      {series.map((item, index) => (
        <g key={`${item.language ?? "uk"}-${item.method}-${index}`}>
          <polyline points={item.points.map((point) => `${x(value(item, point.size))},${y(point.recall)}`).join(" ")} fill="none" stroke={item.language === "en" ? "#6b7280" : COLORS[index % COLORS.length]} strokeWidth={2.5} strokeDasharray={item.method === "hybrid_lemma" || item.dashed ? "6 4" : undefined} />
          {item.points.map((point) => <circle key={point.size} cx={x(value(item, point.size))} cy={y(point.recall)} r={3.5} fill={item.language === "en" ? "#6b7280" : COLORS[index % COLORS.length]} />)}
        </g>
      ))}
    </svg>
  );
}

function RouterChart({ points }: { points: { share: number; recall: number; random: number }[] }) {
  const width = 640, height = 280, left = 44, right = 16, top = 16, bottom = 36;
  const values = points.flatMap((point) => [point.recall, point.random]);
  const low = Math.floor(Math.min(...values) * 50) / 50, high = Math.ceil(Math.max(...values) * 50) / 50;
  const x = (share: number) => left + share * (width - left - right);
  const y = (value: number) => top + (1 - (value - low) / (high - low || 1)) * (height - top - bottom);
  const line = (key: "recall" | "random") => points.map((point) => `${x(point.share)},${y(point[key])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label="Якість маршрутизатора проти випадкового вибору">
      {[0, 0.25, 0.5, 0.75, 1].map((share) => <text key={share} x={x(share)} y={height - 12} className="tick" textAnchor="middle">{percent(share)}</text>)}
      {[low, high].map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="grid" /><text x={left - 6} y={y(tick) + 4} className="tick" textAnchor="end">{percent(tick)}</text></g>)}
      <polyline points={line("random")} fill="none" stroke="#9aa59f" strokeWidth={2} strokeDasharray="6 4" />
      <polyline points={line("recall")} fill="none" stroke="#27634d" strokeWidth={2.5} />
    </svg>
  );
}

function HnswAnimation({ trace }: { trace: Trace }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    setShown(0);
    const timer = window.setInterval(() => setShown((value) => (value >= trace.steps.length ? value : value + 4)), 30);
    return () => window.clearInterval(timer);
  }, [trace]);
  const position = useMemo(() => new Map(trace.nodes.map((node) => [node.id, node])), [trace]);
  const size = 360, pad = 14, scale = (value: number) => pad + value * (size - 2 * pad);
  const done = shown >= trace.steps.length;
  return (
    <div>
      <svg viewBox={`0 0 ${size} ${size}`} className="hnsw" role="img" aria-label="Анімація пошуку HNSW">
        {trace.steps.slice(0, shown).map((step, index) => {
          const from = position.get(step.from)!, to = position.get(step.to)!;
          return <line key={index} x1={scale(from.x)} y1={scale(from.y)} x2={scale(to.x)} y2={scale(to.y)} className={step.level > 0 ? "edge upper" : "edge"} />;
        })}
        {trace.nodes.map((node) => <circle key={node.id} cx={scale(node.x)} cy={scale(node.y)} r={done && node.found ? 5 : 2.2} className={done && node.found ? "node found" : "node"} />)}
        <circle cx={scale(trace.query[0])} cy={scale(trace.query[1])} r={6} className="query" />
      </svg>
      <p className="note">Демонстраційний індекс HNSW на {sizeLabel(trace.index_size)} уривків, {trace.levels + 1} рівні. Помаранчеві лінії — швидкі переходи на верхніх рівнях, зелені — уточнення на нижньому рівні. Великі точки — знайдені уривки. {trace.matches_hnswlib ? "Шлях відтворено зі зв'язків індексу; результат збігається з бібліотекою hnswlib." : "Увага: результат відтворення не збігся з hnswlib."}</p>
      <button type="button" className="ghost" onClick={() => setShown(0)}>Повторити анімацію</button>
    </div>
  );
}

function ArticleGraph({ graph }: { graph: LinkGraph }) {
  const size = 420, center = size / 2;
  const seeds = graph.nodes.filter((node) => node.seed), others = graph.nodes.filter((node) => !node.seed);
  const place = (list: typeof graph.nodes, radius: number) => list.map((node, index) => ({ ...node, x: center + radius * Math.cos((2 * Math.PI * index) / Math.max(list.length, 1) - Math.PI / 2), y: center + radius * Math.sin((2 * Math.PI * index) / Math.max(list.length, 1) - Math.PI / 2) }));
  const nodes = [...place(seeds, 90), ...place(others, 175)];
  const at = new Map(nodes.map((node) => [node.id, node]));
  return (
    <div>
      <svg viewBox={`0 0 ${size} ${size}`} className="graph" role="img" aria-label="Підграф посилань між статтями">
        {graph.edges.map((edge, index) => <line key={index} x1={at.get(edge.source)!.x} y1={at.get(edge.source)!.y} x2={at.get(edge.target)!.x} y2={at.get(edge.target)!.y} className="link" />)}
        {nodes.map((node) => (
          <a key={node.id} href={node.url} target="_blank" rel="noreferrer">
            <circle cx={node.x} cy={node.y} r={node.seed ? 7 : 4.5} className={node.seed ? "article seed" : "article"} />
            <text x={node.x} y={node.y - 10} textAnchor="middle" className="article-title">{node.title.length > 22 ? `${node.title.slice(0, 21)}…` : node.title}</text>
          </a>
        ))}
      </svg>
      <p className="note">Темні вузли — статті знайдених уривків, світлі — статті, на які вони найчастіше посилаються. Лінії — посилання Вікіпедії (знімок 2022 року). Натисніть на вузол, щоб відкрити статтю.</p>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [question, setQuestion] = useState("");
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [retrieval, setRetrieval] = useState(true);
  const [size, setSize] = useState(1_000_000);
  const [result, setResult] = useState<Result | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [curveIndex, setCurveIndex] = useState(4);
  const [tokenAxis, setTokenAxis] = useState(false);

  useEffect(() => { request<Meta>("/meta").then(setMeta).catch((reason: Error) => setError(reason.message)); }, []);

  const run = async (text: string, example: string | null) => {
    if (!text.trim()) return;
    setLoading(true); setError(""); setTrace(null);
    try {
      setResult(await request<Result>("/search", { question: text, size, retrieval, example_id: example }));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setLoading(false);
    }
  };
  const submit = (event: FormEvent) => { event.preventDefault(); run(question, exampleId); };
  const choose = (example: { id: string; question: string }) => { setQuestion(example.question); setExampleId(example.id); run(example.question, example.id); };
  const loadTrace = () => { if (!trace && question.trim()) request<Trace>("/trace", { question }).then(setTrace).catch((reason: Error) => setError(reason.message)); };

  const curves = meta?.research?.curves ?? [];
  const curveSizes = curves[0]?.points.map((point) => point.size) ?? [];
  const selectedSize = curveSizes[Math.min(curveIndex, curveSizes.length - 1)] ?? 1_000_000;

  return (
    <main>
      <header className="hero">
        <h1>Пошук відповідей у Вікіпедії</h1>
        <p>Система знаходить уривки в українській Вікіпедії і складає коротку відповідь з посиланнями на них.</p>
      </header>

      <form className="search" onSubmit={submit}>
        <input value={question} onChange={(event) => { setQuestion(event.target.value); setExampleId(null); }} placeholder="Поставте запитання українською" aria-label="Запитання" />
        <button type="submit" disabled={loading || !question.trim()}>{loading ? "Шукаю…" : "Знайти"}</button>
      </form>

      {meta && (
        <div className="examples">
          <span>Приклади:</span>
          {meta.examples.map((example) => <button type="button" key={example.id} className="chip" onClick={() => choose(example)}>{example.question}</button>)}
        </div>
      )}

      <Toggle label="Режим" value={retrieval} onChange={setRetrieval} on="З пошуком" off="Без пошуку" />

      <details className="settings">
        <summary>Додаткові налаштування</summary>
        <label>Розмір бази для пошуку <select value={size} onChange={(event) => setSize(Number(event.target.value))}>{(meta?.sizes ?? [10_000, 100_000, 1_000_000]).map((value) => <option key={value} value={value}>{sizeLabel(value)} уривків</option>)}</select></label>
        <p className="note">Індекс HNSW (M = 32, efSearch = 128), векторна модель multilingual-e5-small, генератор Qwen3-235B через OpenRouter (провайдер DeepInfra, fp8).</p>
      </details>

      {error && <p className="error">{error}</p>}

      {result && (
        <ol className="flow">
          <Step number={1} title="Запитання"><p className="question">{result.question}</p></Step>
          <Step number={2} title="Пошук у статтях Вікіпедії">
            <p>{result.hits.length ? `Переглянуто базу з ${sizeLabel(result.size)} уривків і вибрано 5 найближчих за змістом.` : "Пошук вимкнено: модель відповідає лише з власних знань, без джерел."}</p>
          </Step>
          {result.hits.length > 0 && (
            <Step number={3} title="Знайдені уривки">
              <div className="passages">{result.hits.map((hit, index) => <Passage key={hit.id} hit={hit} label={`D${index + 1}`} cited={result.answer.citations.includes(`D${index + 1}`)} />)}</div>
            </Step>
          )}
          <Step number={result.hits.length ? 4 : 3} title={result.hits.length ? "Відповідь із посиланнями" : "Відповідь"}>
            {result.answer.error && !result.answer.text ? <p className="error">{result.answer.error}</p> : (
              <p className="answer">{result.answer.text || "Модель не змогла визначити відповідь."} {result.answer.citations.map((label) => <a key={label} href={`#${label}`} className="label">{label}</a>)}</p>
            )}
          </Step>
        </ol>
      )}

      {result?.check && (
        <section className="check">
          <h2>Перевірка</h2>
          <p>Це запитання з дослідницької вибірки, тому відома правильна відповідь: <strong>{result.check.answers.join(" / ")}</strong></p>
          {result.check.passages.map((hit) => <Passage key={hit.id} hit={hit} />)}
        </section>
      )}

      <section className="levels">
        <details onToggle={(event) => (event.currentTarget as HTMLDetailsElement).open && loadTrace()}>
          <summary>Як працює пошук</summary>
          <p>Уривки зберігаються як точки у просторі змісту. HNSW — це граф, у якому кожна точка з'єднана з близькими сусідами; пошук стрибає графом до точок, найближчих до запитання, і не перебирає всю базу.</p>
          {trace ? <HnswAnimation trace={trace} /> : <p className="note">{question.trim() ? "Завантажую шлях пошуку…" : "Спершу поставте запитання."}</p>}
        </details>
        <details>
          <summary>Зв'язки статей</summary>
          {result?.graph?.nodes.length ? <ArticleGraph graph={result.graph} /> : <p className="note">Підграф з'явиться після пошуку.</p>}
        </details>
        <details>
          <summary>Що відбувається з великою базою</summary>
          <p>Чим більше уривків у базі, тим більше схожих, але неправильних кандидатів, і тим рідше правильний уривок потрапляє в перші п'ять. Криві — частка запитань тестової вибірки, для яких правильний уривок знайдено в перших п'яти.</p>
          {curves.length > 0 && (
            <>
              <input type="range" min={0} max={curveSizes.length - 1} value={Math.min(curveIndex, curveSizes.length - 1)} onChange={(event) => setCurveIndex(Number(event.target.value))} aria-label="Розмір бази" />
              <p className="readout">Розмір бази: <strong>{sizeLabel(selectedSize)} уривків</strong></p>
              <Toggle label="Вісь графіка" value={!tokenAxis} onChange={(passages) => setTokenAxis(!passages)} on="Вісь: уривки" off="Вісь: токени" />
              <LineChart series={tokenAxis ? [...curves, ...(meta?.research?.english ?? []).map((item) => ({ ...item, language: "en" as const }))] : curves} selected={selectedSize} tokens={tokenAxis ? meta?.research?.tokens : undefined} />
              {tokenAxis && <p className="note">На осі «токени» українська база з 1 млн уривків — це {millions(meta!.research!.tokens.uk)} млн токенів, англійська — {millions(meta!.research!.tokens.en)} млн: українські уривки довші. Сірі лінії — англійський контроль (HotpotQA, точний пошук): суцільна — частка знайдених доказів (Recall@5), пунктирна — частка запитань, де знайдено хоча б один доказ (Hit@5). Різниця між кривими — це різниця двох бенчмарків, а не лише мов.</p>}
              <ul className="legend">
                {curves.map((item, index) => {
                  const point = item.points.find((entry) => entry.size === selectedSize);
                  return <li key={item.method}><span style={{ background: COLORS[index % COLORS.length] }} />{item.label}: {point ? percent(point.recall) : "не вимірювалося"}</li>;
                })}
              </ul>
              <p className="note">Кольоровий пунктир — додатковий аналіз після основного протоколу. Переранжування вимірювалося лише на 10 тис., 100 тис. і 1 млн.</p>
            </>
          )}
        </details>
        <details>
          <summary>Що обрати для своєї бази</summary>
          <p>Практичні висновки з вимірювань (українська Вікіпедія, частка запитань, для яких правильний уривок у перших п'яти).</p>
          {meta?.research?.recommendations ? (
            <div className="table-wrap"><table className="recommendations"><thead><tr><th>Розмір бази</th><th>Що використати</th><th>Чого уникати</th></tr></thead><tbody>{meta.research.recommendations.map((row) => <tr key={row.size}><td>{row.size}</td><td>{row.use}</td><td>{row.avoid}</td></tr>)}</tbody></table></div>
          ) : <p className="note">Рекомендації ще формуються.</p>}
        </details>
        <details>
          <summary>Розумний вибір методу</summary>
          <p>Дорогі методи (переранжування, граф) допомагають переважно «тісним» запитанням, де поруч багато схожих уривків. Маршрутизатор за ознаками швидкого пошуку вирішує, коли їх вмикати.</p>
          {meta?.research?.router ? (
            <>
              <RouterChart points={meta.research.router} />
              <p className="note">Суцільна лінія — маршрутизатор, пунктир — випадковий вибір з тією ж часткою дорогих запитів. По горизонталі — частка запитів, відправлених на дорогий метод.</p>
            </>
          ) : <p className="note">Результати маршрутизатора ще обчислюються.</p>}
        </details>
      </section>

      <footer>Джерела — лише статті uk.wikipedia.org. Дослідження: «Деградація якості пошуку в RAG-системах при зростанні корпусу та графові методи її компенсації».</footer>
    </main>
  );
}
