import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const suggestions = [
  "What changed in the latest commits?",
  "Show me the open issues",
  "What are the latest pull requests?",
  "What project decisions are documented in Notion?",
  "What are the latest Jira issues for ECP?",
];

const sourceLabels = {
  github: "GitHub",
  gmail: "Gmail",
  notion: "Notion",
  jira: "Jira",
};

function evidenceCount(result) {
  const evidence = result?.result?.evidence;
  if (Array.isArray(evidence)) return evidence.length;
  if (evidence && typeof evidence === "object") {
    return Object.keys(evidence).length;
  }
  return evidence ? 1 : 0;
}

function Icon({ name, size = 18 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
  };

  const paths = {
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z" /><path d="m19 14 1.2 2.8L23 18l-2.8 1.2L19 22l-1.2-2.8L15 18l2.8-1.2L19 14Z" /></>,
    arrow: <><path d="M7 17 17 7" /><path d="M7 7h10v10" /></>,
    plus: <><path d="M12 5v14" /><path d="M5 12h14" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></>,
    github: <><path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 6v-3.9a3.4 3.4 0 0 0-.9-2.7c3-.3 6.2-1.5 6.2-6.8A5.3 5.3 0 0 0 19 4.9 4.9 4.9 0 0 0 18.9 1S17.7.7 15 2.5a13.5 13.5 0 0 0-7 0C5.3.7 4.1 1 4.1 1A4.9 4.9 0 0 0 4 4.9a5.3 5.3 0 0 0-1.3 3.7c0 5.3 3.2 6.5 6.2 6.8a3.4 3.4 0 0 0-.9 2.7V22" /></>,
    jira: <><path d="M12 3h8l-8 9 8 9h-8l-8-9 8-9Z" /><path d="M4 3h8l-8 9 8 9H4l-8-9 8-9Z" transform="translate(4 0)" /></>,
    gmail: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m3 7 9 6 9-6" /></>,
    check: <><path d="m5 12 4 4L19 6" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    chevron: <><path d="m9 18 6-6-6-6" /></>,
    external: <><path d="M14 3h7v7" /><path d="M10 14 21 3" /><path d="M21 14v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h6" /></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState(null);

  useEffect(() => {
    fetch("/api/sources")
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load source status.");
        return response.json();
      })
      .then(setSources)
      .catch(() => setSources({ api_unavailable: true }));
  }, []);

  async function ask(nextQuestion = question) {
    const trimmedQuestion = nextQuestion.trim();
    if (!trimmedQuestion || loading) return;

    setQuestion(trimmedQuestion);
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmedQuestion }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "The request could not be completed.");
      }
      setResult(payload);
      fetch("/api/sources")
        .then((sourceResponse) => sourceResponse.ok ? sourceResponse.json() : null)
        .then((nextSources) => {
          if (nextSources) setSources(nextSources);
        })
        .catch(() => {});
    } catch (requestError) {
      setError(requestError.message || "Unable to connect to the assistant.");
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    ask();
  }

  const successfulSources = result?.retrieval_results?.filter(
    (item) => item.status === "success",
  ).length ?? 0;
  const connectedSourceCount = ["github", "notion", "jira", "gmail"]
    .filter((source) => source === "gmail"
      ? sources?.gmail_authenticated
      : sources?.[source]).length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Orbit home">
          <span className="brand-mark"><Icon name="spark" size={21} /></span>
          <span>orbit<span className="brand-period">.</span></span>
        </a>

        <div className="workspace-switcher">
          <span className="workspace-avatar">A</span>
          <span className="workspace-copy"><strong>Workspace</strong></span>
        </div>

        <div className="nav-section-label">ORBIT</div>
        <nav className="main-nav" aria-label="Main navigation">
          <a className="nav-item active" href="#ask"><Icon name="spark" /><span>Ask Orbit</span></a>
          <a className="nav-item" href="#sources"><Icon name="github" /><span>Connected sources</span><span className="nav-count">{sources ? connectedSourceCount : "—"}</span></a>
        </nav>

        <div className="sidebar-bottom">
          <div className="source-connections" id="sources">
            {[
              { key: "github", label: "GitHub", icon: <Icon name="github" size={15} /> },
              { key: "notion", label: "Notion", icon: <span className="notion-mark">N</span> },
              { key: "jira", label: "Jira", icon: <Icon name="jira" size={15} /> },
              { key: "gmail", label: "Gmail", icon: <Icon name="gmail" size={15} /> },
            ].map((source) => {
              const configured = sources?.[source.key];
              const connected = source.key === "gmail"
                ? sources?.gmail_authenticated
                : configured;
              const status = !sources || sources.api_unavailable
                ? sources?.api_unavailable ? "API unavailable" : "checking..."
                : source.key === "gmail"
                  ? !configured
                    ? "not configured"
                    : connected
                      ? "authorization saved"
                      : "needs authorization"
                  : connected
                    ? "configured"
                    : "not configured";
              const detail = sources?.api_unavailable
                ? "Start the backend API"
                : source.key === "gmail"
                ? !configured
                  ? "Add OAuth client JSON"
                  : connected
                    ? "Google access checked on Gmail search"
                    : "Sign-in opens on first Gmail search"
                : "Read-only access";
              return (
                <div className="connection-card" key={source.key}>
                  <span className={`connection-icon ${source.key}`}>{source.icon}</span>
                  <div><strong>{source.label} {status}</strong><small>{detail}</small></div>
                  {connected && <span className="live-dot" />}
                </div>
              );
            })}
          </div>
          <button className="profile-row" type="button" aria-label="Profile">
            <span className="profile-avatar">SA</span>
            <span className="profile-copy"><strong>Shahab Alam</strong></span>
          </button>
        </div>
      </aside>

      <main className="main-area" id="ask">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><Icon name="chevron" size={14} /><strong>Ask Orbit</strong></div>
          <div className="topbar-right"><span className="top-avatar">SA</span></div>
        </header>

        <div className={`content ${result ? "content-results" : ""}`}>
          {!result && !loading && (
            <section className="welcome">
              <h1>Ask your workspace.</h1>

              <form className="ask-card" onSubmit={handleSubmit}>
                <label className="sr-only" htmlFor="question">Ask a question about your project</label>
                <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask anything about your project..." rows={3} maxLength={2000} />
                <div className="ask-card-footer">
                  <div className="ask-context">
                    {sources?.github && <span className="source-chip"><Icon name="github" size={15} /> GitHub</span>}
                    {sources?.notion && <span className="source-chip"><span className="notion-mark">N</span> Notion</span>}
                    {sources?.jira && <span className="source-chip"><Icon name="jira" size={15} /> Jira</span>}
                    {sources?.gmail && <span className="source-chip"><Icon name="gmail" size={15} /> Gmail</span>}
                    {(sources?.github || sources?.notion || sources?.jira) && <span className="context-divider" />}
                  </div>
                  <button className="ask-button" type="submit" disabled={!question.trim()}><span>Ask Orbit</span><Icon name="arrow" size={16} /></button>
                </div>
              </form>

              <div className="suggestions">{suggestions.slice(0, 3).map((suggestion) => <button key={suggestion} type="button" className="suggestion-chip" onClick={() => { setQuestion(suggestion); ask(suggestion); }}>{suggestion}<Icon name="arrow" size={13} /></button>)}</div>
            </section>
          )}

          {loading && (
            <section className="loading-state" aria-live="polite">
              <div className="loading-orbit"><span /><span /><span /></div>
              <h2>Retrieving evidence</h2>
              <div className="loading-steps"><span className="step-active"><Icon name="check" size={14} /> Decompose question</span><span className="step-active"><Icon name="search" size={14} /> Retrieve live data</span><span><Icon name="spark" size={14} /> Write cited answer</span></div>
            </section>
          )}

          {error && !loading && (
            <section className="error-panel" role="alert">
              <span className="error-mark">!</span><div><strong>That didn’t work</strong><p>{error}</p><button type="button" onClick={() => ask()}>Try again</button></div>
            </section>
          )}

          {result && !loading && (
            <section className="result-layout">
              <div className="answer-column">
                <button className="back-link" type="button" onClick={() => { setResult(null); setError(""); }}><span>←</span> Ask another question</button>
                <h1 className="result-question">{result.original_question}</h1>
                <div className="answer-card">
                  <div className="answer-heading"><span className="answer-icon"><Icon name="spark" size={17} /></span><span>ORBIT’S ANSWER</span><span className="grounded-badge"><Icon name="check" size={13} /> {result.citations?.length || 0} linked source{result.citations?.length === 1 ? "" : "s"}</span></div>
                  <div className="answer-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {result.final_answer || "No answer was generated."}
                    </ReactMarkdown>
                  </div>
                  {result.citations?.length > 0 && (
                    <div className="citation-list">
                      <div className="citation-heading">SOURCES <span>{result.citations.length}</span></div>
                      {result.citations.map((citation) => (
                        <a className="citation-item" key={citation.number} href={citation.url} target="_blank" rel="noreferrer">
                          <span className="citation-number">[{citation.number}]</span>
                          <span className="citation-details"><strong>{sourceLabels[citation.source] || citation.source}</strong><small>{citation.subquestion}</small></span>
                          <Icon name="external" size={15} />
                        </a>
                      ))}
                    </div>
                  )}
                </div>
                <button className="ask-again-button" type="button" onClick={() => { setResult(null); setQuestion(""); }}> <Icon name="plus" size={16} /> Ask another question</button>
              </div>

              <aside className="activity-column">
                <div className="activity-header"><div><span className="activity-kicker">BEHIND THE ANSWER</span><h2>Research trail</h2></div><span className="trail-check"><Icon name="check" size={15} /></span></div>
                <div className="trail-summary"><span className="live-dot" /> <strong>{successfulSources}</strong> successful retrieval{successfulSources === 1 ? "" : "s"}<span className="trail-summary-divider" /><strong>{result.citations?.length ?? 0}</strong> linked source{result.citations?.length === 1 ? "" : "s"}</div>
                <div className="trail-list">
                  <div className="trail-item"><span className="trail-index done"><Icon name="check" size={13} /></span><div><strong>Question decomposed</strong><small>{result.subquestions?.length ?? 0} focused subquestion{result.subquestions?.length === 1 ? "" : "s"}</small></div></div>
                  <div className="trail-item"><span className="trail-index done"><Icon name="check" size={13} /></span><div><strong>Sources selected</strong><small>{result.routes?.map((route) => route.sources?.map((source) => sourceLabels[source] || source).join(" + ")).filter(Boolean).join(", ") || "No sources"}</small></div></div>
                  {result.subquestions?.map((subquestion, index) => {
                    const retrievals = result.retrieval_results?.filter((item) => item.subquestion === subquestion) ?? [];
                    return retrievals.map((retrieval, retrievalIndex) => {
                      const succeeded = retrieval.status === "success";
                      const retrievalIcon = retrieval.source === "github" ? "github" : retrieval.source === "jira" ? "jira" : "search";
                      const count = evidenceCount(retrieval);
                      return <div className="retrieval-step" key={`${subquestion}-${index}-${retrievalIndex}`}><div className={`retrieval-source-icon ${succeeded ? "source-success" : "source-pending"}`}><Icon name={retrievalIcon} size={15} /></div><div className="retrieval-copy"><strong>{sourceLabels[retrieval.source] || retrieval.source} retrieval <span className="retrieval-count">{succeeded ? `${count} result${count === 1 ? "" : "s"}` : ""}</span></strong><small>{subquestion}</small>{retrieval.source === "jira" && retrieval.result?.jira_question && <small className="retrieval-detail"><b>Jira query:</b> {retrieval.result.jira_question}</small>}{retrieval.result?.tool && <small className="retrieval-detail"><b>Tool:</b> {retrieval.result.tool}</small>}{retrieval.error && <small className="retrieval-error">{retrieval.error}</small>}</div><span className={`retrieval-status ${succeeded ? "success" : "pending"}`}>{succeeded ? "DONE" : retrieval.status?.replace("_", " ") || "SKIPPED"}</span></div>;
                    });
                  })}
                  <div className="trail-item final-step"><span className="trail-index done"><Icon name="spark" size={13} /></span><div><strong>Answer prepared</strong><small>{result.citations?.length ?? 0} verified source citations</small></div></div>
                </div>
                <div className="trail-note"><Icon name="clock" size={15} /><span>Live data is retrieved when you ask. Results may change as your project changes.</span></div>
              </aside>
            </section>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
