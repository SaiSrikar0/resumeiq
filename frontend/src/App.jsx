import { useState, useEffect } from 'react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const ACCEPTED_FORMATS = ".pdf,.docx,.txt";
const FORMAT_LABEL = "PDF, DOCX, or TXT";

function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [customJdText, setCustomJdText] = useState('');
  const [isCustomJd, setIsCustomJd] = useState(false);
  const [resumeText, setResumeText] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('Java Developer');

  // Results
  const [scoreData, setScoreData] = useState(null);
  const [feedbackData, setFeedbackData] = useState(null);

  // UI States
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'feedback'
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const categories = [
    "Java Developer", "Python Developer", "Data Science", "DevOps", "SQL Developer",
    "Database", "Testing", "Web Designing", "React Developer", "Business Analyst",
    "DotNet Developer", "Software Developer", "ETL Developer", "Network Security Engineer",
    "Full Stack Developer", "Digital Media", "SAP Developer", "Cloud Engineer",
    "Machine Learning Engineer", "Frontend Developer", "Backend Developer", "AI Engineer",
    "Cybersecurity Analyst", "QA Engineer", "Database Administrator", "UI/UX Designer",
    "Site Reliability Engineer", "Mobile Developer", "System Administrator", "Blockchain",
    "Technical Lead", "Blockchain Developer", "Engineering Manager", "Principal Engineer",
    "Product Manager", "Technical Writer"
  ];

  useEffect(() => {
    fetch(`${BACKEND_URL}/jobs`)
      .then(res => res.json())
      .then(res => {
        if (res.status === 'success') {
          setJobs(res.jobs);
          if (res.jobs.length > 0) setSelectedJobId(res.jobs[0].jd_id);
        }
      })
      .catch(() => {
        setErrorMessage("Could not connect to FastAPI backend. Make sure it is running on port 8000.");
      });
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setResumeFile(e.target.files[0]);
      setResumeText('');
    }
  };

  const getFileIcon = () => {
    if (!resumeFile) return null;
    const name = resumeFile.name.toLowerCase();
    if (name.endsWith('.pdf'))  return '📄';
    if (name.endsWith('.docx')) return '📝';
    return '📃';
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage('');
    setErrorMessage('');
    setScoreData(null);
    setFeedbackData(null);

    try {
      // 1. Upload Resume
      let resumeId = '';
      const formData = new FormData();
      formData.append("category", selectedCategory);

      if (resumeFile) {
        formData.append("file", resumeFile);
      } else if (resumeText.trim()) {
        formData.append("text", resumeText.trim());
      } else {
        throw new Error("Please paste resume text or upload a PDF, DOCX, or TXT file first.");
      }

      const uploadRes = await fetch(`${BACKEND_URL}/resumes`, { method: 'POST', body: formData });
      if (!uploadRes.ok) {
        const errDetail = await uploadRes.json().catch(() => ({ detail: uploadRes.statusText }));
        throw new Error(errDetail.detail || "Upload failed.");
      }
      const uploadData = await uploadRes.json();
      resumeId = uploadData.resume_id;

      // 2. Score Match
      const scoreBody = { resume_id: resumeId };
      if (isCustomJd) {
        scoreBody.jd_text = customJdText.trim();
      } else {
        scoreBody.jd_id = selectedJobId;
      }

      const scoreRes = await fetch(`${BACKEND_URL}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scoreBody)
      });
      if (!scoreRes.ok) {
        const errDetail = await scoreRes.json().catch(() => ({ detail: scoreRes.statusText }));
        throw new Error(errDetail.detail || "Scoring failed.");
      }
      const scoreResult = await scoreRes.json();
      setScoreData(scoreResult);

      // 3. Feedback Report
      const feedbackRes = await fetch(`${BACKEND_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_id: resumeId,
          jd_id: isCustomJd ? null : selectedJobId,
          jd_text: isCustomJd ? customJdText.trim() : null
        })
      });
      if (!feedbackRes.ok) {
        const errDetail = await feedbackRes.json().catch(() => ({ detail: feedbackRes.statusText }));
        throw new Error(errDetail.detail || "Feedback generation failed.");
      }
      const feedbackResult = await feedbackRes.json();
      setFeedbackData(feedbackResult);

      setActiveTab('dashboard');
      setMessage("Analysis complete!");
    } catch (err) {
      console.error(err);
      setErrorMessage(err.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return { ring: 'border-emerald-500', text: 'text-emerald-400', bg: 'bg-emerald-500/10', label: 'Excellent Match', bar: 'bg-emerald-500' };
    if (score >= 50) return { ring: 'border-amber-500',   text: 'text-amber-400',   bg: 'bg-amber-500/10',   label: 'Moderate Match', bar: 'bg-amber-500' };
    return              { ring: 'border-rose-500',    text: 'text-rose-400',    bg: 'bg-rose-500/10',    label: 'Weak Match',     bar: 'bg-rose-500' };
  };

  const expMatchPct = (bd) => {
    if (!bd) return 0;
    if (bd.experience_gap <= 0) return 100;
    if (!bd.required_experience) return 0;
    return Math.max(0, (1 - bd.experience_gap / bd.required_experience) * 100);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-40 px-6 py-4 flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="bg-sky-500 text-slate-900 rounded px-1.5 py-0.5 font-black text-lg">IQ</span>
            ResumeIQ
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">AI-Powered Resume Match Screener &amp; Gap Analysis</p>
        </div>

        <div className="flex items-center gap-2 text-xs bg-amber-500/10 border border-amber-500/30 text-amber-300 px-3 py-1.5 rounded-lg max-w-xs">
          <span className="font-bold shrink-0">⚠ Disclaimer:</span>
          <span>All job descriptions are programmatically generated synthetic templates.</span>
        </div>
      </header>

      {/* ── Main Layout ─────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* ── Left: Input Panel ───────────────────────────── */}
        <section className="lg:col-span-4 flex flex-col gap-5">
          <div className="bg-slate-800/50 border border-slate-700/60 p-6 rounded-2xl flex flex-col gap-5 shadow-xl">
            <h2 className="text-base font-bold text-white border-b border-slate-700 pb-3 flex items-center gap-2">
              <span className="w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/50 text-sky-400 text-xs flex items-center justify-center font-bold">1</span>
              Candidate Profile
            </h2>

            <form onSubmit={handleAnalyze} className="flex flex-col gap-4">
              {/* Category */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Resume Category</label>
                <select
                  id="category-select"
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                >
                  {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                </select>
              </div>

              {/* File Upload */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                  Upload Resume <span className="text-sky-400 normal-case font-normal">({FORMAT_LABEL})</span>
                </label>
                <label
                  htmlFor="resume-file-input"
                  className={`flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-xl p-5 cursor-pointer transition-all ${
                    resumeFile
                      ? 'border-sky-500/50 bg-sky-500/5'
                      : 'border-slate-700 hover:border-sky-600/50 hover:bg-slate-800/60'
                  }`}
                >
                  <input
                    id="resume-file-input"
                    type="file"
                    accept={ACCEPTED_FORMATS}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  {resumeFile ? (
                    <div className="text-center">
                      <div className="text-2xl mb-1">{getFileIcon()}</div>
                      <div className="text-sky-400 font-semibold text-sm truncate max-w-[220px]">{resumeFile.name}</div>
                      <div className="text-slate-500 text-xs mt-0.5">{(resumeFile.size / 1024).toFixed(1)} KB — click to change</div>
                    </div>
                  ) : (
                    <>
                      <div className="text-3xl">📁</div>
                      <div className="text-center">
                        <div className="text-sky-400 font-semibold text-sm">Click to upload</div>
                        <div className="text-slate-500 text-xs mt-0.5">Supports PDF, DOCX, TXT</div>
                      </div>
                    </>
                  )}
                </label>
              </div>

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-slate-700"></div>
                <span className="text-xs text-slate-500 font-semibold uppercase tracking-widest">or paste text</span>
                <div className="flex-1 h-px bg-slate-700"></div>
              </div>

              {/* Text area */}
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Paste Resume Text</label>
                <textarea
                  id="resume-text-input"
                  value={resumeText}
                  onChange={(e) => { setResumeText(e.target.value); setResumeFile(null); }}
                  placeholder="Paste complete resume contents here..."
                  rows={7}
                  className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 font-mono resize-none transition-all"
                />
              </div>

              {/* JD Section */}
              <div className="border-t border-slate-700 pt-4 flex flex-col gap-3">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-sky-500/20 border border-sky-500/50 text-sky-400 text-xs flex items-center justify-center font-bold">2</span>
                  Target Job Description
                </h3>

                <div className="flex gap-4">
                  <label className="flex items-center gap-2 text-xs cursor-pointer text-slate-300">
                    <input type="radio" checked={!isCustomJd} onChange={() => setIsCustomJd(false)} className="accent-sky-500" />
                    Use JD Library
                  </label>
                  <label className="flex items-center gap-2 text-xs cursor-pointer text-slate-300">
                    <input type="radio" checked={isCustomJd} onChange={() => setIsCustomJd(true)} className="accent-sky-500" />
                    Paste Custom JD
                  </label>
                </div>

                {!isCustomJd ? (
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                      Select JD <span className="text-amber-400 font-normal normal-case">(Synthetic)</span>
                    </label>
                    <select
                      id="jd-select"
                      value={selectedJobId}
                      onChange={(e) => setSelectedJobId(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 transition-all"
                    >
                      {jobs.map(job => (
                        <option key={job.jd_id} value={job.jd_id}>
                          [{job.jd_id}] {job.title}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Custom Job Description</label>
                    <textarea
                      id="jd-text-input"
                      value={customJdText}
                      onChange={(e) => setCustomJdText(e.target.value)}
                      placeholder="Paste job description text here..."
                      rows={5}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500/30 resize-none transition-all"
                    />
                  </div>
                )}
              </div>

              {/* Submit */}
              <button
                id="analyze-btn"
                type="submit"
                disabled={isLoading}
                className="w-full bg-sky-600 hover:bg-sky-500 active:bg-sky-700 disabled:bg-slate-700 disabled:cursor-not-allowed transition-all text-white font-bold py-3 rounded-xl mt-1 flex items-center justify-center gap-2 text-sm shadow-lg shadow-sky-600/20"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analysing Resume...
                  </>
                ) : (
                  <> Screen Candidate Profile</>
                )}
              </button>
            </form>

            {message     && <div id="success-msg" className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 border border-emerald-500/30 px-3 py-2.5 rounded-xl">{message}</div>}
            {errorMessage && <div id="error-msg"   className="text-xs text-rose-400 font-semibold bg-rose-500/10 border border-rose-500/30 px-3 py-2.5 rounded-xl">{errorMessage}</div>}
          </div>
        </section>

        {/* ── Right: Results ───────────────────────────────── */}
        <section className="lg:col-span-8 flex flex-col gap-5">
          {scoreData ? (
            <>
              {/* Tab bar — only 2 tabs */}
              <div className="flex bg-slate-800/60 border border-slate-700/60 p-1 rounded-2xl gap-1">
                {[
                  { id: 'dashboard', label: '📊 Match Dashboard' },
                  { id: 'feedback',  label: '📋 Gap Analysis Report' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    id={`tab-${tab.id}`}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-1 py-2.5 text-sm font-semibold rounded-xl transition-all ${
                      activeTab === tab.id
                        ? 'bg-sky-600 text-white shadow-md shadow-sky-600/20'
                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* ── Dashboard Tab ─────────────────────────── */}
              {activeTab === 'dashboard' && (() => {
                const bd = scoreData.score_breakdown;
                const col = getScoreColor(bd.baseline_fit_score);
                return (
                  <div className="flex flex-col gap-5">

                    {/* Score hero */}
                    <div className="bg-slate-800/50 border border-slate-700/60 p-6 rounded-2xl grid grid-cols-1 md:grid-cols-12 gap-6 items-center shadow-lg">
                      {/* Big score circle */}
                      <div className="md:col-span-4 flex flex-col items-center text-center gap-3 md:border-r md:border-slate-700 md:pr-6">
                        <div className="text-xs font-bold text-slate-400 uppercase tracking-widest">Weighted Match Score</div>
                        <div className={`text-6xl font-black rounded-full w-36 h-36 flex items-center justify-center border-4 ${col.ring} ${col.text} ${col.bg} shadow-lg`}>
                          {bd.baseline_fit_score.toFixed(1)}%
                        </div>
                        <div className={`text-sm font-bold px-3 py-1 rounded-full border ${col.ring} ${col.text} ${col.bg}`}>
                          {col.label}
                        </div>
                        <div className="text-xs text-slate-500 font-mono">
                          vs <span className="text-slate-300">{scoreData.jd_details.title}</span>
                        </div>
                      </div>

                      {/* Sub-score bars */}
                      <div className="md:col-span-8 flex flex-col gap-4">
                        <h3 className="text-sm font-bold text-white">Feature Breakdown</h3>

                        {/* Skill overlap */}
                        <div>
                          <div className="flex justify-between text-xs font-semibold mb-1.5">
                            <span className="text-slate-300">Skill Overlap <span className="text-slate-500 font-normal">(40% weight)</span></span>
                            <span className="text-sky-400">{(bd.skill_overlap_ratio * 100).toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800">
                            <div className={`${col.bar} h-full rounded-full transition-all duration-700`} style={{ width: `${bd.skill_overlap_ratio * 100}%` }} />
                          </div>
                          <div className="text-[10px] text-slate-500 mt-1">
                            Matched {bd.matched_skills.length} of {bd.matched_skills.length + bd.missing_skills.length} required skills
                          </div>
                        </div>

                        {/* Experience */}
                        <div>
                          <div className="flex justify-between text-xs font-semibold mb-1.5">
                            <span className="text-slate-300">Experience Alignment <span className="text-slate-500 font-normal">(30% weight)</span></span>
                            <span className="text-sky-400">{expMatchPct(bd).toFixed(0)}%</span>
                          </div>
                          <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800">
                            <div className={`${col.bar} h-full rounded-full transition-all duration-700`} style={{ width: `${expMatchPct(bd)}%` }} />
                          </div>
                          <div className="text-[10px] text-slate-500 mt-1">
                            Candidate: {bd.candidate_experience.toFixed(1)} yrs &nbsp;|&nbsp; Required: {bd.required_experience.toFixed(1)} yrs
                            {bd.experience_gap > 0 ? ` (gap: ${bd.experience_gap.toFixed(1)} yrs)` : ' (meets requirement)'}
                          </div>
                        </div>

                        {/* Education + Semantic */}
                        <div className="grid grid-cols-2 gap-3 mt-1">
                          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3 flex items-center justify-between">
                            <div>
                              <div className="text-xs font-semibold text-slate-300">Education <span className="text-slate-500 font-normal">(15%)</span></div>
                              <div className="text-[10px] text-slate-500 mt-0.5">
                                Req: {bd.education_fit.required_education} · Has: {bd.education_fit.candidate_education}
                              </div>
                            </div>
                            <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-lg border ${bd.education_fit.is_match ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40' : 'bg-rose-500/20 text-rose-400 border-rose-500/40'}`}>
                              {bd.education_fit.is_match ? 'Meets' : 'Below'}
                            </span>
                          </div>

                          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
                            <div className="flex justify-between items-start">
                              <span className="text-xs font-semibold text-slate-300">Semantic Fit <span className="text-slate-500 font-normal">(15%)</span></span>
                              <span className="text-sky-400 font-bold text-sm">{bd.baseline_fit_score.toFixed(0)}%</span>
                            </div>
                            <div className="text-[10px] text-slate-500 mt-0.5">Vocabulary &amp; context similarity</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Skills grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      {/* Matched */}
                      <div className="bg-slate-800/50 border border-slate-700/60 p-5 rounded-2xl shadow">
                        <h3 className="text-sm font-bold text-emerald-400 mb-3 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                          Matched Skills ({bd.matched_skills.length})
                        </h3>
                        {bd.matched_skills.length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {bd.matched_skills.map(s => (
                              <span key={s} className="text-xs bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 px-2.5 py-1 rounded-lg">{s}</span>
                            ))}
                          </div>
                        ) : (
                          <div className="text-xs text-slate-500 italic">No matching technical skills detected.</div>
                        )}
                      </div>

                      {/* Missing */}
                      <div className="bg-slate-800/50 border border-slate-700/60 p-5 rounded-2xl shadow">
                        <h3 className="text-sm font-bold text-rose-400 mb-3 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-rose-400 inline-block" />
                          Missing Skills ({bd.missing_skills.length})
                        </h3>
                        {bd.missing_skills.length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {bd.missing_skills.map(s => (
                              <span key={s} className="text-xs bg-rose-500/10 border border-rose-500/30 text-rose-300 px-2.5 py-1 rounded-lg">{s}</span>
                            ))}
                          </div>
                        ) : (
                          <div className="text-xs text-emerald-400 font-semibold">✓ Perfect technical skills match!</div>
                        )}
                      </div>
                    </div>

                    {/* Natural language explanation + attention tokens */}
                    <div className="bg-slate-800/50 border border-slate-700/60 p-5 rounded-2xl shadow flex flex-col gap-4">
                      <div>
                        <h3 className="text-sm font-bold text-white mb-2">Natural Language Assessment</h3>
                        <p className="text-slate-300 text-sm leading-relaxed">{bd.explanation_text}</p>
                      </div>
                      {bd.attention_top_tokens?.length > 0 && (
                        <div className="border-t border-slate-700 pt-3">
                          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Neural Attention Tokens</div>
                          <div className="flex flex-wrap gap-2">
                            <span className="text-[10px] text-slate-600">Top matching vocab:</span>
                            {bd.attention_top_tokens.map(tok => (
                              <span key={tok} className="text-xs font-mono bg-sky-950/60 border border-sky-800/50 text-sky-400 px-2 py-0.5 rounded">{tok}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}

              {/* ── Gap Analysis Tab ──────────────────────── */}
              {activeTab === 'feedback' && (
                <div className="bg-slate-800/50 border border-slate-700/60 p-6 rounded-2xl shadow flex flex-col gap-4">
                  <div className="flex items-center justify-between border-b border-slate-700 pb-3">
                    <h2 className="text-base font-bold text-white">Retrieval-Grounded Gap Analysis</h2>
                    <span className="text-xs text-slate-400 font-mono bg-slate-900 border border-slate-700 px-2 py-1 rounded-lg">
                      {scoreData.jd_id} · {scoreData.jd_details.title}
                    </span>
                  </div>
                  {feedbackData ? (
                    <div className="text-xs md:text-sm text-slate-300 font-mono whitespace-pre-wrap leading-relaxed max-h-[620px] overflow-y-auto bg-slate-950/80 p-5 border border-slate-800 rounded-xl">
                      {feedbackData.feedback_report}
                    </div>
                  ) : (
                    <div className="text-slate-500 text-center py-14 italic">No gap report available.</div>
                  )}
                </div>
              )}
            </>
          ) : (
            /* Empty state */
            <div className="bg-slate-800/30 border border-dashed border-slate-700 p-10 rounded-2xl flex flex-col items-center justify-center text-center gap-5 min-h-[420px]">
              <div className="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center text-3xl">📄</div>
              <div>
                <h3 className="text-slate-300 font-bold text-lg mb-1">Ready to Analyse</h3>
                <p className="text-slate-500 text-sm max-w-sm">
                  Upload a <strong className="text-slate-400">PDF, DOCX, or TXT</strong> resume — or paste text — select a target JD, and click <em>Screen Candidate Profile</em>.
                </p>
              </div>
              <div className="flex gap-3 flex-wrap justify-center text-xs text-slate-500">
                <span className="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg">📊 Match Dashboard</span>
                <span className="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg">📋 Gap Analysis</span>
                <span className="bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg">🤖 Skill Breakdown</span>
              </div>
            </div>
          )}
        </section>

      </main>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="bg-slate-950 border-t border-slate-800 px-6 py-4 text-center text-xs text-slate-500">
        ResumeIQ — Powered by LightGBM · DistilBERT · FAISS · PyMuPDF · python-docx
      </footer>
    </div>
  );
}

export default App;
