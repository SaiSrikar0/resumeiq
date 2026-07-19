import { useState, useEffect, useRef } from 'react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [customJdText, setCustomJdText] = useState('');
  const [isCustomJd, setIsCustomJd] = useState(false);
  const [resumeText, setResumeText] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('Java Developer');
  
  // Results
  const [activeResumeId, setActiveResumeId] = useState('');
  const [activeJdId, setActiveJdId] = useState('');
  const [scoreData, setScoreData] = useState(null);
  const [feedbackData, setFeedbackData] = useState(null);
  
  // Chat
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [sessionId] = useState(() => 'session_' + Math.random().toString(36).substr(2, 9));
  
  // UI States
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard' | 'feedback' | 'chat'
  const [isLoading, setIsLoading] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const chatEndRef = useRef(null);

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
    // Retrieve JDs
    fetch(`${BACKEND_URL}/jobs`)
      .then(res => res.json())
      .then(res => {
        if (res.status === 'success') {
          setJobs(res.jobs);
          if (res.jobs.length > 0) {
            setSelectedJobId(res.jobs[0].jd_id);
          }
        }
      })
      .catch(err => {
        console.error("Failed to load jobs from backend:", err);
        setErrorMessage("Could not connect to FastAPI backend. Make sure it is running on port 8000.");
      });
  }, []);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setResumeFile(e.target.files[0]);
      setResumeText(''); // clear text pasted if file is selected
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage('');
    setErrorMessage('');
    setScoreData(null);
    setFeedbackData(null);
    setChatHistory([]);

    try {
      // 1. Upload Resume
      let resumeId = '';
      if (resumeFile) {
        const formData = new FormData();
        formData.append("file", resumeFile);
        formData.append("category", selectedCategory);
        
        const uploadRes = await fetch(`${BACKEND_URL}/resumes`, {
          method: 'POST',
          body: formData
        });
        if (!uploadRes.ok) throw new Error(await uploadRes.text());
        const uploadData = await uploadRes.json();
        resumeId = uploadData.resume_id;
      } else if (resumeText) {
        const formData = new FormData();
        formData.append("text", resumeText);
        formData.append("category", selectedCategory);
        
        const uploadRes = await fetch(`${BACKEND_URL}/resumes`, {
          method: 'POST',
          body: formData
        });
        if (!uploadRes.ok) throw new Error(await uploadRes.text());
        const uploadData = await uploadRes.json();
        resumeId = uploadData.resume_id;
      } else {
        throw new Error("Please paste resume text or select a file first.");
      }

      setActiveResumeId(resumeId);

      // 2. Score Match
      const scoreBody = { resume_id: resumeId };
      if (isCustomJd) {
        if (!customJdText.strip) {
          scoreBody.jd_text = customJdText;
        } else {
          scoreBody.jd_text = customJdText.trim();
        }
      } else {
        scoreBody.jd_id = selectedJobId;
      }

      const scoreRes = await fetch(`${BACKEND_URL}/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scoreBody)
      });
      if (!scoreRes.ok) throw new Error(await scoreRes.text());
      const scoreDataResult = await scoreRes.json();
      setScoreData(scoreDataResult);
      setActiveJdId(scoreDataResult.jd_id);

      // 3. Generate Feedback Report
      const feedbackRes = await fetch(`${BACKEND_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_id: resumeId,
          jd_id: isCustomJd ? null : selectedJobId,
          jd_text: isCustomJd ? customJdText : null
        })
      });
      if (!feedbackRes.ok) throw new Error(await feedbackRes.text());
      const feedbackDataResult = await feedbackRes.json();
      setFeedbackData(feedbackDataResult);
      
      // Initialize welcome chat message
      setChatHistory([
        {
          sender: 'agent',
          text: `Hello! I am ResumeIQ, your AI career coach. I have evaluated your profile against the '${scoreDataResult.jd_details.title}' requirements and calculated a fit score of ${scoreDataResult.score_breakdown.baseline_fit_score.toFixed(1)}%. Ask me anything about your strengths, skill gaps, or how to optimize your resume bullets!`,
          tools: []
        }
      ]);
      
      setActiveTab('dashboard');
      setMessage("Resume analyzed successfully!");
    } catch (err) {
      console.error(err);
      setErrorMessage(err.message || "An error occurred during analysis.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput.trim() || isChatLoading) return;

    const userMessageText = chatInput;
    setChatInput('');
    setIsChatLoading(true);
    
    // Add user message to history
    setChatHistory(prev => [...prev, { sender: 'user', text: userMessageText, tools: [] }]);

    try {
      const response = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resume_id: activeResumeId,
          jd_id: activeJdId,
          session_id: sessionId,
          message: userMessageText
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      
      setChatHistory(prev => [...prev, {
        sender: 'agent',
        text: data.response,
        tools: data.tool_calls || []
      }]);
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, {
        sender: 'agent',
        text: "Sorry, I encountered an error communicating with the chat agent. Please try again.",
        tools: []
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const sendQuickPrompt = (promptText) => {
    setChatInput(promptText);
    setTimeout(() => {
      // Simulate submit
      const inputForm = document.getElementById("chat-form");
      if (inputForm) {
        inputForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
      }
    }, 50);
  };

  // Helper to color fit score badge
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-emerald-400 border-emerald-500 bg-emerald-500/10';
    if (score >= 50) return 'text-amber-400 border-amber-500 bg-amber-500/10';
    return 'text-rose-400 border-rose-500 bg-rose-500/10';
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="bg-sky-500 text-slate-900 rounded px-1.5 py-0.5 font-black text-lg">IQ</span>
            ResumeIQ
          </h1>
          <p className="text-xs text-slate-400">AI-Powered Intelligent Resume Match Screener & Feedback Dashboard</p>
        </div>
        <div className="flex gap-3 text-xs bg-slate-800/50 border border-slate-700 px-3 py-1.5 rounded-lg max-w-sm">
          <span className="text-amber-500 font-semibold uppercase tracking-wider">Disclaimer:</span>
          <span className="text-slate-300">All matching jobs in this workspace are programmatically generated synthetic templates.</span>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Upload / Paste & Controls */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          <div className="glass-panel p-6 rounded-xl flex flex-col gap-4">
            <h2 className="text-lg font-bold text-white border-b border-slate-700 pb-2">1. Load Candidate Profile</h2>
            
            <form onSubmit={handleAnalyze} className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Resume Category</label>
                <select 
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-sky-500"
                >
                  {categories.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Upload Plain-Text Resume</label>
                <div className="border border-dashed border-slate-700 rounded-lg p-4 text-center cursor-pointer hover:border-sky-500 transition-colors">
                  <input 
                    type="file" 
                    accept=".txt"
                    onChange={handleFileChange}
                    className="hidden" 
                    id="resume-file-input"
                  />
                  <label htmlFor="resume-file-input" className="cursor-pointer block text-xs text-slate-400">
                    {resumeFile ? (
                      <span className="text-sky-400 font-semibold block truncate">{resumeFile.name}</span>
                    ) : (
                      <>
                        <span className="text-sky-400 font-semibold block mb-1">Click to upload TXT file</span>
                        (UTF-8 plain text format)
                      </>
                    )}
                  </label>
                </div>
              </div>

              <div className="text-center text-xs text-slate-500 font-semibold uppercase tracking-widest">— OR —</div>

              <div>
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Paste Resume Text</label>
                <textarea
                  value={resumeText}
                  onChange={(e) => {
                    setResumeText(e.target.value);
                    setResumeFile(null); // clear file upload
                  }}
                  placeholder="Paste complete plain-text OCR or parsed resume contents here..."
                  rows={8}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500 font-mono resize-none"
                />
              </div>

              <div className="border-t border-slate-800 pt-4 flex flex-col gap-3">
                <h3 className="text-sm font-bold text-white">2. Target Requirements</h3>

                <div className="flex gap-4 items-center mb-2">
                  <label className="flex items-center gap-2 text-xs cursor-pointer text-slate-300">
                    <input 
                      type="radio" 
                      checked={!isCustomJd} 
                      onChange={() => setIsCustomJd(false)}
                      className="accent-sky-500" 
                    />
                    Use Synthetic JD List
                  </label>
                  <label className="flex items-center gap-2 text-xs cursor-pointer text-slate-300">
                    <input 
                      type="radio" 
                      checked={isCustomJd} 
                      onChange={() => setIsCustomJd(true)}
                      className="accent-sky-500" 
                    />
                    Paste Custom JD
                  </label>
                </div>

                {!isCustomJd ? (
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      Select Target JD <span className="text-amber-500 font-black ml-1">(Synthetic)</span>
                    </label>
                    <select
                      value={selectedJobId}
                      onChange={(e) => setSelectedJobId(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                    >
                      {jobs.map(job => (
                        <option key={job.jd_id} value={job.jd_id}>
                          [{job.jd_id}] {job.title} ({job.category})
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  <div>
                    <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Custom Job Description Text</label>
                    <textarea
                      value={customJdText}
                      onChange={(e) => setCustomJdText(e.target.value)}
                      placeholder="Paste target job description details (skills, experience requirements) here..."
                      rows={5}
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500 resize-none"
                    />
                  </div>
                )}
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 transition-colors text-white font-bold py-2.5 rounded-lg mt-2 flex items-center justify-center gap-2 text-sm shadow-lg shadow-sky-600/10"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Profile...
                  </>
                ) : (
                  "Screen Candidate Profile"
                )}
              </button>
            </form>
            
            {message && <div className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 border border-emerald-500/30 p-2.5 rounded-lg">{message}</div>}
            {errorMessage && <div className="text-xs text-rose-400 font-semibold bg-rose-500/10 border border-rose-500/30 p-2.5 rounded-lg">{errorMessage}</div>}
          </div>
        </section>

        {/* Right Column: Dynamic Tabs & Display Dashboard */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          {scoreData ? (
            <>
              {/* Tab Navigation */}
              <div className="flex border-b border-slate-800 bg-slate-950/40 p-1 rounded-xl">
                <button
                  onClick={() => setActiveTab('dashboard')}
                  className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${activeTab === 'dashboard' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}
                >
                  Match Dashboard
                </button>
                <button
                  onClick={() => setActiveTab('feedback')}
                  className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${activeTab === 'feedback' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}
                >
                  Gap Report
                </button>
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${activeTab === 'chat' ? 'bg-sky-600 text-white shadow-md' : 'text-slate-400 hover:text-white'}`}
                >
                  Career Coach Chat
                </button>
              </div>

              {/* Tab Content 1: Match Dashboard */}
              {activeTab === 'dashboard' && (
                <div className="flex flex-col gap-6 animate-fade-in">
                  
                  {/* Overall Fit Score Card */}
                  <div className="glass-panel p-6 rounded-xl grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                    <div className="md:col-span-4 flex flex-col items-center justify-center text-center border-r md:border-slate-800 pr-0 md:pr-6 gap-2">
                      <div className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Weighted Match</div>
                      <div className={`text-5xl font-black rounded-full px-5 py-4 border-2 ${getScoreColor(scoreData.score_breakdown.baseline_fit_score)}`}>
                        {scoreData.score_breakdown.baseline_fit_score.toFixed(1)}%
                      </div>
                      <div className="text-xs font-bold text-slate-300">
                        {scoreData.score_breakdown.baseline_fit_score >= 80 ? "Excellent Match" : scoreData.score_breakdown.baseline_fit_score >= 50 ? "Moderate Match" : "Weak Match"}
                      </div>
                    </div>
                    
                    <div className="md:col-span-8 flex flex-col gap-4">
                      <h3 className="text-base font-bold text-white">Composite Feature Breakdown:</h3>
                      
                      {/* Sub-scores */}
                      <div className="flex flex-col gap-3 text-xs">
                        {/* 1. Skill Overlap */}
                        <div>
                          <div className="flex justify-between font-semibold mb-1">
                            <span className="text-slate-300">Skill Overlap (40% Weight)</span>
                            <span className="text-sky-400">{(scoreData.score_breakdown.skill_overlap_ratio*100).toFixed(0)}% Match</span>
                          </div>
                          <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                            <div className="bg-sky-500 h-full rounded-full" style={{ width: `${scoreData.score_breakdown.skill_overlap_ratio*100}%` }}></div>
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">
                            Matched {scoreData.score_breakdown.matched_skills.length} of {scoreData.score_breakdown.matched_skills.length + scoreData.score_breakdown.missing_skills.length} core technical requirements.
                          </div>
                        </div>

                        {/* 2. Experience Match */}
                        <div>
                          <div className="flex justify-between font-semibold mb-1">
                            <span className="text-slate-300">Experience Alignment (30% Weight)</span>
                            <span className="text-sky-400">
                              {scoreData.score_breakdown.experience_gap <= 0 ? "100% Match" : `${Math.max(0, (1 - scoreData.score_breakdown.experience_gap / scoreData.score_breakdown.required_experience)*100).toFixed(0)}% Match`}
                            </span>
                          </div>
                          <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
                            <div 
                              className="bg-sky-500 h-full rounded-full" 
                              style={{ width: `${scoreData.score_breakdown.experience_gap <= 0 ? 100 : Math.max(0, 1 - scoreData.score_breakdown.experience_gap / scoreData.score_breakdown.required_experience)*100}%` }}
                            ></div>
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">
                            Candidate has {scoreData.score_breakdown.candidate_experience.toFixed(1)} years | Role requires {scoreData.score_breakdown.required_experience.toFixed(1)} years (Gap: {scoreData.score_breakdown.experience_gap > 0 ? `${scoreData.score_breakdown.experience_gap.toFixed(1)} yrs` : 'None'}).
                          </div>
                        </div>

                        {/* 3. Degree Match & 4. Model Semantic Score */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                          <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between">
                            <div>
                              <div className="font-semibold text-slate-300">Education (15% Weight)</div>
                              <div className="text-[10px] text-slate-400">Req: {scoreData.score_breakdown.education_fit.required_education} | Has: {scoreData.score_breakdown.education_fit.candidate_education}</div>
                            </div>
                            <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${scoreData.score_breakdown.education_fit.is_match ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                              {scoreData.score_breakdown.education_fit.is_match ? 'Meets' : 'Below'}
                            </span>
                          </div>

                          <div className="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5 flex flex-col justify-between">
                            <div className="flex justify-between items-center">
                              <span className="font-semibold text-slate-300">Semantic Fit (15% Weight)</span>
                              <span className="text-sky-400 font-bold">{scoreData.score_breakdown.baseline_fit_score.toFixed(0)}%</span>
                            </div>
                            <div className="text-[10px] text-slate-400">Vocabulary & context matching probability score.</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Skills Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="glass-panel p-6 rounded-xl">
                      <h3 className="text-sm font-bold text-emerald-400 flex items-center gap-2 mb-3">
                        <span className="bg-emerald-500/20 w-2 h-2 rounded-full inline-block"></span>
                        Matched Skills ({scoreData.score_breakdown.matched_skills.length})
                      </h3>
                      {scoreData.score_breakdown.matched_skills.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {scoreData.score_breakdown.matched_skills.map(s => (
                            <span key={s} className="text-xs bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 px-2 py-1 rounded-lg">{s}</span>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-slate-500">No matching technical skills identified.</div>
                      )}
                    </div>

                    <div className="glass-panel p-6 rounded-xl">
                      <h3 className="text-sm font-bold text-rose-400 flex items-center gap-2 mb-3">
                        <span className="bg-rose-500/20 w-2 h-2 rounded-full inline-block"></span>
                        Missing Skills ({scoreData.score_breakdown.missing_skills.length})
                      </h3>
                      {scoreData.score_breakdown.missing_skills.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {scoreData.score_breakdown.missing_skills.map(s => (
                            <span key={s} className="text-xs bg-rose-500/10 border border-rose-500/30 text-rose-300 px-2 py-1 rounded-lg">{s}</span>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-emerald-400 font-semibold">Perfect technical skills match!</div>
                      )}
                    </div>
                  </div>

                  {/* Plain Language Explanation & Neural Attention */}
                  <div className="glass-panel p-6 rounded-xl flex flex-col gap-4">
                    <div>
                      <h3 className="text-sm font-bold text-white mb-2">Natural Language Fit Assessment</h3>
                      <p className="text-slate-300 text-sm leading-relaxed">{scoreData.score_breakdown.explanation_text}</p>
                    </div>

                    <div className="border-t border-slate-800 pt-4">
                      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Neural Attention Highlight</h4>
                      <div className="flex flex-wrap gap-2 items-center">
                        <span className="text-[10px] text-slate-500">Top matching vocab tokens:</span>
                        {scoreData.score_breakdown.attention_top_tokens.map(token => (
                          <span key={token} className="text-xs font-mono bg-sky-950/60 border border-sky-800/50 text-sky-400 px-2 py-0.5 rounded">{token}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab Content 2: Gap Report */}
              {activeTab === 'feedback' && (
                <div className="glass-panel p-6 rounded-xl flex flex-col gap-4 animate-fade-in">
                  <h2 className="text-lg font-bold text-white border-b border-slate-700 pb-2">Retrieval-Grounded Feedback Report</h2>
                  
                  {feedbackData ? (
                    <div className="text-xs md:text-sm text-slate-300 font-mono whitespace-pre-wrap leading-relaxed max-h-[600px] overflow-y-auto bg-slate-950/80 p-6 border border-slate-800 rounded-lg">
                      {feedbackData.feedback_report}
                    </div>
                  ) : (
                    <div className="text-slate-500 text-center py-10">No feedback report available.</div>
                  )}
                </div>
              )}

              {/* Tab Content 3: Career Coach Chat */}
              {activeTab === 'chat' && (
                <div className="glass-panel rounded-xl flex flex-col h-[650px] overflow-hidden border border-slate-800 animate-fade-in">
                  {/* Chat Header */}
                  <div className="bg-slate-950/80 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">AI Career Coach Session</h3>
                      <p className="text-[10px] text-slate-400">Contextual routing backed by LangGraph matching tools</p>
                    </div>
                    <span className="text-[10px] bg-sky-500/20 text-sky-400 font-mono px-2 py-0.5 rounded">{sessionId}</span>
                  </div>

                  {/* Messages list */}
                  <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 bg-slate-900/60">
                    {chatHistory.map((msg, i) => (
                      <div key={i} className={`flex flex-col max-w-[85%] ${msg.sender === 'user' ? 'self-end items-end' : 'self-start items-start'}`}>
                        <div className={`rounded-xl px-4 py-2.5 text-sm ${msg.sender === 'user' ? 'bg-sky-600 text-white rounded-br-none' : 'bg-slate-800/80 text-slate-200 border border-slate-700 rounded-bl-none'}`}>
                          {msg.text}
                        </div>
                        {msg.tools && msg.tools.length > 0 && (
                          <div className="mt-1 flex items-center gap-1.5 text-[9px] text-slate-500 font-mono">
                            <span className="bg-slate-800 text-slate-400 rounded px-1.5 py-0.5 border border-slate-700">
                              Tool called: {msg.tools[0].name}
                            </span>
                          </div>
                        )}
                      </div>
                    ))}
                    {isChatLoading && (
                      <div className="self-start flex flex-col items-start max-w-[85%]">
                        <div className="rounded-xl px-4 py-2.5 text-sm bg-slate-800/80 border border-slate-700 rounded-bl-none flex items-center gap-2">
                          <span className="flex gap-1">
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                          </span>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef}></div>
                  </div>

                  {/* Quick prompts */}
                  <div className="px-4 py-2 bg-slate-950/20 border-t border-slate-800/50 flex flex-wrap gap-2">
                    <button 
                      onClick={() => sendQuickPrompt("Can you break down my fit score?")} 
                      className="text-[10px] bg-slate-800 hover:bg-slate-700 transition-colors text-slate-300 px-2 py-1 rounded border border-slate-700"
                    >
                      📊 Fit Breakdown
                    </button>
                    <button 
                      onClick={() => sendQuickPrompt("Which required skills am I missing?")} 
                      className="text-[10px] bg-slate-800 hover:bg-slate-700 transition-colors text-slate-300 px-2 py-1 rounded border border-slate-700"
                    >
                      ⚠️ Missing Skills
                    </button>
                    <button 
                      onClick={() => sendQuickPrompt("Can you show me similar benchmark profiles?")} 
                      className="text-[10px] bg-slate-800 hover:bg-slate-700 transition-colors text-slate-300 px-2 py-1 rounded border border-slate-700"
                    >
                      👥 Peer Benchmarks
                    </button>
                  </div>

                  {/* Input form */}
                  <form id="chat-form" onSubmit={handleSendMessage} className="bg-slate-950 p-4 border-t border-slate-800 flex gap-2">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Ask a follow-up question..."
                      className="flex-1 bg-slate-900 border border-slate-750 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-sky-500"
                    />
                    <button
                      type="submit"
                      disabled={isChatLoading}
                      className="bg-sky-600 hover:bg-sky-500 text-white font-bold px-4 rounded-lg text-sm transition-colors"
                    >
                      Send
                    </button>
                  </form>
                </div>
              )}
            </>
          ) : (
            <div className="glass-panel p-10 rounded-xl flex flex-col items-center justify-center text-center gap-4 text-slate-500 py-32 border border-dashed border-slate-800">
              <svg className="w-12 h-12 text-slate-700 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
              </svg>
              <div>
                <h3 className="text-slate-400 font-bold mb-1">No Profile Selected</h3>
                <p className="text-xs">Select a category, load a resume file or text, choose a target JD, and click screen to view evaluations.</p>
              </div>
            </div>
          )}
        </section>

      </main>

      {/* Footer */}
      <footer className="bg-slate-950 border-t border-slate-850 px-6 py-4 text-center text-xs text-slate-500">
        ResumeIQ Evaluation Suite — Powered by LightGBM, DistilBERT, FAISS & LangGraph checkpoints.
      </footer>
    </div>
  );
}

export default App;
