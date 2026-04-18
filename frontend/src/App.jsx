import { useEffect, useMemo, useRef, useState } from "react";

const REVIEW_TEMPLATES = [
  {
    name: "spiderman.png",
    label: "Hero Action",
    meta: "Full body",
    preview: "./review-assets/spiderman.png",
  },
  {
    name: "superman adult.png",
    label: "Executive Suit",
    meta: "Business",
    preview: "./review-assets/superman%20adult.png",
  },
  {
    name: "superman.png",
    label: "Power Stance",
    meta: "Studio hero",
    preview: "./review-assets/superman.png",
  },
];

const REVIEW_RECENT_FACES = [
  { id: "face-01", label: "Face 01", preview: "./review-assets/faces/face-01.png" },
  { id: "face-02", label: "Face 02", preview: "./review-assets/faces/face-02.png" },
  { id: "face-03", label: "Face 03", preview: "./review-assets/faces/face-03.png" },
  { id: "face-04", label: "Face 04", preview: "./review-assets/faces/face-04.png" },
];

const REVIEW_RESULT_IMAGES = {
  final_result: "./review-assets/final_00001_.png",
  generated_head: "./review-assets/generated_head_00001_.png",
  target_head_mask: "./review-assets/target_head_mask_00001_.png",
};

function createReviewApi() {
  let jobs = [];
  let authenticated = false;

  return {
    async session() {
      return {
        enabled: true,
        authenticated,
        username: authenticated ? "reviewer" : null,
      };
    },
    async login({ username, password }) {
      if (username !== "reviewer" || password !== "review-pass") {
        throw new Error("Invalid credentials.");
      }
      authenticated = true;
      return { enabled: true, authenticated: true, username };
    },
    async logout() {
      authenticated = false;
      return { enabled: true, authenticated: false, username: null };
    },
    async health() {
      return {
        shared_env_path: "/Users/blue/dev/comfyui/faceswap/.env",
        shared_root: "/Users/blue/dev/comfyui/faceswap",
        stable_workflow_api_path:
          "/Users/blue/dev/comfyui/faceswap/workflows/stable/visual_prompt_hybrid_v1_api.json",
        stable_workflow_ui_path:
          "/Users/blue/dev/comfyui/faceswap/workflows/stable/visual_prompt_hybrid_v1_ui.json",
        mode: "review",
        auth_enabled: true,
        database_ready: true,
        storage_ready: true,
        missing: [],
      };
    },
    async templates() {
      return REVIEW_TEMPLATES;
    },
    async jobs() {
      return jobs;
    },
    async createJob({ subjectFile, targetTemplate, preserveLighting, refinementLevel }) {
      const now = new Date().toISOString();
      const id = `review-${Date.now()}`;
      const subjectUrl = URL.createObjectURL(subjectFile);
      const template = REVIEW_TEMPLATES.find((item) => item.name === targetTemplate);
      const templatePreview = template?.preview || "";
      const job = {
        id,
        status: "queued",
        subject_original_name: subjectFile.name,
        subject_r2_url: subjectUrl,
        target_template: targetTemplate,
        workflow_name: "visual_prompt_hybrid_v1_api.json (review)",
        comfy_prompt_id: `review-${id.slice(-6)}`,
        error_message: null,
        created_at: now,
        updated_at: now,
        preserve_lighting: preserveLighting,
        refinement_level: refinementLevel,
        artifacts: [],
      };
      jobs = [job, ...jobs];

      window.setTimeout(() => {
        job.status = "running";
        job.updated_at = new Date().toISOString();
        jobs = [...jobs];
      }, 500);

      window.setTimeout(() => {
        job.status = "completed";
        job.updated_at = new Date().toISOString();
        job.artifacts = [
          artifact("subject_input", subjectUrl),
          artifact("target_template", templatePreview),
          artifact("generated_head", REVIEW_RESULT_IMAGES.generated_head),
          artifact("target_head_mask", REVIEW_RESULT_IMAGES.target_head_mask),
          artifact("final_result", REVIEW_RESULT_IMAGES.final_result),
        ];
        jobs = [...jobs];
      }, 1600);

      return job;
    },
  };
}

function artifact(name, displayUrl) {
  return {
    name,
    display_url: displayUrl,
    proxy_url: displayUrl,
    filename: displayUrl.split("/").pop(),
  };
}

function createHttpApi() {
  return {
    async session() {
      return fetchJson("/api/auth/session");
    },
    async login(payload) {
      return fetchJson("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    async logout() {
      return fetchJson("/api/auth/logout", { method: "POST" });
    },
    async health() {
      return fetchJson("/api/health");
    },
    async templates() {
      return fetchJson("/api/templates");
    },
    async jobs() {
      return fetchJson("/api/jobs");
    },
    async createJob({ subjectFile, targetTemplate }) {
      const formData = new FormData();
      formData.append("subject", subjectFile);
      formData.append("target_template", targetTemplate);
      return fetchJson("/api/jobs", { method: "POST", body: formData });
    },
  };
}

function detectMode() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("review") === "1" || window.location.protocol === "file:") {
    return "review";
  }
  return "http";
}

export default function App() {
  const mode = useMemo(() => detectMode(), []);
  const apiRef = useRef(mode === "review" ? createReviewApi() : createHttpApi());

  const [session, setSession] = useState(null);
  const [health, setHealth] = useState(null);
  const [templates, setTemplates] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [subjectFile, setSubjectFile] = useState(null);
  const [subjectPreviewUrl, setSubjectPreviewUrl] = useState("");
  const [selectedJobId, setSelectedJobId] = useState("");
  const [selectedFaceId, setSelectedFaceId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [loginUsername, setLoginUsername] = useState(mode === "review" ? "reviewer" : "");
  const [loginPassword, setLoginPassword] = useState(mode === "review" ? "review-pass" : "");
  const [loginError, setLoginError] = useState("");
  const [preserveLighting, setPreserveLighting] = useState(true);
  const [refinementLevel, setRefinementLevel] = useState("Balanced");

  const selectedJob = jobs.find((job) => job.id === selectedJobId) || jobs[0] || null;
  const selectedTemplateItem = templates.find((item) => item.name === selectedTemplate) || null;
  const previewUrl =
    artifactDisplay(selectedJob, "final_result") ||
    selectedTemplateItem?.preview ||
    (selectedTemplate ? `/api/templates/${encodeURIComponent(selectedTemplate)}/preview` : "");
  const previewTitle = selectedJob?.status === "completed" ? "Final Preview" : "Draft Preview";
  const queueState = selectedJob ? statusLabel(selectedJob.status) : "Idle";
  const runId = selectedJob ? selectedJob.id.slice(-6) : "Pending";

  const recentFaces = useMemo(() => {
    const uploadedFace =
      subjectPreviewUrl && subjectFile
        ? [{ id: "current-upload", label: "Current", preview: subjectPreviewUrl, fromUpload: true }]
        : [];
    return [...uploadedFace, ...REVIEW_RECENT_FACES].slice(0, 4);
  }, [subjectFile, subjectPreviewUrl]);

  const pastSwaps = useMemo(() => {
    return jobs.slice(0, 4).map((job) => ({
      id: job.id,
      preview: artifactDisplay(job, "final_result") || artifactDisplay(job, "target_template") || "",
      status: statusLabel(job.status),
      time: formatShortTime(job.updated_at || job.created_at),
    }));
  }, [jobs]);

  useEffect(() => {
    let active = true;

    async function loadAuthedData() {
      const [nextHealth, nextTemplatesRaw, nextJobs] = await Promise.all([
        apiRef.current.health(),
        apiRef.current.templates(),
        apiRef.current.jobs(),
      ]);
      if (!active) {
        return;
      }
      const nextTemplates = normalizeTemplates(nextTemplatesRaw);
      setHealth(nextHealth);
      setTemplates(nextTemplates);
      setSelectedTemplate((current) => current || nextTemplates[0]?.name || null);
      setJobs(nextJobs);
      setSelectedJobId((current) => current || nextJobs[0]?.id || "");
      setStatusText(mode === "review" ? "Review mode ready." : "");
    }

    async function boot() {
      try {
        const nextSession = await apiRef.current.session();
        if (!active) {
          return;
        }
        setSession(nextSession);
        if (nextSession.authenticated) {
          await loadAuthedData();
        }
      } catch (error) {
        if (active) {
          setStatusText(error.message);
        }
      }
    }

    boot();
    const timer = window.setInterval(async () => {
      try {
        const currentSession = await apiRef.current.session();
        if (!active) {
          return;
        }
        setSession(currentSession);
        if (!currentSession.authenticated) {
          return;
        }
        const nextJobs = await apiRef.current.jobs();
        if (!active) {
          return;
        }
        setJobs(nextJobs);
        setSelectedJobId((current) => current || nextJobs[0]?.id || "");
      } catch (_error) {
      }
    }, 1200);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [mode]);

  async function handleLogin(event) {
    event.preventDefault();
    setLoginError("");
    try {
      const nextSession = await apiRef.current.login({
        username: loginUsername,
        password: loginPassword,
      });
      setSession(nextSession);
      const [nextHealth, nextTemplatesRaw, nextJobs] = await Promise.all([
        apiRef.current.health(),
        apiRef.current.templates(),
        apiRef.current.jobs(),
      ]);
      const nextTemplates = normalizeTemplates(nextTemplatesRaw);
      setHealth(nextHealth);
      setTemplates(nextTemplates);
      setSelectedTemplate(nextTemplates[0]?.name || null);
      setJobs(nextJobs);
      setSelectedJobId(nextJobs[0]?.id || "");
      setStatusText("Signed in.");
    } catch (error) {
      setLoginError(error.message);
    }
  }

  async function handleLogout() {
    const nextSession = await apiRef.current.logout();
    setSession(nextSession);
    setHealth(null);
    setTemplates([]);
    setJobs([]);
    setSelectedJobId("");
    setStatusText("");
    setSubjectFile(null);
    setSubjectPreviewUrl("");
    setSelectedFaceId("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!subjectFile || !selectedTemplate) {
      setStatusText("Upload a face and choose a template.");
      return;
    }

    setSubmitting(true);
    setStatusText(mode === "review" ? "Creating review run..." : "Queueing full body swap...");
    try {
      const job = await apiRef.current.createJob({
        subjectFile,
        targetTemplate: selectedTemplate,
        preserveLighting,
        refinementLevel,
      });
      setSelectedJobId(job.id);
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setStatusText(mode === "review" ? "Run queued in review mode." : "Run queued.");
    } catch (error) {
      setStatusText(error.message);
    } finally {
      setSubmitting(false);
    }
  }

  function onSubjectChange(file, faceId = file ? "current-upload" : "") {
    setSubjectFile(file);
    setSelectedFaceId(faceId);
    if (!file) {
      setSubjectPreviewUrl("");
      return;
    }
    setSubjectPreviewUrl(URL.createObjectURL(file));
  }

  async function selectRecentFace(face) {
    try {
      const response = await fetch(face.preview);
      const blob = await response.blob();
      const extension = blob.type.includes("png") ? "png" : "jpg";
      const file = new File([blob], `${face.id}.${extension}`, { type: blob.type || "image/png" });
      onSubjectChange(file, face.id);
    } catch (_error) {
      setStatusText("Could not load that face.");
    }
  }

  if (session && !session.authenticated) {
    return (
      <div className="studio-shell studio-shell--auth">
        <div className="studio-glow studio-glow--left" />
        <div className="studio-glow studio-glow--right" />
        <section className="auth-card">
          <div className="brand-block">
            <p className="brand-kicker">Face Swap Studio</p>
            <h1>Secure sign-in</h1>
            <p>Access studio previews from desktop and mobile.</p>
          </div>
          <form className="auth-form" onSubmit={handleLogin}>
            <label className="field-stack">
              <span>Username</span>
              <input aria-label="Username" value={loginUsername} onChange={(event) => setLoginUsername(event.target.value)} />
            </label>
            <label className="field-stack">
              <span>Password</span>
              <input
                aria-label="Password"
                type="password"
                value={loginPassword}
                onChange={(event) => setLoginPassword(event.target.value)}
              />
            </label>
            <button className="primary-cta" type="submit">
              Sign In
            </button>
            <p className="field-note">{loginError || (mode === "review" ? "Review credentials: reviewer / review-pass" : "")}</p>
          </form>
        </section>
      </div>
    );
  }

  return (
    <div className="studio-shell">
      <div className="studio-glow studio-glow--left" />
      <div className="studio-glow studio-glow--right" />

      <header className="studio-header">
        <div className="header-spacer" />
        <div className="brand-lockup">
          <p className="brand-kicker">Studio</p>
          <h1>Face Swap Studio</h1>
        </div>
        <div className="header-actions">
          <span className="session-chip">{session?.username || "session"}</span>
          <button className="ghost-pill" onClick={handleLogout} type="button">
            Log Out
          </button>
        </div>
      </header>

      <form className="studio-grid" onSubmit={handleSubmit}>
        <section className="studio-panel zone-input">
          <div className="panel-heading">
            <p className="panel-kicker">Zone 1</p>
            <h2>Face Input</h2>
          </div>

          <label className="drop-zone">
            <input
              type="file"
              accept="image/*"
              onChange={(event) => onSubjectChange(event.target.files?.[0] || null, event.target.files?.[0] ? "current-upload" : "")}
            />
            <div className="drop-zone__icon">↑</div>
            <strong>Upload Portrait</strong>
            <span>Single face image</span>
          </label>

          <div className="panel-subsection">
            <div className="section-row">
              <strong>Recent Faces</strong>
              <span>{recentFaces.length} ready</span>
            </div>
            <div className="recent-face-grid">
              {recentFaces.map((face) => (
                <button
                  className={`recent-face ${selectedFaceId === face.id ? "recent-face--selected" : ""}`}
                  key={face.id}
                  onClick={() => selectRecentFace(face)}
                  type="button"
                >
                  <img alt={face.label} src={face.preview} />
                </button>
              ))}
            </div>
          </div>

          <div className="face-preview-card">
            {subjectPreviewUrl ? (
              <>
                <img alt="Selected face" src={subjectPreviewUrl} />
                <div className="face-preview-meta">
                  <strong>Active Face</strong>
                  <span>{subjectFile?.name || "Selected portrait"}</span>
                </div>
              </>
            ) : (
              <span>Active face preview</span>
            )}
          </div>
        </section>

        <section className="studio-panel zone-templates">
          <div className="panel-heading">
            <p className="panel-kicker">Zone 2</p>
            <h2>Template Library</h2>
          </div>

          <div className="template-library">
            {templates.map((template) => (
              <button
                className={`template-card ${selectedTemplate === template.name ? "template-card--selected" : ""}`}
                key={template.name}
                onClick={() => setSelectedTemplate(template.name)}
                type="button"
              >
                <img alt={template.label} src={template.preview || `/api/templates/${encodeURIComponent(template.name)}/preview`} />
                <div className="template-card__meta">
                  <strong>{template.label}</strong>
                  <span>{template.meta}</span>
                </div>
              </button>
            ))}
          </div>

          <div className="mobile-queue-bar">
            <button className="primary-cta primary-cta--full" disabled={submitting} type="submit">
              {submitting ? "Queueing..." : "Queue Full Body Swap"}
            </button>
          </div>
        </section>

        <section className="studio-panel zone-output">
          <div className="panel-heading">
            <p className="panel-kicker">Zone 3</p>
            <h2>Workspace & Output</h2>
          </div>

          <div className="workspace-preview">
            {previewUrl ? <img alt="Workspace preview" src={previewUrl} /> : <span>Preview</span>}
          </div>
          <div className="preview-caption">
            <strong>{previewTitle}</strong>
            <span>{selectedTemplateItem?.label || "Choose a template"}</span>
          </div>

          <div className="run-facts">
            <div className="run-fact">
              <span>Run ID</span>
              <strong>{runId}</strong>
            </div>
            <div className="run-fact">
              <span>Status</span>
              <strong>{queueState}</strong>
            </div>
            <div className="run-fact">
              <span>Workflow</span>
              <strong>Stable v1.2</strong>
            </div>
          </div>

          <div className="config-stack">
            <div className="toggle-row">
              <div>
                <strong>Preserve Lighting</strong>
                <span>Keep original scene tone</span>
              </div>
              <button
                aria-pressed={preserveLighting}
                className={`toggle-switch ${preserveLighting ? "toggle-switch--active" : ""}`}
                onClick={() => setPreserveLighting((current) => !current)}
                type="button"
              >
                <span />
              </button>
            </div>

            <div className="refinement-block">
              <div className="section-row">
                <strong>Refinement Level</strong>
                <span>{refinementLevel}</span>
              </div>
              <div className="segmented-control">
                {["Clean", "Balanced", "High"].map((level) => (
                  <button
                    className={`segment ${refinementLevel === level ? "segment--active" : ""}`}
                    key={level}
                    onClick={() => setRefinementLevel(level)}
                    type="button"
                  >
                    {level}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button className="primary-cta primary-cta--desktop" disabled={submitting} type="submit">
            {submitting ? "Queueing..." : "Queue Full Body Swap"}
          </button>

          <div className="panel-subsection">
            <div className="section-row">
              <strong>Past Swaps</strong>
              <span>{pastSwaps.length} recent</span>
            </div>
            <div className="past-swap-list">
              {pastSwaps.length ? (
                pastSwaps.map((item) => (
                  <article className="past-swap-item" key={item.id}>
                    <div className="past-swap-thumb">
                      {item.preview ? <img alt={item.status} src={item.preview} /> : <span /> }
                    </div>
                    <div className="past-swap-copy">
                      <strong>{item.status}</strong>
                      <span>{item.time}</span>
                    </div>
                  </article>
                ))
              ) : (
                <div className="past-swap-empty">No past swaps yet</div>
              )}
            </div>
          </div>
        </section>
      </form>

      <footer className="studio-footer">
        <span>{statusText || "Ready"}</span>
        <span>{health?.mode === "review" ? "Review mode" : "Live mode"}</span>
      </footer>
    </div>
  );
}

function fetchJson(url, options) {
  return fetch(url, options).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
    }
    return data;
  });
}

function normalizeTemplates(items) {
  return items.map((item) => ({
    ...item,
    label: item.label || prettifyName(item.name),
    meta: item.meta || "Template",
  }));
}

function prettifyName(value) {
  return value.replace(/\.[a-z0-9]+$/i, "").replace(/[-_]+/g, " ");
}

function artifactDisplay(job, name) {
  return job?.artifacts?.find((item) => item.name === name)?.display_url || "";
}

function statusLabel(status) {
  if (status === "completed") return "Complete";
  if (status === "running") return "Running";
  if (status === "queueing") return "Queueing";
  if (status === "queued") return "Queued";
  if (status === "failed") return "Failed";
  return "Pending";
}

function formatShortTime(value) {
  if (!value) {
    return "Now";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
