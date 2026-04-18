const state = {
  selectedTemplate: null,
  health: null,
  jobs: [],
};

const healthEl = document.getElementById("health");
const templatesEl = document.getElementById("templates");
const jobsEl = document.getElementById("jobs");
const formEl = document.getElementById("job-form");
const subjectInput = document.getElementById("subject");
const subjectPreviewEl = document.getElementById("subject-preview");
const formStatusEl = document.getElementById("form-status");
const submitButton = document.getElementById("submit-button");

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  }
  return data;
}

async function loadHealth() {
  state.health = await fetchJson("/api/health");
  renderHealth();
}

async function loadTemplates() {
  const templates = await fetchJson("/api/templates");
  if (!state.selectedTemplate && templates.length) {
    state.selectedTemplate = templates[0].name;
  }
  renderTemplates(templates);
}

async function loadJobs() {
  state.jobs = await fetchJson("/api/jobs");
  renderJobs();
}

function renderHealth() {
  const missing = state.health?.missing || [];
  const cards = [
    {
      label: "Shared Env",
      value: state.health?.shared_env_path || "Unavailable",
    },
    {
      label: "Workflow",
      value: state.health?.stable_workflow_api_path?.split("/").pop() || "Unavailable",
    },
    {
      label: "Neon",
      value: state.health?.database_ready ? "Ready" : "Missing",
    },
    {
      label: "R2",
      value: state.health?.storage_ready ? "Ready" : "Missing",
    },
  ];
  if (missing.length) {
    cards.push({
      label: "Blocked By",
      value: missing.join(", "),
    });
  }
  healthEl.innerHTML = cards
    .map(
      (card) => `
        <article class="health-card">
          <span>${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
        </article>
      `,
    )
    .join("");
}

function renderTemplates(templates) {
  templatesEl.innerHTML = templates
    .map(
      (template) => `
        <button class="template-card ${template.name === state.selectedTemplate ? "selected" : ""}"
          type="button"
          data-template="${escapeHtml(template.name)}">
          <img src="/api/templates/${encodeURIComponent(template.name)}/preview" alt="${escapeHtml(template.label)}" />
          <footer>
            <strong>${escapeHtml(template.label)}</strong>
            <span>${escapeHtml(template.name)}</span>
          </footer>
        </button>
      `,
    )
    .join("");

  templatesEl.querySelectorAll("[data-template]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTemplate = button.dataset.template;
      renderTemplates(templates);
    });
  });
}

function renderJobs() {
  if (!state.jobs.length) {
    jobsEl.innerHTML = `<div class="empty-state">No jobs queued yet. Submit a subject image to test the stable workflow path.</div>`;
    return;
  }

  jobsEl.innerHTML = state.jobs
    .map((job) => {
      const artifacts = (job.artifacts || [])
        .map(
          (artifact) => `
            <a class="artifact" href="${escapeHtml(artifact.display_url)}" target="_blank" rel="noreferrer">
              <img src="${escapeHtml(artifact.display_url)}" alt="${escapeHtml(artifact.name)}" />
              <span>${escapeHtml(artifact.name)}</span>
            </a>
          `,
        )
        .join("");

      return `
        <article class="job-card">
          <header>
            <div>
              <h3>${escapeHtml(job.subject_original_name)}</h3>
              <div class="job-meta">
                <span>Target: ${escapeHtml(job.target_template)}</span>
                <span>Workflow: ${escapeHtml(job.workflow_name)}</span>
                ${job.comfy_prompt_id ? `<span>Prompt: ${escapeHtml(job.comfy_prompt_id)}</span>` : ""}
              </div>
            </div>
            <span class="badge ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span>
          </header>
          ${
            job.error_message
              ? `<p class="form-status">${escapeHtml(job.error_message)}</p>`
              : ""
          }
          ${artifacts ? `<div class="artifact-grid">${artifacts}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

subjectInput.addEventListener("change", () => {
  const [file] = subjectInput.files || [];
  if (!file) {
    subjectPreviewEl.className = "subject-preview empty";
    subjectPreviewEl.innerHTML = "<span>Upload preview</span>";
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  subjectPreviewEl.className = "subject-preview";
  subjectPreviewEl.innerHTML = `<img src="${objectUrl}" alt="Subject preview" />`;
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = subjectInput.files || [];
  if (!file || !state.selectedTemplate) {
    formStatusEl.textContent = "Choose a subject image and a target template first.";
    return;
  }

  submitButton.disabled = true;
  formStatusEl.textContent = "Submitting job to the stable workflow queue...";
  const formData = new FormData();
  formData.append("subject", file);
  formData.append("target_template", state.selectedTemplate);

  try {
    await fetchJson("/api/jobs", {
      method: "POST",
      body: formData,
    });
    formEl.reset();
    subjectPreviewEl.className = "subject-preview empty";
    subjectPreviewEl.innerHTML = "<span>Upload preview</span>";
    formStatusEl.textContent = "Job queued. Polling for progress.";
    await loadJobs();
  } catch (error) {
    formStatusEl.textContent = error.message;
  } finally {
    submitButton.disabled = false;
  }
});

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function boot() {
  try {
    await Promise.all([loadHealth(), loadTemplates(), loadJobs()]);
  } catch (error) {
    formStatusEl.textContent = error.message;
  }
  setInterval(() => {
    loadJobs().catch(() => {});
  }, 5000);
}

boot();
