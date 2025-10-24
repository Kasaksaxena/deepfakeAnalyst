// script.js - Deepfake Content Analyzer
// Beginner-friendly JavaScript that uploads a video file to a local backend
// and displays analysis results (label, ethical_score, deepfake_probability, transcript).

// Get references to DOM elements we'll interact with
const fileInput = document.getElementById('video-upload');
const analyzeBtn = document.getElementById('analyze-btn');
const spinner = document.getElementById('spinner');
const resultsContainer = document.getElementById('results');

// The backend endpoint (running locally as you mentioned)
const ANALYZE_URL = 'http://127.0.0.1:8000/api/v1/analyze';

// Utility: toggle spinner visibility and ARIA attributes
function showLoading(show = true) {
  if (show) {
    spinner.classList.add('active');
    spinner.setAttribute('aria-hidden', 'false');
    analyzeBtn.disabled = true;
  } else {
    spinner.classList.remove('active');
    spinner.setAttribute('aria-hidden', 'true');
    analyzeBtn.disabled = false;
  }
}

// Utility: show a friendly message inside results area
function setResultsMessage(message) {
  // Always show default or error message on result-front, never on resultsContainer!
  const frontDiv = document.getElementById('result-front');
  const inner = document.getElementById('result-card-inner');
  if(frontDiv && inner) {
    frontDiv.innerHTML = `<div class="results-empty">${message}</div>`;
    document.getElementById('result-back').innerHTML = ''; // Clear back
    inner.classList.remove('flipped');
  }
}


// Utility: map probability to a badge class and label
function probabilityBadge(prob) {
  // prob is expected between 0 and 1 (or 0..100). We'll normalize if necessary.
  if (prob > 1) prob = prob / 100; // handle percent values like 85
  const p = Number(prob);
  if (isNaN(p)) return { text: 'Unknown', cls: '' };
  if (p >= 0.75) return { text: `${Math.round(p * 100)}% — Likely Deepfake`, cls: 'badge fake' };
  if (p >= 0.4) return { text: `${Math.round(p * 100)}% — Suspicious`, cls: 'badge suspect' };
  return { text: `${Math.round(p * 100)}% — Likely Real`, cls: 'badge safe' };
}

// Convert a value (maybe null/undefined) to a safe string for display
function safeString(v, fallback = '—') {
  if (v === null || v === undefined) return fallback;
  if (typeof v === 'number') return String(v);
  return String(v);
}

function renderResults(data) {
  // --- Calculate tags + summary
  const prob = Number(data.deepfake_probability ?? data.probability ?? 0);
  const isFake = prob >= 0.7;
  const probPercent = Math.round(prob * 100);

  // Front (user summary)
  document.getElementById('result-front').innerHTML = `
    <div class="result-item">
      <div style="font-size:18px;margin-bottom:2px;font-weight:600;">
        ${
          isFake
            ? 'Fake Video 🛑'
            : 'Real Video ✅'
        }
      </div>
      <div class="badge" style="font-size:15px;
        background:${isFake ? '#c71c3c33' : '#19875433'};
        color:${isFake ? '#ffd1d9' : '#b2ffdf'};
        margin-bottom:8px;display:inline-block;">
        ${isFake ? 'Likely Deepfake' : 'Safe and Ethical'}
      </div>
      <div style="margin-top:18px">
        <span class="label" style="font-size:15px;">Ethical score</span><br>
        <span class="value" style="font-size:17px">${data.ethical_score ?? 'N/A'}</span>
      </div>
      <div style="margin-top:18px">
        <span class="label" style="font-size:15px;">Deepfake score</span><br>
        <span class="value" style="font-size:17px">${probPercent}%</span>
      </div>
    </div>
    <div class="result-flip-tip">Click card to view all backend data</div>
  `;

  // --- Back (raw backend result, nicely formatted)
  document.getElementById('result-back').innerHTML = `
    <h3 style="font-size:15px;font-weight:400;margin:0 0 8px;">Raw backend result</h3>
    <pre style="overflow-x:auto;border-radius:6px;background:#23273c99; padding: 16px;">${JSON.stringify(data, null, 2)}</pre>
    <div class="result-flip-tip">Click card to flip back</div>
  `;

  // Reset to non-flipped if last state was flipped!
  document.getElementById('result-card-inner').classList.remove('flipped');
}

// --- Card flip handler (add end of JS file) ---
document.getElementById('results').addEventListener('click', function(e) {
  // Only flip if result is present (to avoid flipping on empty state)
  const inner = document.getElementById('result-card-inner');
  if(inner && document.getElementById('result-front').innerHTML.trim() !== "") {
    inner.classList.toggle('flipped');
  }
});


// Main function invoked when user clicks "Start analyzing"
async function analyzeFile() {
  // Make sure a file is selected
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setResultsMessage('Please choose a video file to analyze.');
    return;
  }

  // Prepare form data. Many backends expect the field name 'file' or 'video'.
  // If your backend expects a different key, change 'file' below accordingly.
  const formData = new FormData();
  formData.append('video', file);

  // Show loading state
  showLoading(true);
  setResultsMessage('Loading — analyzing the uploaded video. This may take a little while...');

  try {
    // Send the request using fetch. We do not set the Content-Type header;
    // the browser will set the correct multipart/form-data boundary when sending FormData.
    const resp = await fetch(ANALYZE_URL, {
      method: 'POST',
      body: formData,
      // If your backend requires CORS credentials or a token, set them here.
      // credentials: 'include',
      // headers: { 'Authorization': 'Bearer <token>' }
    });

    if (!resp.ok) {
      // Try to read any error message from the server
      let errText = '';
      try { errText = await resp.text(); } catch (e) { errText = resp.statusText; }
      throw new Error(`Server responded with ${resp.status}: ${errText}`);
    }

    // Parse JSON response. Expected shape (example):
    // { label: 'deepfake', ethical_score: 0.32, deepfake_probability: 0.87, transcript: 'Hello world' }
    const json = await resp.json();

    // Render the result fields into the results container
    renderResults(json);
  } catch (err) {
  // Show the error to the user in a friendly way
  console.error('Analysis error:', err);
  setResultsMessage('Error: ' + safeString(err.message, 'An unexpected error occurred.'));
}
 finally {
    // Always hide loading state and re-enable button
    showLoading(false);
  }
}

// Wire up the analyze button click
analyzeBtn.addEventListener('click', (e) => {
  e.preventDefault();
  analyzeFile();
});

// Optional: allow pressing Enter when file input has focus
fileInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') analyzeFile();
});

// Initialize helpful message on load
setResultsMessage('No results yet. Upload a video and click Start analyzing.');

// --- File selection UI helpers ---
// Assumes existing constants: fileInput, analyzeBtn and a .file-prompt element inside the label.
const filePrompt = document.querySelector('.file-prompt');

// Show filename, enable analyze button, and log the File object for debugging
function onFileSelected() {
  const file = fileInput.files && fileInput.files[0];

  if (!file) {
    // no file selected (user cancelled)
    filePrompt.textContent = 'Choose a video or drag & drop here';
    analyzeBtn.disabled = true;
    console.log('No file selected.');
    return;
  }

  // Show friendly filename (not full path) and enable analyze button
  const shortName = file.name.length > 48
    ? file.name.slice(0, 24) + '…' + file.name.slice(-18)
    : file.name;
  filePrompt.textContent = `Selected: ${shortName}`;
  analyzeBtn.disabled = false;

  // Helpful debug info to inspect in Console
  console.log('File selected for analysis:', {
    name: file.name,
    size: file.size,
    type: file.type,
    lastModified: file.lastModified
  });
}

// Attach the change handler to the input
fileInput.addEventListener('change', onFileSelected);

// Optional: initialize button disabled until a file is picked
analyzeBtn.disabled = !(fileInput.files && fileInput.files.length > 0);
