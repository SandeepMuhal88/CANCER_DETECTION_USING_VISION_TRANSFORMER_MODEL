"""
app.py  –  Cancer Detection Web App
Chest X-Ray (Normal vs Pneumonia) using ResNet-18
Run:  python app.py
"""

import os
import io
import base64
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "models" / "xray_model.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE    = 224

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    if MODEL_PATH.exists():
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  No model found at {MODEL_PATH}. "
              f"Run Backend/train.py first. Using random weights for now.")
    m.eval()
    return m.to(device)

model = load_model()

# ── Transform (same as training) ──────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image):
    tensor = infer_transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0].cpu().numpy()
    pred  = int(np.argmax(probs))
    return {
        "prediction": CLASS_NAMES[pred],
        "confidence": float(probs[pred]) * 100,
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASS_NAMES, probs)
        },
    }

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Cancer Detection – Chest X-Ray AI</title>
<meta name="description" content="AI-powered chest X-ray analysis for Pneumonia vs Normal detection using ResNet-18 Vision model."/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #1e2d45;
    --accent:   #3b82f6;
    --accent2:  #06b6d4;
    --green:    #10b981;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --radius:   16px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* ── Header ── */
  header {
    width: 100%;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
  }
  header h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.3px; }
  header span { font-size: 0.8rem; color: var(--muted); font-weight: 400; margin-left: 6px; }
  .badge {
    margin-left: auto;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.3);
    color: var(--accent);
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  /* ── Main ── */
  main {
    width: 100%;
    max-width: 960px;
    padding: 48px 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 32px;
  }

  .hero-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--text) 30%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-sub {
    text-align: center;
    color: var(--muted);
    font-size: 0.95rem;
    max-width: 520px;
    line-height: 1.65;
  }

  /* ── Upload card ── */
  .upload-card {
    width: 100%;
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px 32px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    cursor: pointer;
    transition: border-color 0.25s, background 0.25s;
    position: relative;
  }
  .upload-card:hover, .upload-card.drag-over {
    border-color: var(--accent);
    background: rgba(59,130,246,0.05);
  }
  .upload-icon { font-size: 3rem; }
  .upload-card h2 { font-size: 1.1rem; font-weight: 600; }
  .upload-card p  { color: var(--muted); font-size: 0.85rem; text-align: center; line-height: 1.5; }
  #fileInput { display: none; }

  .btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 28px;
    border: none; border-radius: 10px; cursor: pointer;
    font-family: inherit; font-size: 0.9rem; font-weight: 600;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.2s;
  }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,0,0,0.35); }
  .btn:active { transform: translateY(0); }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
  }
  .btn-secondary {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }

  /* ── Preview ── */
  #previewSection {
    width: 100%;
    display: none;
    flex-direction: column;
    gap: 24px;
    align-items: center;
  }
  .preview-img {
    max-height: 320px;
    max-width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    object-fit: contain;
    background: #000;
  }

  /* ── Result ── */
  #resultSection {
    width: 100%;
    display: none;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    gap: 24px;
    flex-direction: column;
  }
  .result-header {
    display: flex; align-items: center; gap: 16px;
  }
  .result-icon {
    width: 56px; height: 56px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
  }
  .result-icon.normal    { background: rgba(16,185,129,0.15); }
  .result-icon.pneumonia { background: rgba(239,68,68,0.15);  }
  .result-label {
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px;
  }
  .result-label.normal    { color: var(--green); }
  .result-label.pneumonia { color: var(--red);   }
  .result-sub { color: var(--muted); font-size: 0.85rem; margin-top: 2px; }

  /* Confidence bars */
  .bars { display: flex; flex-direction: column; gap: 14px; }
  .bar-row { display: flex; flex-direction: column; gap: 6px; }
  .bar-meta { display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 500; }
  .bar-track {
    height: 10px; background: var(--surface);
    border-radius: 99px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.7s cubic-bezier(0.4,0,0.2,1);
  }
  .bar-fill.normal    { background: linear-gradient(90deg, #10b981, #34d399); }
  .bar-fill.pneumonia { background: linear-gradient(90deg, #ef4444, #f97316); }

  /* ── Spinner ── */
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Info chips ── */
  .chips { display: flex; gap: 10px; flex-wrap: wrap; }
  .chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--muted);
  }
  .chip strong { color: var(--text); }

  /* ── Disclaimer ── */
  .disclaimer {
    width: 100%;
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.8rem;
    color: #f87171;
    line-height: 1.6;
  }

  footer {
    margin-top: auto;
    padding: 24px;
    color: var(--muted);
    font-size: 0.78rem;
    text-align: center;
  }

  @media (max-width: 600px) {
    .hero-title { font-size: 1.5rem; }
    header { padding: 16px 20px; }
    main   { padding: 32px 16px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-icon">🔬</div>
  <div>
    <h1>CancerDetect AI <span>| Chest X-Ray</span></h1>
  </div>
  <div class="badge">ResNet-18</div>
</header>

<main>
  <h2 class="hero-title">Pneumonia Detection from Chest X-Rays</h2>
  <p class="hero-sub">
    Upload a chest X-ray image and our AI model will classify it as
    <strong>Normal</strong> or <strong>Pneumonia</strong> with confidence scores.
  </p>

  <!-- Upload -->
  <div class="upload-card" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <div class="upload-icon">📁</div>
    <h2>Drop your X-ray here</h2>
    <p>Accepts JPEG, PNG, BMP &nbsp;|&nbsp; Max 20 MB</p>
    <button class="btn btn-secondary" onclick="event.stopPropagation(); document.getElementById('fileInput').click()">
      Browse File
    </button>
    <input type="file" id="fileInput" accept="image/*"/>
  </div>

  <!-- Preview -->
  <div id="previewSection">
    <img id="previewImg" class="preview-img" src="" alt="X-Ray preview"/>
    <div style="display:flex; gap:12px; align-items:center;">
      <button class="btn btn-primary" id="analyzeBtn" onclick="analyze()">
        🔍 Analyze X-Ray
      </button>
      <button class="btn btn-secondary" onclick="reset()">✕ Clear</button>
      <div class="spinner" id="spinner"></div>
    </div>
  </div>

  <!-- Result -->
  <div id="resultSection">
    <div class="result-header">
      <div class="result-icon" id="resultIcon"></div>
      <div>
        <div class="result-label" id="resultLabel"></div>
        <div class="result-sub" id="resultSub"></div>
      </div>
    </div>

    <div class="bars" id="barsContainer"></div>

    <div class="chips" id="chipsContainer"></div>
  </div>

  <!-- Disclaimer -->
  <div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational
    purposes only. It is <strong>not</strong> a certified medical device and must not be
    used to make clinical decisions. Always consult a qualified healthcare professional.
  </div>
</main>

<footer>Cancer Detection AI · Vision Transformer Project · For Research Use Only</footer>

<script>
  let selectedFile = null;

  // Drag & drop
  const dropZone = document.getElementById('dropZone');
  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) handleFile(f);
  });

  document.getElementById('fileInput').addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      document.getElementById('previewImg').src = ev.target.result;
      document.getElementById('previewSection').style.display = 'flex';
      document.getElementById('resultSection').style.display  = 'none';
    };
    reader.readAsDataURL(file);
  }

  function reset() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('resultSection').style.display  = 'none';
  }

  async function analyze() {
    if (!selectedFile) return;
    const btn     = document.getElementById('analyzeBtn');
    const spinner = document.getElementById('spinner');
    btn.disabled  = true;
    spinner.style.display = 'block';

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
      const res  = await fetch('/predict', { method: 'POST', body: fd });
      const data = await res.json();

      if (data.error) { alert('Error: ' + data.error); return; }

      // Populate result
      const pred  = data.prediction;           // "NORMAL" or "PNEUMONIA"
      const cls   = pred.toLowerCase();
      const conf  = data.confidence.toFixed(1);
      const probs = data.probabilities;

      document.getElementById('resultIcon').className  = `result-icon ${cls}`;
      document.getElementById('resultIcon').textContent = cls === 'normal' ? '✅' : '⚠️';
      document.getElementById('resultLabel').className  = `result-label ${cls}`;
      document.getElementById('resultLabel').textContent = pred;
      document.getElementById('resultSub').textContent   = `Confidence: ${conf}%`;

      // Bars
      const barsHtml = Object.entries(probs).map(([name, pct]) => `
        <div class="bar-row">
          <div class="bar-meta">
            <span>${name}</span><span>${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${name.toLowerCase()}"
                 style="width: 0%"
                 data-target="${pct}"></div>
          </div>
        </div>
      `).join('');
      document.getElementById('barsContainer').innerHTML = barsHtml;

      // Chips
      document.getElementById('chipsContainer').innerHTML = `
        <div class="chip">Model: <strong>ResNet-18</strong></div>
        <div class="chip">Input: <strong>224 × 224</strong></div>
        <div class="chip">Classes: <strong>Normal / Pneumonia</strong></div>
        <div class="chip">Device: <strong>${data.device || 'CPU'}</strong></div>
      `;

      document.getElementById('resultSection').style.display = 'flex';

      // Animate bars
      requestAnimationFrame(() => {
        document.querySelectorAll('.bar-fill').forEach(el => {
          el.style.width = el.dataset.target + '%';
        });
      });

    } catch(err) {
      alert('Request failed: ' + err.message);
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
    }
  }
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 400

    result = predict(pil_img)
    result["device"] = device.upper()
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "device": device,
    })


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 Cancer Detection Web App")
    print(f"   Model : {MODEL_PATH}")
    print(f"   Device: {device}")
    if not MODEL_PATH.exists():
        print("   ⚠️  Model not found — run:  python Backend/train.py")
    print("\n   Open → http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
