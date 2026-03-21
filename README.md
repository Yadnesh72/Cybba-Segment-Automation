# Cybba Segment Automation Platform

## Overview

This repository contains the **Cybba Segment Automation Platform** — a full-stack web application that automatically discovers new audience segment opportunities by analyzing the data marketplace, then generates segment names, taxonomy, descriptions, and pricing ready for submission.

Certain large files (trained models, raw input CSVs, output data) are **not included in this repo** due to GitHub file size limits. They are stored separately and must be placed manually before running the project.

---

## Prerequisites

Make sure the following are installed on your machine before starting:

| Tool | Purpose | Download |
|---|---|---|
| **Docker Desktop** | Runs the entire application | [docker.com](https://www.docker.com/products/docker-desktop/) |
| **Ollama** | Runs the local AI models | [ollama.com](https://ollama.com) |
| **Git** | Clone the repo | Pre-installed on Mac/Linux |

---

## Setup Guide

### Step 1 — Clone the repo

```bash
git clone https://github.com/Yadnesh72/Cybba-Segment-Automation.git
cd Cybba-Segment-Automation
```

---

### Step 2 — Download large files from SharePoint

Download the following folders from SharePoint (contact Yadnesh for the link):

| SharePoint Folder | What's Inside |
|---|---|
| `backend/models/` | Trained pricing + taxonomy `.joblib` model files |
| `backend/Data/Models/` | Supporting taxonomy model data |
| `Data/Input/Raw_segments/` | Raw input segment CSVs (marketplace catalog + Cybba components) |
| `Data/Output/` | Output folder (can be empty, pipeline will populate it) |

---

### Step 3 — Place files in the correct locations

After downloading, place each folder exactly as shown inside your local project root:

```
Cybba-Segment-Automation/
│
├── backend/
│   ├── models/                   ← place backend/models/ here
│   │   ├── pricing_model.joblib
│   │   ├── cybba_taxonomy_L1.joblib
│   │   ├── cybba_taxonomy_L2.joblib
│   │   └── cybba_taxonomy_Leaf.joblib
│   │
│   └── Data/
│       └── Models/               ← place backend/Data/Models/ here
│
├── Data/
│   ├── Input/
│   │   └── Raw_segments/         ← place Data/Input/Raw_segments/ here
│   │   │   ├── Data_Marketplace_Full_Catalog.csv
│   │   │   ├── Data_Marketplace_Segments_for_Cybba.csv
│   │   │   └── Output_format.csv
│   │
│   └── Output/                   ← place Data/Output/ here (can be empty)
│
├── docker-compose.yml
└── frontend/
```

---

### Step 4 — Add your credentials file

Create the credentials file the backend needs:

```bash
mkdir -p backend/config
touch backend/config/credentials.env
```

Open `backend/config/credentials.env` and add the following (ask Yadnesh for values):

```
ANTHROPIC_API_KEY=your_key_here
```

> ⚠️ Never commit this file — it is already covered by `.gitignore`

---

### Step 5 — Pull the Ollama AI models

Make sure Ollama is running, then pull the two models the platform uses:

```bash
ollama pull llama3.1
ollama pull llama3.2-vision
```

---

### Step 6 — Verify your file structure

Run this to confirm all required files are in place:

```bash
ls backend/models/
ls backend/Data/Models/
ls Data/Input/Raw_segments/
```

Each should list files, not return empty.

---

### Step 7 — Start the platform

```bash
docker compose up --build
```

This will build and start both the backend and frontend. First build takes a few minutes.

Once running, open your browser and go to:

**http://localhost:5173**

---

## How to Use

1. Open **http://localhost:5173** in your browser
2. Click **Run** in the top right
3. Select which features to enable (descriptions, pricing, taxonomy, gap generation)
4. Watch segments generate in real time
5. Use the **Comparison**, **Suggestions**, and **Analytics** pages to explore results
6. Click **Download** to export the final CSV for submission

---

## Project Structure

```
Cybba-Segment-Automation/
│
├── backend/                    # FastAPI backend (Python)
│   ├── api.py                  # All API routes + SSE streaming
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config/
│   │   ├── config.yml          # All pipeline settings
│   │   └── credentials.env     # API keys (not in repo — add manually)
│   └── src/
│       ├── pipeline.py         # Main pipeline orchestration
│       ├── web_assistance.py   # LLM gap analysis engine
│       ├── segment_expansion_model.py  # TF-IDF matching core
│       ├── pricing_engine.py   # Pricing model inference
│       ├── llm_descriptions.py # AI description generation
│       ├── suggestions.py      # Smart suggestions
│       ├── comparison.py       # Catalog comparison search
│       └── persist.py          # SQLite run persistence
│
├── frontend/                   # React + TypeScript (Vite)
│   ├── Dockerfile
│   ├── src/
│   │   ├── App.tsx             # Main app + streaming handler
│   │   └── api.ts              # API client
│   ├── components/             # All UI pages and components
│   └── styles.css              # Full dark + light theme
│
├── docker-compose.yml          # Runs backend + frontend together
├── DOCUMENTATION.md            # Full technical documentation
└── README.md                   # This file
```

---

## Stopping the Platform

```bash
docker compose down
```

---

## Common Issues

| Problem | Fix |
|---|---|
| Page won't load | Make sure Docker Desktop is running and you ran `docker compose up` |
| Descriptions not generating | Make sure Ollama is running and `llama3.1` is pulled |
| Run gives an error | Check that all files are in place per Step 3 |
| Changes to code not showing | Run `docker compose restart backend` or `docker compose restart frontend` |

---

## Notes

- Never commit the `backend/models/`, `Data/Input/`, or `Data/Output/` folders — covered by `.gitignore`
- Never commit `backend/config/credentials.env` — covered by `.gitignore`
- All AI processing runs locally via Ollama — no data leaves your machine

---

## Contact

For SharePoint access, credentials, or setup help:

**Yadnesh Chowkekar** — yadnesh.chowkekar@cybba.com
