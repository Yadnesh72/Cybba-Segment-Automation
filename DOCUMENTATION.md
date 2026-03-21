# Cybba Segment Automation — Project Documentation

> **Version**: 1.0
> **Last Updated**: March 2026
> **Status**: Active Development

---

## Table of Contents

1. [What This Project Solves](#1-what-this-project-solves)
2. [How It Works — The Full Pipeline](#2-how-it-works--the-full-pipeline)
3. [System Architecture](#3-system-architecture)
4. [Feature Reference](#4-feature-reference)
5. [Technology Stack](#5-technology-stack)
6. [API Endpoints](#6-api-endpoints)
7. [Configuration Reference](#7-configuration-reference)
8. [Data Files & Schemas](#8-data-files--schemas)
9. [Docker Setup & Services](#9-docker-setup--services)
10. [LLM Web Assistance — Gap Analysis Engine](#10-llm-web-assistance--gap-analysis-engine)
11. [UI Pages & Components](#11-ui-pages--components)
12. [Database & Persistence](#12-database--persistence)
13. [Roadmap & Planned Work](#13-roadmap--planned-work)

---

## 1. What This Project Solves

### The Problem

Cybba competes in the **data marketplace** — a space where companies like Experian, TransUnion, Acxiom, Oracle, and Neustar sell thousands of audience segments to advertisers. Cybba's catalog is a fraction the size of these incumbents.

Manually identifying which audience segments to build next is extremely time-consuming. A data strategist would have to:
- Read through tens of thousands of competitor segments
- Identify which categories Cybba is missing entirely
- Figure out how to build those segments using Cybba's existing underived data components
- Assign correct taxonomy (L1/L2 categories), write descriptions, and estimate pricing

All of this, repeated constantly as the market evolves.

### What This System Does

**Cybba Segment Automation** automates the entire above process end-to-end:

1. **Reads the full data marketplace catalog** (Input A — 750,000+ rows from all providers including Cybba)
2. **Reads Cybba's underived data components** (Input B — the raw field:value building blocks Cybba owns)
3. **Runs TF-IDF matching** to determine how competitor segments can be replicated using Cybba's components
4. **Generates proposals**: new segment names, taxonomy (L1/L2), descriptions, and prices
5. **Filters and validates**: deduplicates, normalizes naming, checks what's already in Cybba's catalog
6. **Exports a ready-to-use CSV** that can be handed directly to the data team

Additionally, the **LLM gap analysis engine** actively looks for category areas competitors dominate that Cybba hasn't touched yet — then generates creative segments to fill those gaps.

### Business Outcome

- Replaces days of manual analysis with a single pipeline run (minutes)
- Surfaces real market opportunities backed by competitor volume data
- Ensures consistent naming, pricing, and taxonomy across all new segments
- Continuously tracks coverage so the team knows what's built vs. what's not

---

## 2. How It Works — The Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUTS                               │
│                                                              │
│  Input A: Data_Marketplace_Full_Catalog.csv                  │
│  All providers + Cybba (750,000+ segments)                   │
│                                                              │
│  Input B: Data_Marketplace_Segments_for_Cybba.csv            │
│  Underived components (field:value pairs, LiveRamp IDs)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1 — TF-IDF MATCHING  (segment_expansion_model.py)      │
│                                                              │
│  For each competitor segment in Input A:                     │
│    1. Vectorize segment name + description (TF-IDF, 1-2gram) │
│    2. Find top-K matching underived components (Input B)     │
│    3. Compose proposal: Cybba > L1 > L2 > Leaf               │
│    4. Score: cosine similarity (0–1)                         │
│                                                              │
│  → 500+ raw proposals                                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2 — VALIDATION  (validators.py)                        │
│                                                              │
│  ✓ Deduplicate exact matches                                 │
│  ✓ Near-duplicate check (>93% similarity → hash suffix)      │
│  ✓ Enforce Cybba > L1 > L2 > Leaf naming format             │
│  ✓ Remove code-like leaves ("C123", "TX45")                  │
│  ✓ Validate component IDs exist in Input B                   │
│                                                              │
│  → 100–300 validated proposals                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3 — TAXONOMY ASSIGNMENT  (ML models)                   │
│                                                              │
│  L1 Classifier:  logistic regression → L1 + confidence       │
│  L2 Retriever:   nearest-neighbor → top-5 L2 candidates      │
│  Thresholds: L1 confidence ≥ 0.60 | L2 similarity ≥ 0.25    │
│  Fallback: open-world (no forced category if model unsure)   │
│                                                              │
│  → L1 + L2 columns populated                                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4 — LLM DESCRIPTIONS  (llm_descriptions.py)           │
│  [optional — toggle per run]                                  │
│                                                              │
│  Ollama llama3.1 → natural language segment description      │
│  Batch size: 20 | Cached in-memory per session               │
│                                                              │
│  → "Segment Description" column added                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5 — COVERAGE DETECTION  (segment_expansion_model.py)  │
│  [optional — toggle per run]                                  │
│                                                              │
│  TF-IDF search against existing Cybba catalog               │
│  ≥85% match → "covered" → moved to coverage.csv             │
│  Uncovered proposals continue to pricing                     │
│                                                              │
│  → coverage.csv generated                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6 — PRICING PREDICTION  (pricing_engine.py)           │
│  [optional — toggle per run]                                  │
│                                                              │
│  Trained scikit-learn model → 7 price metrics:              │
│    • Digital Ad Targeting CPM                               │
│    • Content Marketing CPM                                   │
│    • TV Targeting CPM                                        │
│    • Cost Per Click (CPC)                                    │
│    • Programmatic % of Media                                 │
│    • CPM Cap                                                 │
│    • Advertiser Direct % of Media                            │
│                                                              │
│  → 7 price columns added                                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7 — LLM WEB ASSISTANCE  (web_assistance.py)           │
│  [optional — toggle per run]                                  │
│                                                              │
│  Real gap analysis: Cybba L1/L2 vs competitor L1/L2          │
│  One focused Ollama call per identified gap                  │
│  Generates segments for categories Cybba is missing          │
│                                                              │
│  → Merged into validated proposals                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         OUTPUTS                              │
│                                                              │
│  proposals.csv                 → Raw 500+ candidates         │
│  proposals_validated.csv       → After validation           │
│  coverage.csv                  → Already-covered segments    │
│  Cybba_New_Additional_Segments.csv → Final with pricing      │
│  run.log                       → Full pipeline log           │
│                                                              │
│  Also persisted: SQLite (runs.db) for per-user tracking      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DOCKER COMPOSE                            │
│                                                              │
│   ┌─────────────────────┐      ┌─────────────────────────┐  │
│   │   FRONTEND          │      │   BACKEND               │  │
│   │   React + TypeScript│◄────►│   FastAPI + Python      │  │
│   │   Vite (HMR)        │      │   uvicorn               │  │
│   │   Port: 5173        │      │   Port: 8000            │  │
│   └─────────────────────┘      └────────────┬────────────┘  │
│                                             │               │
│                               ┌─────────────▼────────────┐  │
│                               │   PIPELINE MODULES        │  │
│                               │                           │  │
│                               │ segment_expansion_model.py│  │
│                               │ validators.py             │  │
│                               │ pricing_engine.py         │  │
│                               │ llm_descriptions.py       │  │
│                               │ web_assistance.py         │  │
│                               │ suggestions.py            │  │
│                               │ analytics.py              │  │
│                               │ comparison.py             │  │
│                               └────────────┬─────────────┘  │
│                                            │                │
│                    ┌───────────────────────▼──────────────┐ │
│                    │  SHARED STORAGE (bind mounts)         │ │
│                    │                                       │ │
│                    │  ./Data/           → CSV inputs/out   │ │
│                    │  ./backend/data/   → SQLite DB        │ │
│                    │  ./backend/models/ → ML models        │ │
│                    │  ./backend/config/ → config.yml       │ │
│                    └───────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼ (host machine)
                    ┌─────────────────────┐
                    │   OLLAMA            │
                    │   host.docker.      │
                    │   internal:11434    │
                    │                     │
                    │   llama3.1         │
                    │   llama3.2-vision  │
                    └─────────────────────┘
```

### Streaming Architecture

The frontend connects to the backend via **Server-Sent Events (SSE)** for real-time pipeline progress:

```
Frontend                           Backend (FastAPI)
   │                                      │
   │── GET /api/run/stream ──────────────►│
   │                                      │ (pipeline starts)
   │◄─ event: run_id ─────────────────────│
   │◄─ event: summary ────────────────────│  (early stats)
   │◄─ event: row ────────────────────────│  (one per validated row)
   │◄─ event: row ────────────────────────│
   │◄─ ...                               │
   │◄─ event: done ───────────────────────│  (full result + run_id)
```

---

## 4. Feature Reference

### Core Pipeline Features

| Feature | Module | Default | Description |
|---------|--------|---------|-------------|
| Segment Generation | `segment_expansion_model.py` | Always on | TF-IDF matching against Input B |
| Validation & Dedup | `validators.py` | Always on | Naming, dedup, safety checks |
| Taxonomy Assignment | ML models | Toggle per run | L1/L2 classification via trained models |
| LLM Descriptions | `llm_descriptions.py` | Toggle per run | Natural language descriptions via Ollama |
| Coverage Detection | `segment_expansion_model.py` | Toggle per run | Finds existing Cybba matches (≥85%) |
| Pricing Prediction | `pricing_engine.py` | Toggle per run | 7 price metrics via trained regression model |
| LLM Web Assistance | `web_assistance.py` | Toggle per run | Gap-based generation via Ollama |

### Analysis & UI Features

| Feature | Module | Description |
|---------|--------|-------------|
| Smart Suggestions | `suggestions.py` | Ranks segments by (high comp match + low Cybba match) |
| Comparison Tool | `comparison.py` | TF-IDF search generated segments vs full catalog |
| Analytics Dashboard | `analytics.py` | Charts: rank/uniqueness distribution, scatter, pie |
| AI Chart Insights | `llm_analytics.py` | Ollama llama3.2-vision analyzes chart screenshots |
| Catalog Browser | `api.py` | Paginated, searchable Cybba segment browser |
| Per-User Run Tracking | `persist.py` | SQLite: run_id → user mapping, auto-restore last run |
| Dark / Light Theme | `styles.css` | Full CSS variable theming, purple sidebar, glass cards |

---

## 5. Technology Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Web framework | FastAPI | Latest |
| Server | uvicorn (ASGI) | Latest |
| Language | Python | 3.10 |
| Vectorization | scikit-learn TF-IDF | Latest |
| ML models | scikit-learn | Latest |
| Data processing | pandas, numpy | Latest |
| LLM integration | Ollama (HTTP) | llama3.1, llama3.2-vision |
| Persistence | SQLite (via `sqlite3`) | Built-in |
| Model serialization | joblib | Latest |
| Similarity | cosine_similarity | sklearn |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | React | 18 |
| Language | TypeScript | Latest |
| Build tool | Vite | Latest |
| Charts | Recharts | Latest |
| Animation | Framer Motion | Latest |
| Screenshot | html2canvas | Latest |
| Styling | CSS (custom theme) | — |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| Containerization | Docker + Docker Compose |
| Backend container | Python 3.10-slim |
| Frontend container | Node 20-alpine |
| Host LLM | Ollama (runs on host machine) |
| Storage | Bind-mounted volumes (CSV, SQLite, models) |

---

## 6. API Endpoints

### Pipeline

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| `POST` | `/api/run` | Same as stream | Full run, returns complete result at once |
| `GET` | `/api/run/stream` | `max_rows`, `enable_descriptions`, `enable_pricing`, `enable_taxonomy`, `enable_coverage`, `enable_llm_generation`, `enable_llm_web_assistance` | **Recommended.** SSE streaming pipeline run |
| `GET` | `/api/runs/{run_id}` | — | Fetch cached run by ID |
| `GET` | `/api/download.csv` | `run_id`, `mode` (`final`/`validated`/`proposals`/`coverage`) | Download CSV output |

### User / Session

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/users/me/last-run` | Get last run ID for current user |
| `POST` | `/api/users/me/last-run` | Set last run ID for current user |
| `DELETE` | `/api/users/me/last-run` | Clear last run |

### Catalog

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| `GET` | `/api/catalog/cybba` | `page`, `page_size`, `search` | Browse Cybba segments (paginated) |

### Comparison

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| `GET` | `/api/comparison/search` | `run_id`, `query`, `top_k`, `min_similarity` | TF-IDF search against full catalog |

### Suggestions

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/suggestions/generate` | `run_id` | Generate smart suggestions for a run |
| `POST` | `/api/suggestions/analyze` | `segment_name`, `competitor_matches`, `cybba_matches` | Ollama analysis of one suggestion |

### Analytics

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/analytics/insights` | `image_base64`, `chart_type`, `metadata` | Ollama vision analysis of chart image |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Liveness check (returns `{"status": "ok"}`) |

---

## 7. Configuration Reference

**File**: `backend/config/config.yml`

### Paths

```yaml
paths:
  base_dir: "Data"
  input_dir: "/app/Data/Input/Raw_segments"
  output_dir: "/app/Data/Output/Segment_Expansion"
```

### Files

```yaml
files:
  input_a: "Data_Marketplace_Full_Catalog.csv"        # All providers
  input_b: "Data_Marketplace_Segments_for_Cybba.csv"  # Underived components
  output_format: "Output_format.csv"                   # Column template
```

### Taxonomy Settings

```yaml
taxonomy:
  separator: " > "                           # L1 > L2 > Leaf separator
  strip_leading_asterisks: true              # Remove * prefix from names
  use_taxonomy_gating: false                 # Hard-block unlisted L1/L2 (keep OFF)
  nn_min_similarity: 0.35                   # NN fallback similarity threshold
  l1_min_confidence: 0.60                   # Minimum confidence for L1 model
  enable_l2_cfg_default_fallback: false      # Don't auto-assign L2 from config defaults
  enable_l2_catalog_fallback: false          # Don't fall back to catalog L2
  enable_path_inference: false               # No rule-based path inference (model-first)
  overwrite_existing_taxonomy: false         # Fill blanks only, don't overwrite
```

### Generation Settings

```yaml
generation:
  max_proposals: 500                         # Max candidates before validation
  max_components: 2                          # Max Input B segments per proposal
  retrieval_pool_size: 20                    # Candidate pool before scoring
  max_candidates_from_A: 5000               # Max competitor rows to process
  min_composition_similarity: 0.15          # Minimum TF-IDF score to keep a proposal
  use_allowed_categories_gating: false       # Open-world (no category whitelist)
  fallback_l1: null                          # No forced L1 fallback
  fallback_l2: null                          # No forced L2 fallback
```

### Deduplication

```yaml
dedupe:
  near_duplicate_threshold: 0.93            # >93% similarity = duplicate
  resolve_name_collision: true              # Add hash suffix on collision
  collision_hash_len: 8                     # Hash suffix length
```

### Coverage Detection

```yaml
coverage:
  enable: true
  cover_threshold: 0.85                     # ≥85% match = already covered
  review_threshold: 0.85                    # ≥85% = flag for review
```

### Descriptions (Ollama)

```yaml
descriptions:
  enable: true
  model: "llama3.1"
  batch_size: 20
  ollama_url: "http://host.docker.internal:11434/api/generate"
```

### Pricing Model

```yaml
pricing_model:
  enable: true
  model_path: "models/pricing_model.joblib"
  defaults:
    provider_name: "Cybba"
    country: "USA"
    currency: "USD"
```

### Taxonomy Models

```yaml
taxonomy_model:
  enable: true
  l1_model_path: "/app/Data/Models/cybba_taxonomy_L1.joblib"
  l2_model_path: "/app/Data/Models/cybba_taxonomy_L2.joblib"
  l2_top_k: 5                               # Return top-5 L2 candidates
  l2_min_similarity: 0.25
  leaf_top_k: 10
  leaf_min_similarity: 0.20
  l1_min_confidence: 0.60
  overwrite_existing_taxonomy: false
```

### LLM Web Assistance (Gap Analysis)

```yaml
llm_web_assistance:
  enable: false                              # Toggle in UI per run
  model: "llama3.1"
  ollama_url: "http://host.docker.internal:11434/api/generate"
  max_segments: 20                           # Gap segment budget
  max_search_queries: 5                      # Legacy (unused)
  timeout_seconds: 60
```

### Suggestions

```yaml
suggestions:
  model: "llama3.1"
  ollama_url: "http://host.docker.internal:11434/api/generate"
  analysis_timeout_seconds: 60
  top_n: 25                                  # Return top-25 suggestions
  min_competitor_sim: 0.10
  min_cybba_sim: 0.10
```

### Analytics LLM

```yaml
analytics_llm:
  enable: true
  model: "llama3.2-vision"
  ollama_url: "http://host.docker.internal:11434/api/generate"
```

---

## 8. Data Files & Schemas

### Input A — `Data_Marketplace_Full_Catalog.csv`

Full marketplace catalog. All providers including Cybba. Used for:
- Competitor segment source for TF-IDF matching
- Gap analysis (Cybba vs competitor L1/L2 coverage)
- Comparison tool TF-IDF index

Key columns:
| Column | Description |
|--------|-------------|
| `Provider Name` | e.g. Experian, Acxiom, Cybba, Oracle Data Cloud |
| `Segment Name` | Full hierarchical name, e.g. `Experian > Autos > SUV Buyers` |
| `Segment Description` | Natural language description |

### Input B — `Data_Marketplace_Segments_for_Cybba.csv`

Cybba's underived building blocks. Each row is a field:value pair that can be "composed" into new segments.

Key columns:
| Column | Description |
|--------|-------------|
| `LiveRamp Field ID` | Field identifier |
| `LiveRamp Value ID` | Value identifier |
| `LiveRamp Segment ID` | Fallback segment ID |
| `Segment Name` | Human-readable name of the component |
| `Segment Description` | Description of what the component represents |

### Output — `Cybba_New_Additional_Segments.csv`

Final output file, formatted to match Cybba's marketplace submission template.

Key columns:
| Column | Description |
|--------|-------------|
| `Proposed New Segment Name` | `Cybba > L1 > L2 > Leaf` |
| `Segment Description` | LLM-generated description |
| `L1` | Top-level taxonomy category |
| `L2` | Subcategory |
| `Leaf` | Specific audience name |
| `Non Derived Segments utilized` | Pipe-separated field:value IDs from Input B |
| `Composition Similarity` | TF-IDF score (0–1) how well components match |
| `Digital Ad Targeting CPM` | Predicted price |
| `Content Marketing CPM` | Predicted price |
| `TV Targeting CPM` | Predicted price |
| `Cost Per Click` | Predicted price |
| `Programmatic % of Media` | Predicted percentage |
| `CPM Cap` | Predicted price cap |
| `Advertiser Direct % of Media` | Predicted percentage |
| `Competitor Provider` | Source competitor (or "LLM Assisted") |
| `Competitor Segment Name` | What competitor segment inspired this |
| `Closest Cybba Segment` | Best matching existing Cybba segment |
| `Closest Cybba Similarity` | Similarity score to closest match |

---

## 9. Docker Setup & Services

### `docker-compose.yml` Overview

```yaml
services:

  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: ./backend/config/credentials.env
    environment:
      LOG_LEVEL: "INFO"
    volumes:
      - ./Data:/app/Data                      # CSV files in/out
      - ./backend/data:/app/data              # SQLite runs.db
      - ./backend/src:/app/src                # Hot-reload source
      - ./backend/api.py:/app/api.py          # Hot-reload API
      - ./backend/config:/app/config          # config.yml + credentials
      - ./backend/models:/app/models          # Trained ML models

  frontend:
    build: ./frontend
    ports: ["5173:5173"]
    depends_on: [backend]
    environment:
      VITE_API_BASE: "http://127.0.0.1:8000"
    volumes:
      - ./frontend/src:/app/src               # Hot-reload source
      - ./frontend/components:/app/components  # Hot-reload components
      - ./frontend/styles.css:/app/styles.css  # Hot-reload styles
```

### Key Commands

```bash
# Start all services
docker compose up -d

# View backend logs in real-time
docker compose logs -f backend

# Restart after config/env changes (needed for env_file updates)
docker compose restart backend

# Restart after Dockerfile changes
docker compose up --build

# Run a quick pipeline test in the container
docker compose exec backend python -c "
  import sys; sys.path.insert(0, '/app/src')
  from web_assistance import generate_web_assisted_segments
  import yaml
  cfg = yaml.safe_load(open('/app/config/config.yml'))
  result = generate_web_assisted_segments(cfg, [], max_segments=4)
  print(result[['L1','L2','Proposed New Segment Name']].to_string())
"
```

### Important Notes

- **Hot-reload**: Code changes in `src/`, `api.py`, `components/`, and `styles.css` are reflected instantly (no restart needed) — Docker bind mounts + uvicorn `--reload` + Vite HMR handle this.
- **inode issue**: If you copy a file with `cp` instead of editing in-place, the bind mount breaks. Always `edit` files in-place, or `docker compose restart` to re-sync.
- **env_file**: `credentials.env` is loaded into the backend container. Changes to this file require `docker compose restart backend`.

---

## 10. LLM Web Assistance — Gap Analysis Engine

**File**: `backend/src/web_assistance.py`

### Purpose

Instead of asking a single LLM call to generate all segments (which causes the model to gravitate toward familiar/easy categories), the gap analysis engine:

1. Finds **real gaps** by comparing Cybba's L1/L2 coverage vs top competitor coverage
2. Makes **one focused LLM call per gap** — each call is laser-targeted to a single category
3. Grows an **exclusion list** across calls so no category or name gets repeated
4. Falls back to **trend themes** only after all real gaps are addressed

### Gap Detection Logic

**Competitor providers analyzed** (from Input A):
`Experian, TransUnion, Acxiom, Oracle Data Cloud, Neustar, IRI, LiveRamp, Dun & Bradstreet, Bombora, Epsilon, Nielsen, Polk, comScore, V12 Group, TruAudience, Verisk, Kochava, Lotame`

**Three gap types** (priority order):

| Type | Condition | Priority |
|------|-----------|----------|
| `missing_l2` | Competitor ≥3 segments in this L2, Cybba has **0** | Highest |
| `thin_l2` | Competitor ≥5 segments, Cybba has **<2** | Medium |
| `sparse_l1` | Competitor ≥10 segments in L1, Cybba has **<25%** coverage | Lower |

### Per-Gap Call Flow

```
gaps = _find_gaps(catalog)          # Up to 35 prioritized gaps
already_generated = []

for gap in gaps:
    if len(records) >= max_segments:
        break

    prompt = _build_focused_prompt(
        gap=gap,
        cybba_sample=existing_names[:50],
        already_generated=already_generated,   # ← grows each iteration
        n_segments=2,
    )

    raw = _call_ollama(prompt, ...)
    candidates = _parse_items(raw)             # filter placeholder leaves
    novel = _novel_mask(candidates, existing + already_generated)

    for c in novel_candidates:
        records.append(c)
        already_generated.append(c["name"])    # ← prevent next call repeating
```

### Trend Fallback Themes

If gaps are exhausted before reaching `max_segments`, the system fills the remainder from these trend themes:

```
Biohackers: CGM users, cold-plunge enthusiasts, nootropic buyers
Creator Economy: Substack writers, course creators, podcast hosts
AI Tool Power Users: ChatGPT Pro subscribers, prompt engineers
Women's Health: menopause support seekers, fertility tracking users
EV Ecosystem: EV owners, home charging station buyers
Silver Economy: tech-savvy retirees, active seniors 65+
Mental Health Tech: therapy platform users, mood tracking apps
Sustainable Living: zero-waste adopters, carbon offset purchasers
Buy Now Pay Later: BNPL frequent shoppers, Klarna/Afterpay users
Digital Nomads: remote work travelers, co-living space members
Legal Operations: legal ops managers, e-discovery tool users
Sleep Tech: smart mattress owners, sleep tracking users
Pickleball Players: equipment buyers, court booking users
Embedded Finance: neobank customers, payroll advance app users
Fantasy Sports & Betting: daily fantasy participants, sports betting users
Longevity Seekers: anti-aging supplement buyers, NAD+ purchasers
```

### Novelty Filter

Every candidate passes through a TF-IDF cosine similarity check:
- **Threshold**: 0.45 similarity to any existing Cybba or already-generated name = filtered out
- Prevents near-duplicates even across very different phrasings

---

## 11. UI Pages & Components

### Navigation Structure

```
SideMenu
├── Segments          (main output table, run controls, download)
├── Comparison        (search generated vs full catalog)
├── Analytics         (charts + AI insights)
├── Cybba Segments    (browse existing Cybba catalog)
└── Suggestions       (smart gap suggestions with Ollama analysis)
```

### Segments Page

The main page. Shows generated segments with pricing. Key controls:

**Run Settings** (per-run toggles):
- ☑ Generate Descriptions (Ollama llama3.1)
- ☑ Apply Pricing (ML model)
- ☑ Apply Taxonomy (ML models)
- ☑ Detect Coverage (find existing Cybba matches)
- ☑ LLM Web Assistance (gap-based generation)

**Live KPI Strip** (updates during streaming):
- Rank Score: mean, p10, p50, p90, min
- Uniqueness Score: mean, p10, p50, p90
- Total rows processed

**Table columns**: Segment Name, Description, Digital Ad CPM, Content CPM, TV CPM, CPC, Programmatic %, CPM Cap, Advertiser Direct %

**Download modes**:
- `final` — complete output with pricing (default)
- `validated` — pre-pricing validated rows
- `proposals` — raw 500+ proposals before validation
- `coverage` — segments already covered by Cybba

### Comparison Page

- Search bar to filter segments by name
- For each segment: shows top competitor matches + top Cybba matches with similarity pills
- Adjustable: min similarity slider, top-K results
- Accordion-style — expand any segment to see its matches

### Analytics Page

Four charts:
1. **Rank Score Distribution** — histogram with min/max/percentile markers
2. **Uniqueness Score Distribution** — histogram
3. **Rank vs Uniqueness Scatter** — with Pearson correlation coefficient
4. **L1 Category Breakdown** — pie chart

**AI Insights**: Takes screenshot of charts → sends to Ollama llama3.2-vision → displays trend analysis, anomalies, and recommendations

### Cybba Segments Page

- Browse the full Cybba section of Input A
- Search by name or description
- Paginated (25 rows/page)
- Useful for spot-checking what's already in the catalog before running

### Suggestions Page

Smart gap identifier:
- Scores each generated segment: high competitor TF-IDF match + low Cybba TF-IDF match = strong suggestion (= Cybba is missing this)
- Shows competitor and Cybba similarity scores side-by-side
- "Analyze" button: sends to Ollama for contextual commentary

### Theming

Two complete themes:

**Dark mode** (default): Deep navy background, glass card panels, purple accent

**Light mode**: Soft periwinkle background (`#e8edf8`), white glass cards, purple gradient sidebar, purple table headers

Toggle: sun/moon button in the sidebar footer. Theme persists via `localStorage`.

---

## 12. Database & Persistence

**Location**: `backend/data/runs.db` (SQLite)

### Schema

```sql
-- One record per pipeline run
CREATE TABLE runs (
  id          TEXT PRIMARY KEY,   -- UUID
  created_by  TEXT,               -- user_id (or "anonymous")
  created_at  TEXT,               -- ISO timestamp
  summary     TEXT,               -- JSON: {total_proposals, validated, covered, ...}
  rows        TEXT,               -- JSON list of validated row dicts
  final_rows  TEXT                -- JSON list of priced row dicts
);

-- Maps user → their last run
CREATE TABLE user_last_runs (
  user_id     TEXT PRIMARY KEY,   -- User identifier
  run_id      TEXT,               -- FK → runs.id
  updated_at  TEXT                -- ISO timestamp
);
```

### Behavior

- On run completion: result persisted to `runs` table, user's `user_last_runs` entry updated
- On page load: frontend calls `GET /api/users/me/last-run` → fetches and restores last run automatically
- Download: reads from SQLite, no need to re-run the pipeline
- Cleanup: runs older than 30 minutes are considered stale (new run replaces them for the user)

---

## 13. Roadmap & Planned Work

### In Progress

- [x] LLM Web Assistance — per-gap focused Ollama calls
- [x] Cross-call duplicate exclusion (growing `already_generated` list)
- [x] Placeholder leaf filtering (`"Specific Leaf"`, `"TBD"`, etc.)
- [x] Light mode redesign (glass cards, purple sidebar, periwinkle background)

### Short-Term (Next Steps)

- [ ] **Smarter L2 for hard gaps**: Some gaps (e.g. `Household Member > Year of Birth`) are hard for LLMs to fill creatively — add category-specific prompt variants or skip these gap types
- [ ] **Increase `SEGS_PER_GAP` to 3**: Slightly larger batches per gap call when running larger budgets (max_segments > 30)
- [ ] **UI indicator for LLM-assisted rows**: Show a badge/chip in the table for rows that came from web assistance vs TF-IDF matching
- [ ] **Streaming progress for LLM web assistance**: Show "Gap 3/10 completed" in the KPI strip during streaming

### Medium-Term

- [ ] **Dataset collection for model training** *(high priority)*:
  - IAB Tech Lab Audience Taxonomy (v3.0) — free, official standard
  - IAB Content Taxonomy — companion to audience taxonomy
  - Oracle BlueKai published taxonomy (scraped)
  - Bombora B2B intent topics (~4,000 topics, publicly documented)
  - Lotame, Eyeota, Salesforce DMP published taxonomies
  - US Census ACS (20,000+ demographic variables → segment names)
  - Kaggle datasets: `audience segments`, `consumer purchase intent`, `propensity model`

- [ ] **Fine-tune Ollama model on segment data**:
  - Collect: competitor segment name → Cybba-style segment name pairs
  - Format: instruction-tuning JSONL (system prompt + user query + ideal completion)
  - Target: `llama3.1` or `mistral` fine-tune via Ollama's model import
  - Goal: model that natively understands Cybba naming conventions

- [ ] **Taxonomy model retraining** with larger dataset (currently trained on limited internal data)
- [ ] **Pricing model improvement**: Add more training examples, feature-engineer better L1/L2 signals
- [ ] **Batch export to LiveRamp format**: Direct integration with LiveRamp segment upload API

### Long-Term

- [ ] **Active learning loop**: Generated segments rated by data team → feed ratings back as training signal
- [ ] **Scheduled runs**: Auto-generate suggestions weekly, diff against previous run to show "new opportunities"
- [ ] **Multi-user authentication**: Proper user sessions (currently cookie-based anonymous IDs)
- [ ] **Segment quality scoring**: Train a separate model to predict how likely a segment is to be approved/adopted
- [ ] **Vertical specialization**: Separate prompts and taxonomies for B2B vs B2C segments (currently mixed)
- [ ] **API for external tools**: Expose segment generation as a REST API for integration with Cybba's internal data tools

---

## Appendix — Project File Map

```
cybba_segment_automation/
│
├── DOCUMENTATION.md              ← This file
├── docker-compose.yml
│
├── backend/
│   ├── api.py                    Main FastAPI app (all routes + SSE)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config/
│   │   ├── config.yml            All pipeline settings
│   │   └── credentials.env       API keys + Ollama URL
│   ├── data/
│   │   └── runs.db               SQLite persistence
│   ├── models/
│   │   ├── pricing_model.joblib          7-column price regressor
│   │   ├── cybba_taxonomy_L1.joblib      L1 classifier
│   │   ├── cybba_taxonomy_L2.joblib      L2 retriever
│   │   └── cybba_taxonomy_Leaf.joblib    Leaf classifier
│   └── src/
│       ├── pipeline.py                   Main orchestration
│       ├── segment_expansion_model.py    TF-IDF core generator
│       ├── validators.py                 Naming/dedup/safety
│       ├── pricing_model.py              Price inference
│       ├── pricing_engine.py             Apply pricing to rows
│       ├── llm_descriptions.py           Ollama descriptions
│       ├── llm_analytics.py              Ollama vision (charts)
│       ├── web_assistance.py             Gap analysis + LLM generation
│       ├── suggestions.py                Smart suggestions router
│       ├── comparison.py                 TF-IDF search router
│       ├── analytics.py                  Chart insights router
│       ├── persist.py                    SQLite CRUD
│       ├── taxonomy_bundle.py            L1 model loading
│       └── taxonomy_retriever.py         L2/Leaf retrieval
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts
│   ├── styles.css                Global theme (dark + light)
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx               Routing, streaming handler, KPI strip
│   │   ├── api.ts                Fetch wrapper + TypeScript types
│   │   ├── Theme.tsx             Theme provider
│   │   └── useTheme.ts           Theme hook
│   └── components/
│       ├── SideMenu.tsx
│       ├── SegmentsTable.tsx
│       ├── RunButton.tsx
│       ├── DownloadButton.tsx
│       ├── SummaryCard.tsx
│       ├── AnalyticsPage.tsx
│       ├── ComparisonPage.tsx
│       ├── CybbaSegmentsPage.tsx
│       ├── SuggestionsPage.tsx
│       └── ThemeToggle.tsx
│
└── Data/
    ├── Input/Raw_segments/
    │   ├── Data_Marketplace_Full_Catalog.csv          Input A
    │   ├── Data_Marketplace_Segments_for_Cybba.csv    Input B
    │   ├── Output_format.csv                          Column template
    │   └── Cybba_segment_prices.csv                   Pricing training data
    └── Output/Segment_Expansion/
        ├── proposals.csv
        ├── proposals_validated.csv
        ├── coverage.csv
        └── Cybba_New_Additional_Segments.csv
```
