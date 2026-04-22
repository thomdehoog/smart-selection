# Smart Selection — 3-step Workflow Redesign

Design doc for the workflow restructure on branch
`claude/smart-selection-workflow-6AIde`. Nothing below is implemented yet —
this file exists so we can agree on the shape before touching code. Paths and
line numbers refer to the current tree.

---

## 1. Current state (what's there today)

Frontend is a single React component `frontend/MicroscopySearch.jsx` (~694
lines). It exposes four linear steps:

1. **Load Data** (`StepLoad`, lines 57–174) — one text box for a plate
   directory, `n` fields-of-view, and radio buttons for crop mode / size /
   rotation. "Start" calls `/api/upload_bbbc021` followed by
   `/api/segment_and_embed`, polls `/api/status`, advances on completion.
2. **Select** (`StepSelect`, lines 176–334) — canvas with mask overlay, click
   to toggle cells, sidebar thumbs.
3. **Review** (`StepReview`, lines 336–379) — gallery of the picked cells.
4. **Search** (`StepSearch`, lines 381–499) — positive/negative refine,
   sliders for rejection strength and min similarity.

Backend (`backend/app.py`) exposes `/api/upload_bbbc021`,
`/api/segment_and_embed`, `/api/status`, `/api/mask/<i>`, `/api/image/<i>`,
`/api/objects`, `/api/crops`, `/api/search`, `/api/search_dissimilar`,
`/api/export`. Segmentation is Cellpose-SAM
(`backend/services/segmentation.py:45`). Features are DINOv2 ViT-B/14 pooled
to a 768-dim global embedding (`backend/services/embedding.py`). Search is
FAISS cosine with a negative-centroid adjustment.

Not yet in the codebase:
- any dataset registry / dropdown,
- classic per-object features (area is computed, but nothing richer),
- KNN graph construction,
- Leiden (or any) clustering,
- a segmentation endpoint that runs on one tile.

---

## 2. Target shape: three steps, per-step method dropdown

```
┌─ Step 1 ────────────┐  ┌─ Step 2 ──────────────┐  ┌─ Step 3 ──────────────────┐
│ Dataset             │  │ Segmentation          │  │ Cell Selection             │
│                     │  │                       │  │                            │
│ [ Dataset ▼ ]       │  │ [ Method ▼ ]          │  │ [ Mode ▼ ]                 │
│   presets + custom  │  │   Cellpose-SAM ✓      │  │   Similarity ✓             │
│                     │  │   Cellpose 3 ✗        │  │   Classic (thresholds) ✗   │
│ Fields of view      │  │   smart-analysis ✗    │  │   Clustering (Leiden) ✗    │
│ Crop / size / rot.  │  │                       │  │                            │
│                     │  │ Image tile browser    │  │ Gallery of current picks   │
│ [ Load dataset ]    │  │ + colored mask        │  │ (always visible, top)      │
│                     │  │ overlay (random color │  │                            │
│                     │  │ per object)           │  │ Mode-specific panel below  │
│                     │  │                       │  │                            │
│                     │  │ Opacity slider        │  │                            │
│                     │  │ [ Run preview ]       │  │                            │
│                     │  │ [ Continue → ]        │  │                            │
└─────────────────────┘  └───────────────────────┘  └────────────────────────────┘
```

The per-step dropdown makes the roadmap visible: disabled options stay in the
menu with a short "why" tooltip, so future work is explicit in the UI instead
of living only in a doc.

---

## 3. Step 1 — Dataset

**Dropdown:** dataset source.

| Option         | Enabled | Behavior                                                                   |
| -------------- | ------- | -------------------------------------------------------------------------- |
| BBBC021 Week 1 | yes     | preset path `…/bbbc021_raw/Week1_22123/`                                   |
| Custom path…   | yes     | reveals a text input (current behavior)                                    |
| _more presets_ | —       | easy to add; eventually backed by `GET /api/datasets` (not required today) |

**Other controls** (kept from the current Step 1):
- Fields of view (`n`, default 5)
- Crop mode: single cell / neighborhood
- Size: invariant / aware
- Rotation: invariant / aware

**Action:** "Load dataset" → `POST /api/upload_bbbc021`. On success advance to
Step 2. This is a split of the current Step 1 — we no longer kick off the
full pipeline here; just the image load.

**Why split?** So Step 2 can show a cheap preview before the user pays for
embedding + indexing.

---

## 4. Step 2 — Segmentation

**Dropdown:** segmentation method.

| Option               | Enabled | Notes                                                                                                                    |
| -------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| Cellpose-SAM (cpsam) | yes     | `services/segmentation.py:45`                                                                                            |
| Cellpose 3           | no      | Cellpose 3 and Cellpose-SAM can't live in the same Python env. Candidate path is to run it via the `smart-analysis` repo |
| smart-analysis       | no      | Placeholder for plugging external segmenters via the companion repo; nothing wired yet                                   |

**Tile browser:** vertical strip of `1..n` tile buttons (same pattern as the
current `StepSelect` sidebar, `MicroscopySearch.jsx:287–299`). Picking a tile
selects it as the preview tile.

**Preview:**
1. Click "Run preview" → single-tile Cellpose-SAM call, returns the label
   mask for that one image.
2. Canvas shows the raw image with a per-object color overlay. Colors are
   picked deterministically from the object id (golden-angle hue rotation) so
   the same cell keeps the same color across opacity changes and re-renders.
3. Opacity slider (0–100%) controls overlay alpha. Slider sits above the
   canvas; default around 50%.

Mask pixel encoding matches the existing `/api/mask/<i>` contract (R = id &
0xFF, G = (id >> 8) & 0xFF, B = 0), so the client-side coloring code can be
shared with Step 3's similarity mode.

**Action:** "Continue → Cell selection" kicks off the full pipeline
(`POST /api/segment_and_embed`) and advances to Step 3. Step 3 shows a
progress indicator until the backend reports `complete`.

### Backend delta for Step 2

One new endpoint:

```
POST /api/segment_preview
  body: { image_index: int }
  returns: { image_index, width, height, num_cells, mask_base64 }
```

Runs `segment_cells(state.images[image_index])` and PNG-encodes the label
mask using the same R/G id encoding as `/api/mask`. Does **not** mutate
persistent state — so re-running the preview on other tiles is cheap and
the later full pipeline run is unaffected.

---

## 5. Step 3 — Cell Selection

**Always-visible gallery** at the top of the step: thumbnails of every cell
currently in the selection (`selected` set). Same card style as
`StepReview`. This is the "nice way to show what you've selected" — it
persists as you switch between modes so you never lose sight of your picks.

**Mode toggle** (segmented control or dropdown):

### 5a. Similarity — smart select ✓ wired

The existing select → review → search flow, collapsed into one scrollable
panel:

- Left: the canvas picker from `StepSelect` (click cells on an image, tile
  browser on the side).
- Right/top: the persistent gallery (see above) doubles as the "selected
  cells" view — no separate review step.
- Below: "Find similar" runs `POST /api/search` + `POST /api/search_dissimilar`.
  Results render as two gallery strips (most similar / most dissimilar) with
  accept/reject buttons and the rejection-strength + min-similarity sliders
  from the current `StepSearch`.

No backend work needed; this is the currently-working path reorganized.

### 5b. Classic — feature thresholds ✗ stub

UI skeleton only. Shows a panel with placeholder rows for features we
_intend_ to expose (area, perimeter, circularity, solidity, mean intensity
per channel, …) each with a dual-handle range slider, and a banner reading
"Feature extraction endpoint not implemented — see backend delta below".

This makes the intended UX visible without faking data.

### 5c. Clustering — Leiden on KNN cosine graph ✗ stub

UI skeleton only. Shows:

- Feature source radio: "Classic features" / "DINOv2 deep features"
  (DINOv2 enabled since embeddings exist; Classic disabled until 5b lands).
- KNN `k` input, Leiden resolution input.
- "Compute" button (disabled) + placeholder for a cluster list where each
  cluster entry becomes clickable to add its members to `selected`.
- Banner: "KNN graph + Leiden endpoint not implemented — see backend delta".

### Backend deltas for Step 3 (5b + 5c — not in scope for this PR)

Tracked here so we don't forget:

```
POST /api/features/classic
  body: { }   # implicit: current dataset
  returns: { features: [{object_id, area, perimeter, circularity,
                         solidity, mean_intensity_ch1..3, ...}] }

POST /api/features/threshold
  body: { filters: [{name, min, max}] }
  returns: { object_ids: [...] }

POST /api/cluster
  body: { source: "dinov2" | "classic", k: int, resolution: float }
  returns: { clusters: [{cluster_id, object_ids: [...]}] }
```

Implementation notes:

- Classic features: `scikit-image` `regionprops` over `state.cell_masks` +
  per-channel intensity stats over the raw images. Cache on `DatasetState`
  alongside `global_embeddings`.
- KNN graph: FAISS already gives us cosine neighbors on the normalized
  embeddings — reuse `state.faiss_index` for `k`-NN retrieval, then build an
  igraph `Graph` from the `(src, dst, weight=sim)` triples.
- Leiden: `leidenalg.find_partition(graph, RBConfigurationVertexPartition,
  resolution_parameter=…)`. Add `leidenalg` + `python-igraph` to
  `backend/requirements.txt` when we get there.
- All three endpoints are pure reads after the main pipeline has run; they
  don't touch Cellpose / DINOv2 GPU state.

---

## 6. App shell changes

- Header pills become three: `1. Dataset`, `2. Segmentation`, `3. Selection`.
  Click-to-navigate rules stay the same (only visit current or earlier).
- App state additions:
  - `segMethod` (currently always `"cellpose-sam"`)
  - `previewImgIdx`, `previewMaskData`, `maskAlpha`
  - `selMode: "similarity" | "classic" | "clustering"`
  - `pipelineStatus: "idle" | "running" | "done" | "error"` — so Step 3 can
    gate the selection UI on pipeline completion.
- API client gets one new method: `api.segmentPreview(imageIndex)`.

Styles: add `S.select` and a small `S.segmented` for the mode toggle. Reuse
the existing card / button / slider styles everywhere else.

---

## 7. What this PR will ship vs. defer

**Will ship on this branch:**
- Frontend rewrite to 3 steps with method dropdowns.
- New backend endpoint `POST /api/segment_preview`.
- Step 2 single-tile preview with colored overlay + opacity slider.
- Step 3 similarity mode wired end-to-end (same functionality as today,
  reorganized).
- Step 3 persistent selection gallery.
- Step 3 classic + clustering modes as honest UI stubs with "backend not
  implemented" banners and the intended controls laid out.

**Deferred (separate PRs, tracked above):**
- Classic feature extraction endpoint + feature-threshold filter.
- KNN graph + Leiden clustering endpoint.
- `smart-analysis` integration for Cellpose 3 (and any other external
  segmenters).
- Multi-dataset registry endpoint (`GET /api/datasets`).

---

## 8. Design & quality bar

The end result should look and feel like a simple, clean, professional,
production-ready tool — not a demo, not a research notebook. Concretely:

- **Same tech stack.** Stay on the current stack — one React component in
  `frontend/MicroscopySearch.jsx` with inline CSS-in-JS in the `S` object,
  Flask in `backend/app.py`. No new frameworks, no CSS framework (Tailwind
  etc.), no component library (MUI, Radix, shadcn, etc.), no state library
  (keep `useAppState`), no new build tooling.
- **Visual restraint.** Reuse the existing palette, Inter font, and card /
  button / slider styles from the `S` style object in
  `frontend/MicroscopySearch.jsx` (lines 539–693). Add only what's needed
  (`S.select`, a small `S.segmented` for the mode toggle). No gradients, no
  decorative icons, no emoji.
- **Consistent layout.** Every step is one card (`S.card`), same padding,
  same heading (`S.h2`) + subcopy (`S.muted`) pattern. The per-step method
  dropdown sits at the top of each card, immediately below the subcopy, so
  the "pick a method" affordance is the same muscle memory across all
  three steps.
- **Honest states.** Loading, empty, error, and "backend not wired" states
  are all first-class. Disabled dropdown options render with a short hint
  next to them (e.g. "needs separate env"). Stub modes render their full
  intended UI greyed out with a single banner explaining what's missing —
  never fake data.
- **Keyboard & focus.** Native `<select>`, `<input>`, `<button>` — no
  custom combobox. Visible focus rings. Tab order follows reading order.
- **Responsiveness within reason.** The canvas + gallery + sidebar layout
  should hold together down to ~1280px wide. Below that, stack vertically
  rather than squashing. No mobile design needed.
- **Performance.** The mask-overlay pass already scans every pixel in JS
  (`MicroscopySearch.jsx:235–246`); keep that pattern but precompute the
  per-id color table once per mask load, and cache the offscreen color
  canvas so the opacity slider doesn't re-color on every frame — only the
  `globalAlpha` changes.
- **Copy.** Labels and subcopy are terse and precise. "Cellpose-SAM", not
  "🧬 AI-Powered Cell Segmentation". "Find similar", not "Discover more
  cells like these!". Buttons are verbs.
- **No dead code, no TODO comments.** Stubs render real UI and call real
  (if disabled) controls; the "not implemented" story lives in the banner
  text, not in `// TODO` scatter.
- **Tests don't regress.** `backend/tests.py` (30 tests) and
  `frontend/tests_frontend_logic.py` (35 tests) should still pass. New
  backend endpoint gets at least one happy-path and one error-path test.

---

## 9. Open questions before I start coding

1. Color palette for the mask overlay — golden-angle HSL works well; OK to
   just pick sensible defaults, or do you want a specific palette?
2. Step 2 "Run preview" default — should I auto-run preview on the first
   tile when entering Step 2, or require an explicit button click?
3. Step 3 similarity mode — keep three sub-phases (pick → review → search)
   or flatten into one scrollable panel with the persistent gallery doing
   double duty as "review"?
4. Where does "Continue → full pipeline" live — a dedicated button at the
   bottom of Step 2, or tucked into the Step 3 header as "Ready to select"
   gating?
