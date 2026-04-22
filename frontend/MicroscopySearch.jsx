import { useState, useCallback, useEffect, useMemo, useRef } from "react";

const API = "http://localhost:5050/api";

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  upload: (dir, n) =>
    apiFetch("/upload_bbbc021", { method: "POST", body: JSON.stringify({ image_dir: dir, n }) }),
  process: (cropMode, sizeInvariant, rotationInvariant) =>
    apiFetch("/segment_and_embed", {
      method: "POST",
      body: JSON.stringify({ crop_mode: cropMode, size_invariant: sizeInvariant, rotation_invariant: rotationInvariant }),
    }),
  status: () => apiFetch("/status"),
  image: (i) => apiFetch(`/image/${i}`),
  objects: (i) => apiFetch(`/objects?image_index=${i}`),
  mask: (i) => apiFetch(`/mask/${i}`),
  segmentPreview: (i) =>
    apiFetch("/segment_preview", { method: "POST", body: JSON.stringify({ image_index: i }) }),
  crops: (ids) =>
    apiFetch("/crops", { method: "POST", body: JSON.stringify({ object_ids: ids }) }),
  search: (pos, neg, k, alpha) =>
    apiFetch("/search", {
      method: "POST",
      body: JSON.stringify({ positive_ids: pos, negative_ids: neg, top_k: k, negative_alpha: alpha }),
    }),
  searchDissimilar: (pos, k) =>
    apiFetch("/search_dissimilar", {
      method: "POST",
      body: JSON.stringify({ positive_ids: pos, top_k: k }),
    }),
};

/* ─── Constants ─────────────────────────────────────────────────────────── */

const DATASET_PRESETS = {
  bbbc021_week1: {
    label: "BBBC021 Week 1",
    path: "/Users/thomdehoog/Library/CloudStorage/Dropbox/Projects/smart-selection/microscopy-search/bbbc021_raw/Week1_22123/",
  },
};

const SEG_METHODS = [
  { id: "cellpose-sam", label: "Cellpose-SAM", enabled: true, hint: "" },
  { id: "cellpose3", label: "Cellpose 3", enabled: false, hint: "needs separate env" },
  { id: "smart-analysis", label: "smart-analysis", enabled: false, hint: "integration pending" },
];

const SEL_MODES = [
  { id: "similarity", label: "Similarity — smart select" },
  { id: "classic", label: "Classic — feature thresholds" },
  { id: "clustering", label: "Clustering — Leiden" },
];

/* ─── Color helpers ─────────────────────────────────────────────────────── */
// Golden-angle HSL (saturation 0.7, lightness 0.55). Deterministic per object id.
const GOLDEN_ANGLE = 137.508;
function idToRGB(id) {
  const h = (((id * GOLDEN_ANGLE) % 360) + 360) % 360 / 360;
  const s = 0.7, l = 0.55;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0, g = 0, b = 0;
  if (hp < 1) { r = c; g = x; }
  else if (hp < 2) { r = x; g = c; }
  else if (hp < 3) { g = c; b = x; }
  else if (hp < 4) { g = x; b = c; }
  else if (hp < 5) { r = x; b = c; }
  else { r = c; b = x; }
  const m = l - c / 2;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

/* ─── Global app state ──────────────────────────────────────────────────── */

export function useAppState() {
  const [s, setS] = useState({
    step: 1,
    // Step 1 — dataset
    datasetId: null, numImages: 0, channelNames: [],
    datasetSource: "bbbc021_week1",
    datasetPath: DATASET_PRESETS.bbbc021_week1.path,
    n: 5, cropMode: "single_cell", sizeInvariant: true, rotationInvariant: true,
    // Step 2 — segmentation
    segMethod: "cellpose-sam",
    previewImgIdx: 0, previewMaskData: null, previewNumCells: 0,
    maskAlpha: 0.5,
    // Pipeline
    pipelineStatus: "idle", progress: 0, msg: "", error: null,
    // Step 3 — selection
    selMode: "similarity",
    imgIdx: 0, imageData: null, objects: [],
    selected: new Set(), positive: [], negative: [],
    results: [], dissimilar: [],
    alpha: 0.4, threshold: 0.0,
  });
  const set = useCallback((p) => setS((prev) => ({ ...prev, ...p })), []);
  const toggle = useCallback((id) => {
    setS((prev) => {
      const next = new Set(prev.selected);
      next.has(id) ? next.delete(id) : next.add(id);
      return { ...prev, selected: next };
    });
  }, []);
  return { s, set, toggle };
}

/* ─── Step 1: Dataset ───────────────────────────────────────────────────── */

function StepDataset({ s, set }) {
  const [loading, setLoading] = useState(false);

  const onSourceChange = (v) => {
    const preset = DATASET_PRESETS[v];
    set({ datasetSource: v, datasetPath: preset ? preset.path : "" });
  };

  const load = async () => {
    setLoading(true); set({ error: null });
    try {
      const r = await api.upload(s.datasetPath, s.n);
      if (r.error) { set({ error: r.error }); setLoading(false); return; }
      set({
        datasetId: r.dataset_id,
        numImages: r.num_images,
        channelNames: r.channel_names,
        previewImgIdx: 0,
        previewMaskData: null,
        previewNumCells: 0,
        pipelineStatus: "idle",
        progress: 0,
        msg: "",
        step: 2,
      });
    } catch (e) {
      set({ error: e.message });
    }
    setLoading(false);
  };

  const isPreset = s.datasetSource in DATASET_PRESETS;

  return (
    <section style={S.card}>
      <h2 style={S.h2}>Dataset</h2>
      <p style={S.muted}>Pick a preset plate or point to a local directory.</p>

      <div style={S.formGroup}>
        <label style={S.label}>Dataset</label>
        <select style={S.select} value={s.datasetSource} onChange={e => onSourceChange(e.target.value)}>
          <option value="bbbc021_week1">BBBC021 Week 1</option>
          <option value="custom">Custom path…</option>
        </select>
      </div>

      <div style={{ ...S.formGroup, marginTop: 16 }}>
        <label style={S.label}>{isPreset ? "Path (preset)" : "Image directory"}</label>
        <input
          style={S.input}
          value={s.datasetPath}
          onChange={e => set({ datasetPath: e.target.value })}
          readOnly={isPreset}
          placeholder="/path/to/bbbc021_raw/Week1_22123/"
        />
      </div>

      <div style={{ display: "flex", gap: 24, marginTop: 20, flexWrap: "wrap" }}>
        <div style={S.formGroup}>
          <label style={S.label}>Fields of view</label>
          <input style={{ ...S.input, width: 100 }} type="number" min={1} max={100}
            value={s.n} onChange={e => set({ n: +e.target.value || 5 })} />
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Crop mode</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="cropMode" checked={s.cropMode === "single_cell"}
                onChange={() => set({ cropMode: "single_cell" })} />
              Single cell
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="cropMode" checked={s.cropMode === "neighborhood"}
                onChange={() => set({ cropMode: "neighborhood" })} />
              Neighborhood
            </label>
          </div>
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Size</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="sizeMode" checked={s.sizeInvariant}
                onChange={() => set({ sizeInvariant: true })} />
              Invariant
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="sizeMode" checked={!s.sizeInvariant}
                onChange={() => set({ sizeInvariant: false })} />
              Aware
            </label>
          </div>
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Rotation</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="rotMode" checked={s.rotationInvariant}
                onChange={() => set({ rotationInvariant: true })} />
              Invariant
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="rotMode" checked={!s.rotationInvariant}
                onChange={() => set({ rotationInvariant: false })} />
              Aware
            </label>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        <button style={S.btnPrimary} onClick={load} disabled={loading || !s.datasetPath.trim()}>
          {loading ? "Loading…" : "Load dataset"}
        </button>
      </div>

      {s.error && <p style={S.errorText}>{s.error}</p>}
    </section>
  );
}

/* ─── Step 2: Segmentation ──────────────────────────────────────────────── */

function StepSegmentation({ s, set }) {
  const canvasRef = useRef(null);
  const rawImgRef = useRef(null);
  const colorCvRef = useRef(null);       // Offscreen RGBA canvas with colored mask
  const [rawImg, setRawImg] = useState(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const pollRef = useRef(null);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // Load the raw image for the currently-selected preview tile
  useEffect(() => {
    let cancelled = false;
    setRawImg(null);
    api.image(s.previewImgIdx).then(d => {
      if (cancelled) return;
      setRawImg(d);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [s.previewImgIdx]);

  // When raw image arrives, draw it on the canvas.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !rawImg) return;
    const img = new window.Image();
    img.onload = () => {
      rawImgRef.current = img;
      cv.width = img.width; cv.height = img.height;
      redraw();
    };
    img.src = rawImg.thumbnail_base64;
  }, [rawImg]);

  // Build the colored-mask offscreen canvas once per mask change
  useEffect(() => {
    colorCvRef.current = null;
    if (!s.previewMaskData) { redraw(); return; }
    const mImg = new window.Image();
    mImg.onload = () => {
      const w = mImg.width, h = mImg.height;
      const mCv = document.createElement("canvas");
      mCv.width = w; mCv.height = h;
      const mCtx = mCv.getContext("2d");
      mCtx.drawImage(mImg, 0, 0);
      const mData = mCtx.getImageData(0, 0, w, h).data;

      const out = document.createElement("canvas");
      out.width = w; out.height = h;
      const outCtx = out.getContext("2d");
      const outImg = outCtx.createImageData(w, h);
      const outData = outImg.data;
      const colorCache = new Map();

      for (let p = 0; p < w * h; p++) {
        const idx = p * 4;
        const id = mData[idx] + mData[idx + 1] * 256;
        if (id === 0) continue;
        let rgb = colorCache.get(id);
        if (!rgb) { rgb = idToRGB(id); colorCache.set(id, rgb); }
        outData[idx] = rgb[0];
        outData[idx + 1] = rgb[1];
        outData[idx + 2] = rgb[2];
        outData[idx + 3] = 255;
      }
      outCtx.putImageData(outImg, 0, 0);
      colorCvRef.current = out;
      redraw();
    };
    mImg.src = s.previewMaskData.mask_base64;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s.previewMaskData]);

  // Redraw on opacity change (no recolor — just globalAlpha)
  useEffect(() => { redraw(); /* eslint-disable-next-line */ }, [s.maskAlpha]);

  const redraw = () => {
    const cv = canvasRef.current;
    if (!cv || !rawImgRef.current) return;
    const ctx = cv.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.drawImage(rawImgRef.current, 0, 0, cv.width, cv.height);
    if (colorCvRef.current) {
      ctx.globalAlpha = s.maskAlpha;
      ctx.drawImage(colorCvRef.current, 0, 0, cv.width, cv.height);
      ctx.globalAlpha = 1;
    }
  };

  const runPreview = async () => {
    setLoadingPreview(true); set({ error: null });
    try {
      const d = await api.segmentPreview(s.previewImgIdx);
      if (d.error) set({ error: d.error });
      else set({ previewMaskData: d, previewNumCells: d.num_cells });
    } catch (e) {
      set({ error: e.message });
    }
    setLoadingPreview(false);
  };

  const continueToSelection = async () => {
    set({ pipelineStatus: "running", progress: 0, msg: "Starting pipeline…", error: null, step: 3 });
    try {
      await api.process(s.cropMode, s.sizeInvariant, s.rotationInvariant);
      pollRef.current = setInterval(async () => {
        try {
          const st = await api.status();
          set({ progress: st.progress || 0, msg: st.message || "" });
          if (st.status === "complete") {
            clearInterval(pollRef.current);
            const img = await api.image(0);
            const obj = await api.objects(0);
            set({ pipelineStatus: "done", imageData: img, objects: obj.objects || [], imgIdx: 0 });
          } else if (st.status === "error") {
            clearInterval(pollRef.current);
            set({ pipelineStatus: "error", error: st.error });
          }
        } catch (e) {
          clearInterval(pollRef.current);
          set({ pipelineStatus: "error", error: e.message });
        }
      }, 1500);
    } catch (e) {
      set({ pipelineStatus: "error", error: e.message });
    }
  };

  return (
    <section style={S.card}>
      <h2 style={S.h2}>Segmentation</h2>
      <p style={S.muted}>Preview the segmentation on a single tile before running the full pipeline.</p>

      <div style={S.formGroup}>
        <label style={S.label}>Method</label>
        <select style={S.select} value={s.segMethod}
          onChange={e => set({ segMethod: e.target.value })}>
          {SEG_METHODS.map(m => (
            <option key={m.id} value={m.id} disabled={!m.enabled}>
              {m.label}{m.hint ? ` — ${m.hint}` : ""}
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: "flex", gap: 20, marginTop: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
          {Array.from({ length: s.numImages }, (_, i) => (
            <button key={i} onClick={() => set({ previewImgIdx: i, previewMaskData: null })}
              style={i === s.previewImgIdx ? { ...S.tileBtn, ...S.tileBtnActive } : S.tileBtn}>
              {i + 1}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={S.canvasFrame}>
            {rawImg ? (
              <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
            ) : (
              <div style={S.canvasPlaceholder}>Loading tile…</div>
            )}
          </div>
        </div>

        <aside style={S.sidebar}>
          <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4, color: "#1A1A1A" }}>
            Preview
          </div>
          <div style={{ ...S.caption, marginBottom: 16 }}>
            {s.previewMaskData
              ? `Tile ${s.previewImgIdx + 1} · ${s.previewNumCells} cells`
              : `Tile ${s.previewImgIdx + 1} · no preview yet`}
          </div>

          <label style={S.controlLabel}>Overlay opacity · {Math.round(s.maskAlpha * 100)}%</label>
          <input type="range" min={0} max={1} step={0.05} value={s.maskAlpha}
            onChange={e => set({ maskAlpha: +e.target.value })}
            style={{ ...S.slider, width: "100%", marginTop: 6 }}
            disabled={!s.previewMaskData} />

          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
            <button style={S.btnSecondary} onClick={runPreview}
              disabled={loadingPreview || s.pipelineStatus === "running"}>
              {loadingPreview ? "Running…" : "Run preview"}
            </button>
          </div>

          <div style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid #E5E7EB" }}>
            <button style={{ ...S.btnPrimary, width: "100%" }}
              onClick={continueToSelection}
              disabled={s.pipelineStatus === "running"}>
              Continue → Cell selection
            </button>
            <p style={{ ...S.caption, marginTop: 10 }}>
              Runs the full segmentation + embedding pipeline on all {s.numImages} tiles.
            </p>
          </div>
        </aside>
      </div>

      {s.error && <p style={S.errorText}>{s.error}</p>}
    </section>
  );
}

/* ─── Step 3: Cell Selection ────────────────────────────────────────────── */

function SelectionGallery({ s, set }) {
  const [thumbs, setThumbs] = useState([]);
  useEffect(() => {
    const ids = [...s.selected];
    if (ids.length === 0) { setThumbs([]); return; }
    api.crops(ids).then(d => setThumbs(d.crops || [])).catch(() => {});
  }, [s.selected]);

  const remove = (id) => {
    const ns = new Set(s.selected); ns.delete(id);
    set({
      selected: ns,
      positive: s.positive.filter(x => x !== id),
    });
  };

  return (
    <div style={S.gallery}>
      <div style={S.galleryHeader}>
        <span style={{ fontWeight: 600, color: "#111" }}>Selected cells</span>
        <span style={S.caption}>{s.selected.size} cell{s.selected.size === 1 ? "" : "s"}</span>
        {s.selected.size > 0 && (
          <button style={S.linkBtn} onClick={() => set({ selected: new Set(), positive: [] })}>
            Clear all
          </button>
        )}
      </div>
      {s.selected.size === 0 ? (
        <p style={S.emptyMsg}>No cells selected yet. Pick a mode below to start.</p>
      ) : (
        <div style={S.galleryGrid}>
          {thumbs.map(t => (
            <div key={t.object_id} style={S.resultCard}>
              <div style={S.thumbContainer}>
                <img src={t.thumbnail_base64} alt="" style={S.thumbImg} />
              </div>
              <div style={S.cardFooter}>
                <span style={{ color: "#374151", fontWeight: 500 }}>#{t.object_id}</span>
                <button style={S.linkBtn} onClick={() => remove(t.object_id)}>Remove</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SimilarityPanel({ s, set, toggle }) {
  const canvasRef = useRef(null);
  const [maskData, setMaskData] = useState(null);
  const maskImgRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (s.imageData == null) return;
    api.mask(s.imgIdx).then(d => setMaskData(d)).catch(() => {});
  }, [s.imgIdx, s.imageData]);

  useEffect(() => {
    if (!maskData) return;
    const mImg = new window.Image();
    mImg.onload = () => { maskImgRef.current = mImg; };
    mImg.src = maskData.mask_base64;
  }, [maskData]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !s.imageData) return;
    const ctx = cv.getContext("2d");
    const img = new window.Image();
    img.onload = () => {
      cv.width = img.width; cv.height = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(0, 0, cv.width, cv.height);

      if (maskImgRef.current) {
        const w = s.imageData.width, h = s.imageData.height;
        const mCv = document.createElement("canvas");
        mCv.width = w; mCv.height = h;
        const mCtx = mCv.getContext("2d");
        mCtx.drawImage(maskImgRef.current, 0, 0);
        const mData = mCtx.getImageData(0, 0, w, h).data;
        const sx = cv.width / w, sy = cv.height / h;

        const brightCv = document.createElement("canvas");
        brightCv.width = cv.width; brightCv.height = cv.height;
        brightCv.getContext("2d").drawImage(img, 0, 0);

        const clipCv = document.createElement("canvas");
        clipCv.width = cv.width; clipCv.height = cv.height;
        const clipCtx = clipCv.getContext("2d");

        const highlight = new Set([...s.selected]);
        if (hovered && !highlight.has(hovered)) highlight.add(hovered);

        if (highlight.size > 0) {
          for (let my = 0; my < h; my++) {
            for (let mx = 0; mx < w; mx++) {
              const idx = (my * w + mx) * 4;
              const id = mData[idx] + mData[idx + 1] * 256;
              if (id > 0 && highlight.has(id)) {
                const cx = Math.floor(mx * sx), cy = Math.floor(my * sy);
                const cw = Math.max(Math.ceil(sx), 1), ch = Math.max(Math.ceil(sy), 1);
                clipCtx.fillStyle = hovered === id && !s.selected.has(id)
                  ? "rgba(200,220,255,0.7)" : "white";
                clipCtx.fillRect(cx, cy, cw, ch);
              }
            }
          }
          clipCtx.globalCompositeOperation = "source-in";
          clipCtx.drawImage(brightCv, 0, 0);
          ctx.drawImage(clipCv, 0, 0);
        }
      }
    };
    img.src = s.imageData.thumbnail_base64;
  }, [s.imageData, s.selected, hovered, maskData]);

  const hitTest = (e) => {
    const cv = canvasRef.current;
    if (!cv || !s.imageData || !maskImgRef.current) return null;
    const r = cv.getBoundingClientRect();
    const mx = Math.floor(((e.clientX - r.left) / r.width) * s.imageData.width);
    const my = Math.floor(((e.clientY - r.top) / r.height) * s.imageData.height);
    const mCv = document.createElement("canvas");
    mCv.width = s.imageData.width; mCv.height = s.imageData.height;
    const mCtx = mCv.getContext("2d");
    mCtx.drawImage(maskImgRef.current, 0, 0);
    const pixel = mCtx.getImageData(mx, my, 1, 1).data;
    const id = pixel[0] + pixel[1] * 256;
    return id > 0 ? id : null;
  };

  const switchTile = async (i) => {
    const [img, obj] = await Promise.all([api.image(i), api.objects(i)]);
    setMaskData(null);
    set({ imageData: img, objects: obj.objects || [], imgIdx: i });
  };

  const search = async () => {
    if (s.selected.size === 0) return;
    setBusy(true);
    const pos = [...s.selected];
    try {
      const [sim, dis] = await Promise.all([
        api.search(pos, s.negative, 50, s.alpha),
        api.searchDissimilar(pos, 28),
      ]);
      set({ positive: pos, results: sim.results || [], dissimilar: dis.results || [] });
    } catch (e) {
      set({ error: e.message });
    }
    setBusy(false);
  };

  const accept = (id) => {
    const ns = new Set(s.selected); ns.add(id);
    set({
      selected: ns,
      positive: [...s.positive, id],
      results: s.results.filter(r => r.object_id !== id),
      dissimilar: s.dissimilar.filter(r => r.object_id !== id),
    });
  };
  const reject = (id) => {
    set({
      negative: [...s.negative, id],
      results: s.results.filter(r => r.object_id !== id),
      dissimilar: s.dissimilar.filter(r => r.object_id !== id),
    });
  };

  const visible = s.results.filter(r => r.similarity_score >= s.threshold);
  const COLS = 5, MAX_ROWS = 4, LIMIT = COLS * MAX_ROWS;
  const mostSimilar = visible.slice(0, LIMIT);
  const mostDissimilar = s.dissimilar.slice(0, LIMIT);

  const ResultCard = ({ r }) => (
    <div style={S.resultCard}>
      <div style={S.thumbContainer}>
        <img src={r.thumbnail_base64} alt="" style={S.thumbImg} />
      </div>
      <div style={S.cardFooter}>
        <span>#{r.object_id}</span>
        <span style={{ fontWeight: 600, color: scoreColor(r.similarity_score) }}>
          {(r.similarity_score * 100).toFixed(0)}%
        </span>
      </div>
      <div style={{ display: "flex" }}>
        <button onClick={() => accept(r.object_id)} style={S.acceptBtn}>Similar</button>
        <button onClick={() => reject(r.object_id)} style={S.rejectBtn}>Dissimilar</button>
      </div>
    </div>
  );

  return (
    <div>
      <div style={{ display: "flex", gap: 20, marginTop: 8 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
          {Array.from({ length: s.numImages }, (_, i) => (
            <button key={i} onClick={() => switchTile(i)}
              style={i === s.imgIdx ? { ...S.tileBtn, ...S.tileBtnActive } : S.tileBtn}>
              {i + 1}
            </button>
          ))}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={S.canvasFrame}>
            {s.imageData ? (
              <canvas ref={canvasRef}
                style={{ width: "100%", display: "block", cursor: "crosshair" }}
                onClick={e => { const id = hitTest(e); if (id) toggle(id); }}
                onMouseMove={e => setHovered(hitTest(e))}
                onMouseLeave={() => setHovered(null)} />
            ) : (
              <div style={S.canvasPlaceholder}>Loading tile…</div>
            )}
          </div>
          <p style={{ ...S.caption, marginTop: 8 }}>
            {s.objects.length} cells on this tile · click to select
          </p>
        </div>
      </div>

      <div style={{ ...S.controlBar, marginTop: 20 }}>
        <div style={S.controlGroup}>
          <label style={S.controlLabel}>Rejection strength · {s.alpha.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.05} value={s.alpha}
            onChange={e => set({ alpha: +e.target.value })} style={S.slider} />
        </div>
        <div style={S.controlGroup}>
          <label style={S.controlLabel}>Min similarity · {s.threshold.toFixed(2)}</label>
          <input type="range" min={0} max={1} step={0.05} value={s.threshold}
            onChange={e => set({ threshold: +e.target.value })} style={S.slider} />
        </div>
        <button style={S.btnPrimary} onClick={search}
          disabled={busy || s.selected.size === 0}>
          {busy ? "Searching…" : (s.results.length ? "Recompute" : "Find similar")}
        </button>
        <span style={S.caption}>
          {s.positive.length} positive · {s.negative.length} rejected · {visible.length} matches
        </span>
      </div>

      {(s.results.length > 0 || s.dissimilar.length > 0) && (
        <>
          <div style={S.gallerySimilar}>
            <h3 style={S.galleryTitleGreen}>Most similar</h3>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: 10 }}>
              {mostSimilar.map(r => <ResultCard key={r.object_id} r={r} />)}
            </div>
            {mostSimilar.length === 0 && <p style={S.emptyMsg}>No results above threshold.</p>}
          </div>
          <div style={{ ...S.galleryDissimilar, marginTop: 16 }}>
            <h3 style={S.galleryTitleRed}>Most dissimilar</h3>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: 10 }}>
              {mostDissimilar.map(r => <ResultCard key={r.object_id} r={r} />)}
            </div>
            {mostDissimilar.length === 0 && <p style={S.emptyMsg}>No dissimilar results yet.</p>}
          </div>
        </>
      )}
    </div>
  );
}

const CLASSIC_FEATURES = [
  { key: "area", label: "Area (px)" },
  { key: "perimeter", label: "Perimeter (px)" },
  { key: "circularity", label: "Circularity" },
  { key: "solidity", label: "Solidity" },
  { key: "intensity_ch1", label: "Mean intensity · DAPI" },
  { key: "intensity_ch2", label: "Mean intensity · Tubulin" },
  { key: "intensity_ch3", label: "Mean intensity · Actin" },
];

function ClassicPanel() {
  return (
    <div>
      <div style={S.banner}>
        Feature extraction endpoint not implemented. The controls below show the
        intended UI; they will activate once <code>POST /api/features/classic</code>
        and <code>POST /api/features/threshold</code> land (WORKFLOW.md §5).
      </div>
      <div style={S.featureTable}>
        {CLASSIC_FEATURES.map(f => (
          <div key={f.key} style={S.featureRow}>
            <label style={S.featureLabel}>{f.label}</label>
            <div style={S.rangeInputs}>
              <input type="number" placeholder="min" disabled style={S.numInput} />
              <span style={S.rangeDash}>–</span>
              <input type="number" placeholder="max" disabled style={S.numInput} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 20 }}>
        <button style={S.btnPrimary} disabled>Apply filter</button>
      </div>
    </div>
  );
}

function ClusteringPanel() {
  return (
    <div>
      <div style={S.banner}>
        KNN graph + Leiden endpoint not implemented. The controls below show the
        intended UI; they will activate once <code>POST /api/cluster</code> lands
        (WORKFLOW.md §5).
      </div>
      <div style={S.formGroup}>
        <label style={S.label}>Feature source</label>
        <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
          <label style={{ ...S.radioLabel, opacity: 0.6 }}>
            <input type="radio" disabled /> Classic features
          </label>
          <label style={{ ...S.radioLabel, opacity: 0.6 }}>
            <input type="radio" disabled defaultChecked /> DINOv2 deep features
          </label>
        </div>
      </div>
      <div style={{ display: "flex", gap: 24, marginTop: 20, flexWrap: "wrap" }}>
        <div style={S.formGroup}>
          <label style={S.label}>Neighbors (k)</label>
          <input style={{ ...S.input, width: 120 }} type="number" disabled defaultValue={15} />
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Leiden resolution</label>
          <input style={{ ...S.input, width: 120 }} type="number" disabled
            defaultValue={1.0} step={0.1} />
        </div>
      </div>
      <div style={{ marginTop: 20 }}>
        <button style={S.btnPrimary} disabled>Compute clusters</button>
      </div>
      <div style={{ ...S.emptyMsg, marginTop: 24 }}>
        Cluster list will appear here. Click a cluster to add all its members
        to the selection.
      </div>
    </div>
  );
}

function StepSelection({ s, set, toggle }) {
  if (s.pipelineStatus !== "done") {
    return (
      <section style={S.card}>
        <h2 style={S.h2}>Cell Selection</h2>
        <p style={S.muted}>
          {s.pipelineStatus === "running"
            ? "Running segmentation and embedding on all tiles."
            : s.pipelineStatus === "error"
            ? "Pipeline failed. Go back and try again."
            : "Finish segmentation before picking cells."}
        </p>
        {s.pipelineStatus === "running" && (
          <div style={{ marginTop: 12 }}>
            <div style={S.progressTrack}>
              <div style={{ ...S.progressFill, width: `${(s.progress || 0) * 100}%` }} />
            </div>
            <p style={{ ...S.caption, marginTop: 8 }}>{s.msg}</p>
          </div>
        )}
        {s.error && <p style={S.errorText}>{s.error}</p>}
        <div style={{ marginTop: 20 }}>
          <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>Back to Segmentation</button>
        </div>
      </section>
    );
  }

  const mode = SEL_MODES.find(m => m.id === s.selMode) || SEL_MODES[0];
  return (
    <section style={S.card}>
      <h2 style={S.h2}>Cell Selection</h2>
      <p style={S.muted}>{mode.label}. Pick cells below; your selection is shown at the top.</p>

      <SelectionGallery s={s} set={set} />

      <div style={{ ...S.formGroup, marginTop: 20 }}>
        <label style={S.label}>Mode</label>
        <select style={S.select} value={s.selMode}
          onChange={e => set({ selMode: e.target.value })}>
          {SEL_MODES.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </div>

      <div style={{ marginTop: 20 }}>
        {s.selMode === "similarity" && <SimilarityPanel s={s} set={set} toggle={toggle} />}
        {s.selMode === "classic" && <ClassicPanel />}
        {s.selMode === "clustering" && <ClusteringPanel />}
      </div>
    </section>
  );
}

function scoreColor(v) {
  if (v > 0.8) return "#16A34A";
  if (v > 0.5) return "#CA8A04";
  return "#DC2626";
}

/* ─── App shell ─────────────────────────────────────────────────────────── */

export default function App() {
  const { s, set, toggle } = useAppState();
  const steps = ["Dataset", "Segmentation", "Selection"];
  const canNav = (target) => {
    if (target === 1) return true;
    if (target === 2) return s.datasetId != null;
    if (target === 3) return s.pipelineStatus === "done" || s.pipelineStatus === "running" || s.pipelineStatus === "error";
    return false;
  };
  return (
    <div style={S.root}>
      <header style={S.header}>
        <span style={S.logo}>Smart Selection</span>
        <nav style={{ display: "flex", gap: 4 }}>
          {steps.map((label, i) => {
            const idx = i + 1;
            const reachable = canNav(idx);
            return (
              <span key={idx}
                onClick={() => { if (reachable) set({ step: idx }); }}
                style={{
                  ...S.stepPill,
                  ...(s.step === idx ? S.stepPillActive : {}),
                  ...(s.step > idx ? S.stepPillDone : {}),
                  cursor: reachable ? "pointer" : "default",
                  opacity: reachable ? 1 : 0.5,
                }}>
                {idx}. {label}
              </span>
            );
          })}
        </nav>
      </header>
      <main style={S.main}>
        {s.step === 1 && <StepDataset s={s} set={set} />}
        {s.step === 2 && <StepSegmentation s={s} set={set} />}
        {s.step === 3 && <StepSelection s={s} set={set} toggle={toggle} />}
      </main>
    </div>
  );
}

/* ─── Styles ────────────────────────────────────────────────────────────── */

const S = {
  root: {
    minHeight: "100vh", background: "#F4F5F7",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#1A1A1A", fontSize: 15,
  },
  header: {
    background: "#fff", borderBottom: "1px solid #E2E4E9",
    padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center",
    boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
  },
  logo: { fontWeight: 700, fontSize: 20, letterSpacing: "-0.03em", color: "#111" },
  main: { maxWidth: 1400, margin: "32px auto", padding: "0 32px" },

  stepPill: {
    padding: "8px 16px", fontSize: 14, color: "#9CA3AF", borderRadius: 8,
    fontWeight: 500, transition: "all 0.15s",
  },
  stepPillActive: { background: "#EFF6FF", color: "#2563EB", fontWeight: 600 },
  stepPillDone: { color: "#16A34A" },

  card: {
    background: "#fff", border: "1px solid #E2E4E9", borderRadius: 16,
    padding: 32, boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
  },

  h2: { fontSize: 22, fontWeight: 700, margin: "0 0 6px", letterSpacing: "-0.02em", color: "#111" },
  muted: { color: "#6B7280", fontSize: 15, margin: "0 0 16px", lineHeight: 1.6 },
  caption: { fontSize: 14, color: "#6B7280", margin: 0 },

  formGroup: { display: "flex", flexDirection: "column" },
  label: { display: "block", fontSize: 14, fontWeight: 600, color: "#374151", marginBottom: 6 },
  input: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "10px 14px", fontSize: 15,
    width: "100%", maxWidth: 560, outline: "none", boxSizing: "border-box",
    transition: "border-color 0.15s", fontFamily: "inherit",
  },
  select: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "10px 14px", fontSize: 15,
    width: "100%", maxWidth: 360, outline: "none", boxSizing: "border-box",
    background: "#fff", color: "#1A1A1A", fontFamily: "inherit", cursor: "pointer",
  },
  radioLabel: {
    fontSize: 15, cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
    color: "#374151",
  },
  numInput: {
    border: "1px solid #D1D5DB", borderRadius: 6, padding: "6px 10px", fontSize: 14,
    width: 80, outline: "none", boxSizing: "border-box", background: "#F9FAFB",
    color: "#6B7280", fontFamily: "inherit",
  },

  btnPrimary: {
    background: "#2563EB", color: "#fff", border: "none", borderRadius: 8,
    padding: "10px 24px", fontSize: 15, fontWeight: 600, cursor: "pointer",
    transition: "background 0.15s", letterSpacing: "-0.01em",
  },
  btnSecondary: {
    background: "#fff", color: "#374151", border: "1px solid #D1D5DB", borderRadius: 8,
    padding: "10px 20px", fontSize: 14, fontWeight: 500, cursor: "pointer",
    transition: "background 0.15s",
  },
  linkBtn: {
    background: "none", border: "none", color: "#DC2626", fontSize: 13,
    cursor: "pointer", padding: 0, fontWeight: 500,
  },

  progressTrack: { height: 8, background: "#E5E7EB", borderRadius: 4, overflow: "hidden" },
  progressFill: { height: "100%", background: "#2563EB", borderRadius: 4, transition: "width 0.4s ease" },
  errorText: { color: "#DC2626", fontSize: 14, marginTop: 16, fontWeight: 500 },

  tileBtn: {
    width: 48, height: 48, border: "1px solid #D1D5DB", background: "#fff",
    borderRadius: 10, cursor: "pointer", fontSize: 16, fontWeight: 700,
    color: "#6B7280", display: "flex", alignItems: "center", justifyContent: "center",
    transition: "all 0.15s",
  },
  tileBtnActive: {
    border: "2px solid #2563EB", background: "#EFF6FF", color: "#2563EB",
  },

  canvasFrame: {
    border: "1px solid #E2E4E9", borderRadius: 12, overflow: "hidden", background: "#111",
    minHeight: 200,
  },
  canvasPlaceholder: {
    padding: "80px 0", textAlign: "center", color: "#9CA3AF", fontSize: 14,
  },

  sidebar: {
    width: 260, flexShrink: 0, background: "#FAFBFC", border: "1px solid #E2E4E9",
    borderRadius: 12, padding: 16, display: "flex", flexDirection: "column",
  },

  gallery: {
    background: "#FAFBFC", border: "1px solid #E2E4E9", borderRadius: 12,
    padding: 16, marginTop: 4,
  },
  galleryHeader: {
    display: "flex", alignItems: "center", gap: 12, marginBottom: 12,
  },
  galleryGrid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10,
  },

  resultCard: {
    border: "1px solid #E2E4E9", borderRadius: 10, overflow: "hidden", background: "#fff",
    transition: "box-shadow 0.15s",
  },
  thumbContainer: {
    width: "100%", aspectRatio: "1", background: "#0A0A0A",
    display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
    borderRadius: "9px 9px 0 0",
  },
  thumbImg: { maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" },
  cardFooter: {
    padding: "10px 12px", display: "flex", justifyContent: "space-between",
    alignItems: "center", fontSize: 14, color: "#6B7280",
  },
  acceptBtn: {
    flex: 1, padding: 10, border: "none", borderTop: "1px solid #E2E4E9",
    background: "#F0FDF4", color: "#16A34A", fontSize: 14, fontWeight: 600,
    cursor: "pointer", transition: "background 0.15s",
  },
  rejectBtn: {
    flex: 1, padding: 10, border: "none", borderTop: "1px solid #E2E4E9",
    borderLeft: "1px solid #E2E4E9", background: "#FEF2F2", color: "#DC2626",
    fontSize: 14, fontWeight: 600, cursor: "pointer", transition: "background 0.15s",
  },

  controlBar: {
    display: "flex", gap: 24, alignItems: "flex-end", padding: "16px 20px",
    background: "#FAFBFC", borderRadius: 12, border: "1px solid #E2E4E9",
    flexWrap: "wrap", marginBottom: 20,
  },
  controlGroup: { display: "flex", flexDirection: "column", gap: 6 },
  controlLabel: { fontSize: 13, color: "#6B7280", fontWeight: 500 },
  slider: { width: 180, accentColor: "#2563EB" },

  gallerySimilar: {
    padding: 20, background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: 12,
  },
  galleryDissimilar: {
    padding: 20, background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 12,
  },
  galleryTitleGreen: {
    fontSize: 16, fontWeight: 700, margin: "0 0 14px", color: "#16A34A", letterSpacing: "-0.01em",
  },
  galleryTitleRed: {
    fontSize: 16, fontWeight: 700, margin: "0 0 14px", color: "#DC2626", letterSpacing: "-0.01em",
  },
  emptyMsg: {
    textAlign: "center", color: "#9CA3AF", padding: "24px 0", fontSize: 14, margin: 0,
  },

  banner: {
    background: "#FFFBEB", border: "1px solid #FDE68A", color: "#78350F",
    borderRadius: 10, padding: "12px 16px", fontSize: 14, lineHeight: 1.55,
    marginBottom: 20,
  },

  featureTable: { display: "flex", flexDirection: "column", gap: 10 },
  featureRow: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "10px 14px", background: "#FAFBFC", border: "1px solid #E2E4E9",
    borderRadius: 8,
  },
  featureLabel: { fontSize: 14, color: "#374151", fontWeight: 500 },
  rangeInputs: { display: "flex", alignItems: "center", gap: 8 },
  rangeDash: { color: "#9CA3AF", fontSize: 14 },
};

