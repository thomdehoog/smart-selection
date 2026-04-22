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
export function idToRGB(id) {
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

/* ─── Shared components ─────────────────────────────────────────────────── */

function SectionBar({ title, subtitle, accent = "#374151", count, actions, children }) {
  return (
    <section style={{ ...S.section, borderLeftColor: accent }}>
      <header style={S.sectionHeader}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10, minWidth: 0 }}>
          <h2 style={{ ...S.sectionTitle, color: accent }}>{title}</h2>
          {count != null && <span style={S.sectionCount}>{count}</span>}
        </div>
        {actions && <div style={S.sectionActions}>{actions}</div>}
      </header>
      <div style={S.sectionBody}>
        {subtitle && <p style={S.muted}>{subtitle}</p>}
        {children}
      </div>
    </section>
  );
}

function ToggleGroup({ value, onChange, options, disabled = false }) {
  return (
    <div style={S.toggleGroup} role="radiogroup">
      {options.map(o => {
        const active = value === o.value;
        return (
          <button key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled}
            onClick={() => onChange(o.value)}
            style={active ? { ...S.toggleBtn, ...S.toggleBtnActive } : S.toggleBtn}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function FormField({ label, children }) {
  return (
    <div>
      <div style={S.formSectionTitle}>{label}</div>
      {children}
    </div>
  );
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
    <SectionBar title="Dataset" subtitle="Pick a preset plate or point to a local directory."
      actions={
        <button style={S.btnPrimary} onClick={load}
          disabled={loading || !s.datasetPath.trim()}>
          {loading ? "Loading…" : "Load dataset"}
        </button>
      }>
      <div style={S.formStack}>
        <FormField label="Dataset source">
          <select style={S.select} value={s.datasetSource}
            onChange={e => onSourceChange(e.target.value)}>
            <option value="bbbc021_week1">BBBC021 Week 1</option>
            <option value="custom">Custom path…</option>
          </select>
        </FormField>

        <FormField label={isPreset ? "Path (preset)" : "Image directory"}>
          <input style={S.input} value={s.datasetPath}
            onChange={e => set({ datasetPath: e.target.value })}
            readOnly={isPreset}
            placeholder="/path/to/bbbc021_raw/Week1_22123/" />
        </FormField>

        <FormField label="Fields of view">
          <input style={{ ...S.input, width: 120 }} type="number" min={1} max={100}
            value={s.n} onChange={e => set({ n: +e.target.value || 5 })} />
        </FormField>

        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          <FormField label="Crop mode">
            <ToggleGroup value={s.cropMode} onChange={v => set({ cropMode: v })}
              options={[
                { value: "single_cell", label: "Single cell" },
                { value: "neighborhood", label: "Neighborhood" },
              ]} />
          </FormField>
          <FormField label="Size">
            <ToggleGroup value={s.sizeInvariant} onChange={v => set({ sizeInvariant: v })}
              options={[
                { value: true, label: "Invariant" },
                { value: false, label: "Aware" },
              ]} />
          </FormField>
          <FormField label="Rotation">
            <ToggleGroup value={s.rotationInvariant}
              onChange={v => set({ rotationInvariant: v })}
              options={[
                { value: true, label: "Invariant" },
                { value: false, label: "Aware" },
              ]} />
          </FormField>
        </div>

        {s.error && <p style={S.errorText}>{s.error}</p>}
      </div>
    </SectionBar>
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

  const tiles = Array.from({ length: s.numImages }, (_, i) => i);

  return (
    <SectionBar title="Segmentation"
      subtitle="Preview the segmentation on a single tile before running the full pipeline."
      actions={
        <button style={S.btnPrimary} onClick={continueToSelection}
          disabled={s.pipelineStatus === "running"}>
          Continue → Cell selection
        </button>
      }>
      <div style={S.formStack}>
        <FormField label="Method">
          <select style={S.select} value={s.segMethod}
            onChange={e => set({ segMethod: e.target.value })}>
            {SEG_METHODS.map(m => (
              <option key={m.id} value={m.id} disabled={!m.enabled}>
                {m.label}{m.hint ? ` — ${m.hint}` : ""}
              </option>
            ))}
          </select>
        </FormField>

        <FormField label="Preview tile">
          <div style={S.tileRow}>
            {tiles.map(i => (
              <button key={i}
                onClick={() => set({ previewImgIdx: i, previewMaskData: null })}
                style={i === s.previewImgIdx
                  ? { ...S.tileBtn, ...S.tileBtnActive } : S.tileBtn}>
                {i + 1}
              </button>
            ))}
          </div>
        </FormField>

        <div style={S.previewLayout}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={S.canvasFrame}>
              {rawImg ? (
                <canvas ref={canvasRef} style={{ width: "100%", display: "block" }} />
              ) : (
                <div style={S.canvasPlaceholder}>Loading tile…</div>
              )}
            </div>
          </div>

          <aside style={S.previewAside}>
            <div style={S.formSectionTitle}>Preview</div>
            <p style={{ ...S.caption, marginTop: 0, marginBottom: 12 }}>
              {s.previewMaskData
                ? `Tile ${s.previewImgIdx + 1} · ${s.previewNumCells} cells detected`
                : `Tile ${s.previewImgIdx + 1} · no preview yet`}
            </p>

            <FormField label="Overlay opacity">
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <input type="range" min={0} max={1} step={0.05} value={s.maskAlpha}
                  onChange={e => set({ maskAlpha: +e.target.value })}
                  style={{ ...S.slider, flex: 1 }}
                  disabled={!s.previewMaskData} />
                <span style={{ ...S.caption, width: 36, textAlign: "right" }}>
                  {Math.round(s.maskAlpha * 100)}%
                </span>
              </div>
            </FormField>

            <button style={{ ...S.btnSecondary, marginTop: 12, width: "100%" }}
              onClick={runPreview}
              disabled={loadingPreview || s.pipelineStatus === "running"}>
              {loadingPreview ? "Running…" : "Run preview"}
            </button>

            <p style={{ ...S.caption, marginTop: 16 }}>
              The <strong>Continue</strong> button runs the full segmentation +
              embedding pipeline on all {s.numImages} tiles.
            </p>
          </aside>
        </div>

        {s.error && <p style={S.errorText}>{s.error}</p>}
      </div>
    </SectionBar>
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

  const count = s.selected.size;
  return (
    <SectionBar title="Selected cells"
      count={count === 0 ? "none yet" : `${count} cell${count === 1 ? "" : "s"}`}
      actions={count > 0 && (
        <button style={S.textBtn}
          onClick={() => set({ selected: new Set(), positive: [] })}>
          Clear all
        </button>
      )}>
      {count === 0 ? (
        <p style={S.emptyMsg}>Pick a mode below to start selecting cells.</p>
      ) : (
        <div style={S.galleryGrid}>
          {thumbs.map(t => (
            <div key={t.object_id} style={{ ...S.resultCard, borderColor: "#BFDBFE" }}>
              <div style={S.thumbContainer}>
                <img src={t.thumbnail_base64} alt="" style={S.thumbImg} />
              </div>
              <div style={S.cardFooter}>
                <span style={{ color: "#2563EB", fontWeight: 600 }}>#{t.object_id}</span>
                <button style={S.textBtn} onClick={() => remove(t.object_id)}>Remove</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </SectionBar>
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

  const tiles = Array.from({ length: s.numImages }, (_, i) => i);

  return (
    <div style={S.formStack}>
      <FormField label="Tile">
        <div style={S.tileRow}>
          {tiles.map(i => (
            <button key={i} onClick={() => switchTile(i)}
              style={i === s.imgIdx
                ? { ...S.tileBtn, ...S.tileBtnActive } : S.tileBtn}>
              {i + 1}
            </button>
          ))}
        </div>
      </FormField>

      <div>
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
        <p style={{ ...S.caption, marginTop: 6 }}>
          {s.objects.length} cells on this tile · click to select
        </p>
      </div>

      <div style={S.controlBar}>
        <div style={S.controlGroup}>
          <div style={S.formSectionTitle}>Rejection · {s.alpha.toFixed(2)}</div>
          <input type="range" min={0} max={1} step={0.05} value={s.alpha}
            onChange={e => set({ alpha: +e.target.value })} style={S.slider} />
        </div>
        <div style={S.controlGroup}>
          <div style={S.formSectionTitle}>Min similarity · {s.threshold.toFixed(2)}</div>
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
          <div style={S.galleryDissimilar}>
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
    <div style={S.formStack}>
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
      <div>
        <button style={S.btnPrimary} disabled>Apply filter</button>
      </div>
    </div>
  );
}

function ClusteringPanel() {
  return (
    <div style={S.formStack}>
      <div style={S.banner}>
        KNN graph + Leiden endpoint not implemented. The controls below show the
        intended UI; they will activate once <code>POST /api/cluster</code> lands
        (WORKFLOW.md §5).
      </div>
      <FormField label="Feature source">
        <ToggleGroup value="dinov2" onChange={() => {}} disabled
          options={[
            { value: "classic", label: "Classic features" },
            { value: "dinov2", label: "DINOv2 deep features" },
          ]} />
      </FormField>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        <FormField label="Neighbors (k)">
          <input style={{ ...S.input, width: 120 }} type="number" disabled defaultValue={15} />
        </FormField>
        <FormField label="Leiden resolution">
          <input style={{ ...S.input, width: 120 }} type="number" disabled
            defaultValue={1.0} step={0.1} />
        </FormField>
      </div>
      <div>
        <button style={S.btnPrimary} disabled>Compute clusters</button>
      </div>
      <p style={S.emptyMsg}>
        Cluster list will appear here. Click a cluster to add all its members
        to the selection.
      </p>
    </div>
  );
}

function StepSelection({ s, set, toggle }) {
  if (s.pipelineStatus !== "done") {
    const subtitle =
      s.pipelineStatus === "running"
        ? "Running segmentation and embedding on all tiles."
        : s.pipelineStatus === "error"
        ? "Pipeline failed. Go back and try again."
        : "Finish segmentation before picking cells.";
    return (
      <SectionBar title="Cell Selection" subtitle={subtitle}
        actions={
          <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>
            Back to Segmentation
          </button>
        }>
        {s.pipelineStatus === "running" && (
          <div>
            <div style={S.progressTrack}>
              <div style={{ ...S.progressFill, width: `${(s.progress || 0) * 100}%` }} />
            </div>
            <p style={{ ...S.caption, marginTop: 8 }}>{s.msg}</p>
          </div>
        )}
        {s.error && <p style={S.errorText}>{s.error}</p>}
      </SectionBar>
    );
  }

  const mode = SEL_MODES.find(m => m.id === s.selMode) || SEL_MODES[0];
  return (
    <div style={S.stepStack}>
      <SelectionGallery s={s} set={set} />

      <SectionBar title="Cell Selection" subtitle={`${mode.label}.`}
        actions={
          <select style={S.selectCompact} value={s.selMode}
            onChange={e => set({ selMode: e.target.value })}>
            {SEL_MODES.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        }>
        {s.selMode === "similarity" && <SimilarityPanel s={s} set={set} toggle={toggle} />}
        {s.selMode === "classic" && <ClassicPanel />}
        {s.selMode === "clustering" && <ClusteringPanel />}
      </SectionBar>
    </div>
  );
}

export function scoreColor(v) {
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
      <div style={S.headerWrap}>
        <header style={S.header}>
          <span style={S.logo}>Smart Selection</span>
          <nav style={{ display: "flex", gap: 2 }}>
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
                    opacity: reachable ? 1 : 0.45,
                  }}>
                  {idx}. {label}
                </span>
              );
            })}
          </nav>
        </header>
      </div>
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
    minHeight: "100vh", background: "#F3F4F6",
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#1A1A1A", fontSize: 14,
  },
  headerWrap: {
    background: "#fff", borderBottom: "1px solid #E5E7EB",
    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
  },
  header: {
    maxWidth: 1160, margin: "0 auto", padding: "14px 24px",
    display: "flex", justifyContent: "space-between", alignItems: "center",
    boxSizing: "border-box",
  },
  logo: { fontWeight: 700, fontSize: 18, letterSpacing: "-0.03em", color: "#111" },
  main: { maxWidth: 1160, margin: "24px auto", padding: "0 24px" },

  stepPill: {
    padding: "6px 14px", fontSize: 13, color: "#9CA3AF", borderRadius: 6,
    fontWeight: 500, transition: "all 0.15s", userSelect: "none",
  },
  stepPillActive: { background: "#EFF6FF", color: "#2563EB", fontWeight: 600 },
  stepPillDone: { color: "#16A34A" },

  // Stacks between sections / within a section body
  stepStack: { display: "flex", flexDirection: "column", gap: 16 },
  formStack: { display: "flex", flexDirection: "column", gap: 18 },

  // SectionBar shell
  section: {
    background: "#fff", border: "1px solid #E5E7EB", borderRadius: 12,
    borderLeft: "4px solid #E5E7EB",
    boxShadow: "0 1px 3px rgba(0,0,0,0.04)", overflow: "hidden",
  },
  sectionHeader: {
    display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12,
    padding: "12px 20px", borderBottom: "1px solid #F3F4F6", background: "#FAFBFC",
  },
  sectionTitle: {
    fontSize: 15, fontWeight: 700, margin: 0, letterSpacing: "-0.01em",
  },
  sectionCount: { fontSize: 13, fontWeight: 500, color: "#9CA3AF" },
  sectionActions: { display: "flex", gap: 8, alignItems: "center" },
  sectionBody: { padding: 20 },

  // Typography
  muted: { color: "#6B7280", fontSize: 13, margin: "0 0 14px", lineHeight: 1.55 },
  caption: { fontSize: 12, color: "#6B7280", margin: 0, lineHeight: 1.5 },
  formSectionTitle: {
    fontSize: 11, fontWeight: 700, color: "#9CA3AF",
    textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6,
  },

  // Form controls
  input: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "8px 12px", fontSize: 13,
    color: "#374151", width: "100%", maxWidth: 520, outline: "none",
    boxSizing: "border-box", transition: "border-color 0.15s", fontFamily: "inherit",
    background: "#fff",
  },
  select: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "8px 12px", fontSize: 13,
    width: "100%", maxWidth: 320, outline: "none", boxSizing: "border-box",
    background: "#fff", color: "#1A1A1A", fontFamily: "inherit", cursor: "pointer",
  },
  selectCompact: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "6px 10px", fontSize: 13,
    background: "#fff", color: "#1A1A1A", fontFamily: "inherit", cursor: "pointer",
    outline: "none",
  },
  numInput: {
    border: "1px solid #D1D5DB", borderRadius: 6, padding: "6px 10px", fontSize: 13,
    width: 80, outline: "none", boxSizing: "border-box", background: "#F9FAFB",
    color: "#6B7280", fontFamily: "inherit",
  },

  // Toggle group (tab pills)
  toggleGroup: {
    display: "inline-flex", gap: 1, background: "#F3F4F6",
    borderRadius: 6, padding: 2,
  },
  toggleBtn: {
    padding: "5px 12px", fontSize: 12, fontWeight: 500, border: "none",
    background: "transparent", color: "#6B7280", borderRadius: 4,
    cursor: "pointer", transition: "all 0.15s", fontFamily: "inherit",
  },
  toggleBtnActive: {
    background: "#fff", color: "#111", fontWeight: 600,
    boxShadow: "0 1px 2px rgba(0,0,0,0.08)",
  },

  // Buttons
  btnPrimary: {
    background: "#2563EB", color: "#fff", border: "none", borderRadius: 8,
    padding: "8px 20px", fontSize: 13, fontWeight: 600, cursor: "pointer",
    transition: "background 0.15s",
  },
  btnSecondary: {
    background: "#fff", color: "#374151", border: "1px solid #D1D5DB", borderRadius: 8,
    padding: "8px 16px", fontSize: 13, fontWeight: 500, cursor: "pointer",
    transition: "background 0.15s",
  },
  textBtn: {
    background: "none", border: "none", color: "#9CA3AF", fontSize: 12,
    cursor: "pointer", padding: "4px 8px", fontWeight: 500, fontFamily: "inherit",
  },

  // Progress / error
  progressTrack: { height: 8, background: "#E5E7EB", borderRadius: 4, overflow: "hidden" },
  progressFill: { height: "100%", background: "#2563EB", borderRadius: 4, transition: "width 0.4s ease" },
  errorText: { color: "#DC2626", fontSize: 13, marginTop: 12, fontWeight: 500 },

  // Tile row
  tileRow: { display: "flex", flexWrap: "wrap", gap: 6 },
  tileBtn: {
    minWidth: 36, height: 36, padding: "0 10px",
    border: "1px solid #D1D5DB", background: "#fff", borderRadius: 8,
    cursor: "pointer", fontSize: 13, fontWeight: 700, color: "#6B7280",
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    transition: "all 0.15s", fontFamily: "inherit",
  },
  tileBtnActive: {
    background: "#2563EB", color: "#fff", borderColor: "#2563EB",
  },

  // Canvas
  canvasFrame: {
    border: "1px solid #E5E7EB", borderRadius: 10, overflow: "hidden",
    background: "#0A0A0A", minHeight: 200,
  },
  canvasPlaceholder: {
    padding: "80px 0", textAlign: "center", color: "#9CA3AF", fontSize: 13,
  },

  // Step 2 preview layout (canvas + side panel)
  previewLayout: { display: "flex", gap: 20, alignItems: "flex-start" },
  previewAside: {
    width: 240, flexShrink: 0, background: "#FAFBFC",
    border: "1px solid #E5E7EB", borderRadius: 10, padding: 16,
    display: "flex", flexDirection: "column",
  },

  // Gallery
  galleryGrid: {
    display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 10,
  },
  resultCard: {
    border: "1px solid #E5E7EB", borderRadius: 8, overflow: "hidden", background: "#fff",
    transition: "box-shadow 0.15s",
  },
  thumbContainer: {
    width: "100%", aspectRatio: "1", background: "#0A0A0A",
    display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
  },
  thumbImg: { maxWidth: "100%", maxHeight: "100%", objectFit: "contain", display: "block" },
  cardFooter: {
    padding: "8px 10px", display: "flex", justifyContent: "space-between",
    alignItems: "center", fontSize: 12, color: "#6B7280",
  },
  acceptBtn: {
    flex: 1, padding: 8, border: "none", borderTop: "1px solid #E5E7EB",
    background: "#F0FDF4", color: "#16A34A", fontSize: 12, fontWeight: 600,
    cursor: "pointer", transition: "background 0.15s", fontFamily: "inherit",
  },
  rejectBtn: {
    flex: 1, padding: 8, border: "none", borderTop: "1px solid #E5E7EB",
    borderLeft: "1px solid #E5E7EB", background: "#FEF2F2", color: "#DC2626",
    fontSize: 12, fontWeight: 600, cursor: "pointer",
    transition: "background 0.15s", fontFamily: "inherit",
  },

  // Similarity control bar
  controlBar: {
    display: "flex", gap: 20, alignItems: "flex-end", padding: "14px 16px",
    background: "#FAFBFC", borderRadius: 10, border: "1px solid #E5E7EB",
    flexWrap: "wrap",
  },
  controlGroup: { display: "flex", flexDirection: "column", gap: 4 },
  slider: { width: 160, accentColor: "#2563EB" },

  // Result galleries
  gallerySimilar: {
    padding: 16, background: "#F0FDF4", border: "1px solid #BBF7D0", borderRadius: 10,
  },
  galleryDissimilar: {
    padding: 16, background: "#FEF2F2", border: "1px solid #FECACA", borderRadius: 10,
  },
  galleryTitleGreen: {
    fontSize: 13, fontWeight: 700, margin: "0 0 12px", color: "#16A34A",
    textTransform: "uppercase", letterSpacing: "0.04em",
  },
  galleryTitleRed: {
    fontSize: 13, fontWeight: 700, margin: "0 0 12px", color: "#DC2626",
    textTransform: "uppercase", letterSpacing: "0.04em",
  },
  emptyMsg: {
    textAlign: "center", color: "#9CA3AF", padding: "20px 0", fontSize: 13, margin: 0,
  },

  // Stub banner (Classic / Clustering)
  banner: {
    background: "#FFFBEB", border: "1px solid #FDE68A", color: "#78350F",
    borderRadius: 8, padding: "10px 14px", fontSize: 13, lineHeight: 1.5,
    marginBottom: 16,
  },

  // Feature table (Classic stub)
  featureTable: { display: "flex", flexDirection: "column", gap: 8 },
  featureRow: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "8px 12px", background: "#FAFBFC", border: "1px solid #E5E7EB",
    borderRadius: 8,
  },
  featureLabel: { fontSize: 13, color: "#374151", fontWeight: 500 },
  rangeInputs: { display: "flex", alignItems: "center", gap: 8 },
  rangeDash: { color: "#9CA3AF", fontSize: 13 },
};

