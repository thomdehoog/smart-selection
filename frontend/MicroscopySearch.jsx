import { useState, useCallback, useEffect, useRef } from "react";

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
  process: (cropMode, sizeInvariant, rotationInvariant) => apiFetch("/segment_and_embed", { method: "POST", body: JSON.stringify({ crop_mode: cropMode, size_invariant: sizeInvariant, rotation_invariant: rotationInvariant }) }),
  status: () => apiFetch("/status"),
  image: (i) => apiFetch(`/image/${i}`),
  objects: (i) => apiFetch(`/objects?image_index=${i}`),
  mask: (i) => apiFetch(`/mask/${i}`),
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

export function useAppState() {
  const [s, setS] = useState({
    step: 1, datasetId: null, numImages: 0, channelNames: [],
    imgIdx: 0, imageData: null, objects: [],
    selected: new Set(), positive: [], negative: [], results: [], dissimilar: [],
    busy: false, progress: 0, msg: "", error: null,
    alpha: 0.4, threshold: 0.0, viewMode: "gallery",
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

/* ─── Step 1: Load ──────────────────────────────────────────────────────── */

function StepLoad({ s, set }) {
  const [dir, setDir] = useState("/Users/thomdehoog/Library/CloudStorage/Dropbox/Projects/smart-selection/microscopy-search/bbbc021_raw/Week1_22123/");
  const [n, setN] = useState(5);
  const [cropMode, setCropMode] = useState("single_cell");
  const [sizeInvariant, setSizeInvariant] = useState(true);
  const [rotationInvariant, setRotationInvariant] = useState(true);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef(null);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const loadAndProcess = async () => {
    setLoading(true); set({ error: null });
    try {
      const r = await api.upload(dir, n);
      if (r.error) { set({ error: r.error }); setLoading(false); return; }
      set({ datasetId: r.dataset_id, numImages: r.num_images, channelNames: r.channel_names,
            busy: true, progress: 0, msg: "Starting..." });
      await api.process(cropMode, sizeInvariant, rotationInvariant);
      pollRef.current = setInterval(async () => {
        const st = await api.status();
        set({ progress: st.progress || 0, msg: st.message || "" });
        if (st.status === "complete") {
          clearInterval(pollRef.current);
          const img = await api.image(0);
          const obj = await api.objects(0);
          set({ busy: false, step: 2, imageData: img, objects: obj.objects || [], imgIdx: 0 });
          setLoading(false);
        } else if (st.status === "error") {
          clearInterval(pollRef.current);
          set({ busy: false, error: st.error });
          setLoading(false);
        }
      }, 1500);
    } catch (e) { set({ error: e.message, busy: false }); setLoading(false); }
  };

  return (
    <section style={S.card}>
      <h2 style={S.h2}>Load Dataset</h2>
      <p style={S.muted}>Point to a BBBC021 plate directory. Images will be loaded, segmented, and embedded.</p>

      <div style={S.formGroup}>
        <label style={S.label}>Image directory</label>
        <input style={S.input} value={dir} onChange={e => setDir(e.target.value)}
          placeholder="/path/to/bbbc021_raw/Week1_22123/" />
      </div>

      <div style={{ display: "flex", gap: 24, marginTop: 20 }}>
        <div style={S.formGroup}>
          <label style={S.label}>Fields of view</label>
          <input style={{ ...S.input, width: 100 }} type="number" min={1} max={100}
            value={n} onChange={e => setN(+e.target.value || 5)} />
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Crop mode</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="cropMode" value="single_cell"
                checked={cropMode === "single_cell"} onChange={() => setCropMode("single_cell")} />
              Single cell
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="cropMode" value="neighborhood"
                checked={cropMode === "neighborhood"} onChange={() => setCropMode("neighborhood")} />
              Neighborhood
            </label>
          </div>
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Size</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="sizeMode" checked={sizeInvariant}
                onChange={() => setSizeInvariant(true)} />
              Invariant
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="sizeMode" checked={!sizeInvariant}
                onChange={() => setSizeInvariant(false)} />
              Aware
            </label>
          </div>
        </div>
        <div style={S.formGroup}>
          <label style={S.label}>Rotation</label>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <label style={S.radioLabel}>
              <input type="radio" name="rotMode" checked={rotationInvariant}
                onChange={() => setRotationInvariant(true)} />
              Invariant
            </label>
            <label style={S.radioLabel}>
              <input type="radio" name="rotMode" checked={!rotationInvariant}
                onChange={() => setRotationInvariant(false)} />
              Aware
            </label>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        <button style={S.btnPrimary} onClick={loadAndProcess} disabled={loading || s.busy || !dir.trim()}>
          {loading || s.busy ? "Processing..." : "Start"}
        </button>
      </div>

      {s.busy && (
        <div style={{ marginTop: 20 }}>
          <div style={S.progressTrack}>
            <div style={{ ...S.progressFill, width: `${s.progress * 100}%` }} />
          </div>
          <p style={{ ...S.caption, marginTop: 8 }}>{s.msg}</p>
        </div>
      )}
      {s.error && <p style={S.errorText}>{s.error}</p>}
    </section>
  );
}

/* ─── Step 2: Select ────────────────────────────────────────────────────── */

function StepSelect({ s, set, toggle }) {
  const canvasRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [thumbs, setThumbs] = useState([]);
  const [maskData, setMaskData] = useState(null);
  const imgRef = useRef(null);
  const maskImgRef = useRef(null);

  useEffect(() => {
    const ids = [...s.selected];
    if (ids.length === 0) { setThumbs([]); return; }
    api.crops(ids).then(d => setThumbs(d.crops || [])).catch(() => {});
  }, [s.selected]);

  useEffect(() => {
    api.mask(s.imgIdx).then(d => setMaskData(d)).catch(() => {});
  }, [s.imgIdx]);

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
      imgRef.current = img;
      cv.width = img.width; cv.height = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(0, 0, cv.width, cv.height);

      if (maskImgRef.current) {
        const mCv = document.createElement("canvas");
        mCv.width = s.imageData.width; mCv.height = s.imageData.height;
        const mCtx = mCv.getContext("2d");
        mCtx.drawImage(maskImgRef.current, 0, 0);
        const mData = mCtx.getImageData(0, 0, mCv.width, mCv.height).data;
        const sx = cv.width / s.imageData.width, sy = cv.height / s.imageData.height;

        const brightCv = document.createElement("canvas");
        brightCv.width = cv.width; brightCv.height = cv.height;
        brightCv.getContext("2d").drawImage(img, 0, 0);

        const clipCv = document.createElement("canvas");
        clipCv.width = cv.width; clipCv.height = cv.height;
        const clipCtx = clipCv.getContext("2d");

        const highlightIds = new Set([...s.selected]);
        if (hovered && !highlightIds.has(hovered)) highlightIds.add(hovered);

        if (highlightIds.size > 0) {
          for (let my = 0; my < s.imageData.height; my++) {
            for (let mx = 0; mx < s.imageData.width; mx++) {
              const idx = (my * s.imageData.width + mx) * 4;
              const cellId = mData[idx] + mData[idx + 1] * 256;
              if (cellId > 0 && highlightIds.has(cellId)) {
                const cx = Math.floor(mx * sx), cy = Math.floor(my * sy);
                const cw = Math.max(Math.ceil(sx), 1), ch = Math.max(Math.ceil(sy), 1);
                clipCtx.fillStyle = hovered === cellId && !s.selected.has(cellId) ? "rgba(200,220,255,0.7)" : "white";
                clipCtx.fillRect(cx, cy, cw, ch);
              }
            }
          }
          ctx.globalCompositeOperation = "source-over";
          clipCtx.globalCompositeOperation = "source-in";
          clipCtx.drawImage(brightCv, 0, 0);
          ctx.drawImage(clipCv, 0, 0);
        }
      }
    };
    img.src = s.imageData.thumbnail_base64;
  }, [s.imageData, s.objects, s.selected, hovered, maskData]);

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
    const cellId = pixel[0] + pixel[1] * 256;
    return cellId > 0 ? cellId : null;
  };

  const switchImg = async (i) => {
    const [img, obj, msk] = await Promise.all([api.image(i), api.objects(i), api.mask(i)]);
    setMaskData(msk);
    set({ imageData: img, objects: obj.objects || [], imgIdx: i });
  };

  return (
    <section style={S.card}>
      <div style={{ marginBottom: 12 }}>
        <h2 style={S.h2}>Select Cells</h2>
        <p style={{ ...S.muted, margin: 0 }}>{s.objects.length} cells detected · {s.selected.size} selected</p>
      </div>

      <div style={{ display: "flex", gap: 20, marginTop: 12 }}>
        {/* Image tiles */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
          {Array.from({ length: s.numImages }, (_, i) => (
            <button key={i} onClick={() => switchImg(i)}
              style={{
                width: 48, height: 48, border: i === s.imgIdx ? "2px solid #2563EB" : "1px solid #D1D5DB",
                background: i === s.imgIdx ? "#EFF6FF" : "#fff", borderRadius: 10,
                cursor: "pointer", fontSize: 16, fontWeight: 700,
                color: i === s.imgIdx ? "#2563EB" : "#6B7280",
                display: "flex", alignItems: "center", justifyContent: "center",
                transition: "all 0.15s",
              }}>{i + 1}</button>
          ))}
        </div>

        {/* Image canvas */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={S.canvasFrame}>
            <canvas ref={canvasRef}
              style={{ width: "100%", display: "block", cursor: "crosshair" }}
              onClick={e => { const id = hitTest(e); if (id) toggle(id); }}
              onMouseMove={e => setHovered(hitTest(e))}
              onMouseLeave={() => setHovered(null)} />
          </div>
        </div>

        <div style={S.sidebar}>
          <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 12, color: "#1A1A1A" }}>
            Selected ({s.selected.size})
          </div>
          {s.selected.size === 0 && <p style={{ ...S.caption, color: "#9CA3AF" }}>Click cells in the image to select them</p>}
          <div style={S.thumbGrid}>
            {thumbs.map(t => (
              <div key={t.object_id} style={S.sideThumb} onClick={() => toggle(t.object_id)}>
                <img src={t.thumbnail_base64} alt="" style={{ width: "100%", display: "block", borderRadius: 4 }} />
                <span style={S.sideThumbId}>#{t.object_id}</span>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 16, paddingTop: 16, borderTop: "1px solid #E5E7EB" }}>
            <button style={S.btnSecondary} onClick={() => set({ selected: new Set() })}>Clear</button>
            <button style={{ ...S.btnPrimary, flex: 1 }} disabled={s.selected.size === 0}
              onClick={() => set({ step: 3, positive: [...s.selected] })}>Review</button>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─── Step 3: Review ────────────────────────────────────────────────────── */

function StepReview({ s, set }) {
  const [crops, setCrops] = useState([]);
  useEffect(() => { if (s.positive.length > 0) api.crops(s.positive).then(d => setCrops(d.crops || [])); }, [s.positive]);

  const remove = (id) => {
    const ns = new Set(s.selected); ns.delete(id);
    set({ positive: s.positive.filter(x => x !== id), selected: ns });
  };

  const search = async () => {
    set({ step: 4, results: [], negative: [], dissimilar: [] });
    const [sim, dis] = await Promise.all([
      api.search(s.positive, [], 50, s.alpha),
      api.searchDissimilar(s.positive, 28),
    ]);
    set({ results: sim.results || [], dissimilar: dis.results || [] });
  };

  return (
    <section style={S.card}>
      <h2 style={S.h2}>Review Selections</h2>
      <p style={S.muted}>{crops.length} cells selected. Remove any mistakes before searching.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12, margin: "20px 0" }}>
        {crops.map(c => (
          <div key={c.object_id} style={S.resultCard}>
            <div style={S.thumbContainer}>
              <img src={c.thumbnail_base64} alt="" style={S.thumbImg} />
            </div>
            <div style={S.cardFooter}>
              <span style={{ color: "#374151", fontWeight: 500 }}>#{c.object_id}</span>
              <button style={S.linkBtn} onClick={() => remove(c.object_id)}>Remove</button>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12 }}>
        <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>Back</button>
        <button style={S.btnPrimary} onClick={search} disabled={s.positive.length === 0}>Find Similar</button>
      </div>
    </section>
  );
}

/* ─── Step 4: Search ────────────────────────────────────────────────────── */

function MapView({ s, set }) {
  const canvasRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [maskData, setMaskData] = useState(null);
  const [mapImageData, setMapImageData] = useState(null);
  const [mapImgIdx, setMapImgIdx] = useState(s.imgIdx);
  const imgRef = useRef(null);
  const maskImgRef = useRef(null);

  // Build lookups — mask-based rendering naturally filters to current image
  const resultScores = new Map();
  for (const r of s.results) resultScores.set(r.object_id, r.similarity_score);
  const positiveSet = new Set(s.positive);

  useEffect(() => {
    api.image(mapImgIdx).then(d => setMapImageData(d)).catch(() => {});
    api.mask(mapImgIdx).then(d => setMaskData(d)).catch(() => {});
  }, [mapImgIdx]);

  useEffect(() => {
    if (!maskData) return;
    const mImg = new window.Image();
    mImg.onload = () => { maskImgRef.current = mImg; };
    mImg.src = maskData.mask_base64;
  }, [maskData]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !mapImageData) return;
    const ctx = cv.getContext("2d");
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      cv.width = img.width; cv.height = img.height;
      ctx.drawImage(img, 0, 0);
      ctx.fillStyle = "rgba(0,0,0,0.4)";
      ctx.fillRect(0, 0, cv.width, cv.height);

      if (maskImgRef.current) {
        const mCv = document.createElement("canvas");
        mCv.width = mapImageData.width; mCv.height = mapImageData.height;
        const mCtx = mCv.getContext("2d");
        mCtx.drawImage(maskImgRef.current, 0, 0);
        const mData = mCtx.getImageData(0, 0, mCv.width, mCv.height).data;
        const sx = cv.width / mapImageData.width, sy = cv.height / mapImageData.height;

        const brightCv = document.createElement("canvas");
        brightCv.width = cv.width; brightCv.height = cv.height;
        brightCv.getContext("2d").drawImage(img, 0, 0);

        const clipCv = document.createElement("canvas");
        clipCv.width = cv.width; clipCv.height = cv.height;
        const clipCtx = clipCv.getContext("2d");

        const allHighlight = new Set([...positiveSet]);
        for (const id of resultScores.keys()) allHighlight.add(id);
        if (hovered && !allHighlight.has(hovered)) allHighlight.add(hovered);

        if (allHighlight.size > 0) {
          for (let my = 0; my < mapImageData.height; my++) {
            for (let mx = 0; mx < mapImageData.width; mx++) {
              const idx = (my * mapImageData.width + mx) * 4;
              const cellId = mData[idx] + mData[idx + 1] * 256;
              if (cellId > 0 && allHighlight.has(cellId)) {
                const cx = Math.floor(mx * sx), cy = Math.floor(my * sy);
                const cw = Math.max(Math.ceil(sx), 1), ch = Math.max(Math.ceil(sy), 1);
                if (hovered === cellId && !positiveSet.has(cellId) && !resultScores.has(cellId)) {
                  clipCtx.fillStyle = "rgba(200,220,255,0.7)";
                } else if (positiveSet.has(cellId)) {
                  clipCtx.fillStyle = "rgba(100,160,255,0.9)";
                } else if (resultScores.has(cellId)) {
                  const score = resultScores.get(cellId);
                  const g = Math.round(160 + score * 95);
                  clipCtx.fillStyle = `rgba(80,${g},100,0.85)`;
                } else {
                  clipCtx.fillStyle = "rgba(200,220,255,0.7)";
                }
                clipCtx.fillRect(cx, cy, cw, ch);
              }
            }
          }
          ctx.globalCompositeOperation = "source-over";
          clipCtx.globalCompositeOperation = "source-in";
          clipCtx.drawImage(brightCv, 0, 0);
          ctx.drawImage(clipCv, 0, 0);
        }
      }
    };
    img.src = mapImageData.thumbnail_base64;
  }, [mapImageData, maskData, hovered, s.positive, s.results, s.threshold]);

  const hitTest = (e) => {
    const cv = canvasRef.current;
    if (!cv || !mapImageData || !maskImgRef.current) return null;
    const r = cv.getBoundingClientRect();
    const mx = Math.floor(((e.clientX - r.left) / r.width) * mapImageData.width);
    const my = Math.floor(((e.clientY - r.top) / r.height) * mapImageData.height);
    const mCv = document.createElement("canvas");
    mCv.width = mapImageData.width; mCv.height = mapImageData.height;
    const mCtx = mCv.getContext("2d");
    mCtx.drawImage(maskImgRef.current, 0, 0);
    const pixel = mCtx.getImageData(mx, my, 1, 1).data;
    const cellId = pixel[0] + pixel[1] * 256;
    return cellId > 0 ? cellId : null;
  };

  const hoveredResult = hovered ? s.results.find(r => r.object_id === hovered) : null;
  const hoveredIsPositive = hovered && s.positive.includes(hovered);

  return (
    <div style={{ display: "flex", gap: 20 }}>
      {/* Image tiles */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6, flexShrink: 0 }}>
        {Array.from({ length: s.numImages }, (_, i) => (
          <button key={i} onClick={() => setMapImgIdx(i)}
            style={{
              width: 48, height: 48, border: i === mapImgIdx ? "2px solid #2563EB" : "1px solid #D1D5DB",
              background: i === mapImgIdx ? "#EFF6FF" : "#fff", borderRadius: 10,
              cursor: "pointer", fontSize: 16, fontWeight: 700,
              color: i === mapImgIdx ? "#2563EB" : "#6B7280",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.15s",
            }}>{i + 1}</button>
        ))}
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={S.canvasFrame}>
          <canvas ref={canvasRef}
            style={{ width: "100%", display: "block", cursor: "crosshair" }}
            onMouseMove={e => setHovered(hitTest(e))}
            onMouseLeave={() => setHovered(null)} />
        </div>
      </div>

      {/* Legend + hover info sidebar */}
      <div style={S.sidebar}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 12, color: "#1A1A1A" }}>Legend</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: 4, background: "rgba(100,160,255,0.9)" }} />
            <span style={{ fontSize: 13, color: "#374151" }}>Query cells ({positiveSet.size})</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: 4, background: "rgba(80,220,100,0.85)" }} />
            <span style={{ fontSize: 13, color: "#374151" }}>Similar results ({resultScores.size})</span>
          </div>
        </div>

        {(hoveredResult || hoveredIsPositive) && (
          <div style={{ padding: 12, background: "#F9FAFB", borderRadius: 8, border: "1px solid #E5E7EB" }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: "#111" }}>
              Cell #{hovered}
            </div>
            {hoveredResult && (
              <div style={{ fontSize: 13, color: "#6B7280", marginTop: 4 }}>
                Similarity: <span style={{ fontWeight: 600, color: hoveredResult.similarity_score > 0.8 ? "#16A34A" : hoveredResult.similarity_score > 0.5 ? "#CA8A04" : "#DC2626" }}>
                  {(hoveredResult.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {hoveredIsPositive && (
              <div style={{ fontSize: 13, color: "#2563EB", marginTop: 4, fontWeight: 500 }}>Query cell</div>
            )}
          </div>
        )}
        {!hoveredResult && !hoveredIsPositive && hovered && (
          <div style={{ padding: 12, background: "#F9FAFB", borderRadius: 8, border: "1px solid #E5E7EB" }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: "#111" }}>Cell #{hovered}</div>
            <div style={{ fontSize: 13, color: "#9CA3AF", marginTop: 4 }}>Not in results</div>
          </div>
        )}
      </div>
    </div>
  );
}

function StepSearch({ s, set }) {
  const [posCrops, setPosCrops] = useState([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (s.positive.length > 0) api.crops(s.positive).then(d => setPosCrops(d.crops || [])); }, [s.positive]);

  const accept = (id) => {
    set({ positive: [...s.positive, id], results: s.results.filter(r => r.object_id !== id), dissimilar: s.dissimilar.filter(r => r.object_id !== id) });
  };
  const reject = (id) => {
    set({ negative: [...s.negative, id], results: s.results.filter(r => r.object_id !== id), dissimilar: s.dissimilar.filter(r => r.object_id !== id) });
  };
  const recompute = async () => {
    setBusy(true);
    const [sim, dis] = await Promise.all([
      api.search(s.positive, s.negative, 50, s.alpha),
      api.searchDissimilar(s.positive, 28),
    ]);
    set({ results: sim.results || [], dissimilar: dis.results || [] });
    setBusy(false);
  };

  const visible = s.results.filter(r => r.similarity_score >= s.threshold);
  const COLS = 5;
  const MAX_ROWS = 4;
  const maxItems = COLS * MAX_ROWS;
  const mostSimilar = visible.slice(0, maxItems);
  const mostDissimilar = s.dissimilar.slice(0, maxItems);

  const ResultCard = ({ r }) => (
    <div style={S.resultCard}>
      <div style={S.thumbContainer}>
        <img src={r.thumbnail_base64} alt="" style={S.thumbImg} />
      </div>
      <div style={S.cardFooter}>
        <span>#{r.object_id}</span>
        <span style={{ fontWeight: 600, color: r.similarity_score > 0.8 ? "#16A34A" : r.similarity_score > 0.5 ? "#CA8A04" : "#DC2626" }}>
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
    <section style={S.card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div>
          <h2 style={S.h2}>Search Results</h2>
          <p style={{ ...S.muted, margin: 0 }}>{s.positive.length} positive · {s.negative.length} rejected · {visible.length} matches</p>
        </div>
        <div style={{ display: "flex", gap: 4, background: "#F3F4F6", borderRadius: 8, padding: 3 }}>
          <button
            onClick={() => set({ viewMode: "gallery" })}
            style={{
              ...S.viewToggleBtn,
              ...(s.viewMode === "gallery" ? S.viewToggleBtnActive : {}),
            }}>Gallery</button>
          <button
            onClick={() => set({ viewMode: "map" })}
            style={{
              ...S.viewToggleBtn,
              ...(s.viewMode === "map" ? S.viewToggleBtnActive : {}),
            }}>Map</button>
        </div>
      </div>

      {s.viewMode === "gallery" && (
        <>
          {/* Selected cells */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 12, margin: "16px 0 24px" }}>
            {posCrops.map(c => (
              <div key={c.object_id} style={{ ...S.resultCard, borderColor: "#2563EB", borderWidth: 2 }}>
                <div style={S.thumbContainer}>
                  <img src={c.thumbnail_base64} alt="" style={S.thumbImg} />
                </div>
                <div style={S.cardFooter}>
                  <span style={{ color: "#2563EB", fontWeight: 600 }}>#{c.object_id}</span>
                  <button onClick={() => {
                    const np = s.positive.filter(x => x !== c.object_id);
                    const ns = new Set(s.selected); ns.delete(c.object_id);
                    set({ positive: np, selected: ns });
                  }} style={S.linkBtn}>Remove</button>
                </div>
              </div>
            ))}
          </div>

          {/* Controls */}
          <div style={S.controlBar}>
            <div style={S.controlGroup}>
              <label style={S.controlLabel}>Rejection strength ({s.alpha.toFixed(2)})</label>
              <input type="range" min={0} max={1} step={0.05} value={s.alpha}
                onChange={e => set({ alpha: +e.target.value })} style={S.slider} />
            </div>
            <div style={S.controlGroup}>
              <label style={S.controlLabel}>Min similarity ({s.threshold.toFixed(2)})</label>
              <input type="range" min={0} max={1} step={0.05} value={s.threshold}
                onChange={e => set({ threshold: +e.target.value })} style={S.slider} />
            </div>
            <button style={S.btnPrimary} onClick={recompute} disabled={busy}>
              {busy ? "Searching..." : "Recompute"}
            </button>
          </div>

          {/* Most Similar */}
          <div style={S.gallerySection}>
            <div style={S.gallerySimilar}>
              <h3 style={S.galleryTitleGreen}>Most Similar</h3>
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: 10 }}>
                {mostSimilar.map(r => <ResultCard key={r.object_id} r={r} />)}
              </div>
              {mostSimilar.length === 0 && !busy && (
                <p style={S.emptyMsg}>No results above threshold.</p>
              )}
            </div>
          </div>

          {/* Most Dissimilar */}
          <div style={{ ...S.gallerySection, marginTop: 16 }}>
            <div style={S.galleryDissimilar}>
              <h3 style={S.galleryTitleRed}>Most Dissimilar</h3>
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: 10 }}>
                {mostDissimilar.map(r => <ResultCard key={r.object_id} r={r} />)}
              </div>
              {mostDissimilar.length === 0 && !busy && (
                <p style={S.emptyMsg}>No dissimilar results loaded yet.</p>
              )}
            </div>
          </div>
        </>
      )}

      {s.viewMode === "map" && <MapView s={s} set={set} />}

      <div style={{ marginTop: 24 }}>
        <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>Select More</button>
      </div>
    </section>
  );
}

/* ─── App Shell ─────────────────────────────────────────────────────────── */

export default function App() {
  const { s, set, toggle } = useAppState();
  const steps = ["Load Data", "Select", "Review", "Search"];
  return (
    <div style={S.root}>
      <header style={S.header}>
        <span style={S.logo}>Microscopy Search</span>
        <nav style={{ display: "flex", gap: 4 }}>
          {steps.map((label, i) => (
            <span key={i}
              onClick={() => { if (s.step >= 2 && i + 1 <= s.step) set({ step: i + 1 }); }}
              style={{
                ...S.stepPill,
                ...(s.step === i + 1 ? S.stepPillActive : {}),
                ...(s.step > i + 1 ? S.stepPillDone : {}),
                cursor: s.step >= 2 && i + 1 <= s.step ? "pointer" : "default",
              }}>
              {i + 1}. {label}
            </span>
          ))}
        </nav>
      </header>
      <main style={S.main}>
        {s.step === 1 && <StepLoad s={s} set={set} />}
        {s.step === 2 && <StepSelect s={s} set={set} toggle={toggle} />}
        {s.step === 3 && <StepReview s={s} set={set} />}
        {s.step === 4 && <StepSearch s={s} set={set} />}
      </main>
    </div>
  );
}

/* ─── Styles ────────────────────────────────────────────────────────────── */

const S = {
  // Layout
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

  // Step pills
  stepPill: {
    padding: "8px 16px", fontSize: 14, color: "#9CA3AF", borderRadius: 8,
    fontWeight: 500, transition: "all 0.15s",
  },
  stepPillActive: { background: "#EFF6FF", color: "#2563EB", fontWeight: 600 },
  stepPillDone: { color: "#16A34A" },

  // Card
  card: {
    background: "#fff", border: "1px solid #E2E4E9", borderRadius: 16,
    padding: 32, boxShadow: "0 1px 4px rgba(0,0,0,0.04)",
  },

  // Typography
  h2: { fontSize: 22, fontWeight: 700, margin: "0 0 6px", letterSpacing: "-0.02em", color: "#111" },
  muted: { color: "#6B7280", fontSize: 15, margin: "0 0 16px", lineHeight: 1.6 },
  caption: { fontSize: 14, color: "#6B7280", margin: 0 },

  // Forms
  formGroup: { display: "flex", flexDirection: "column" },
  label: { display: "block", fontSize: 14, fontWeight: 600, color: "#374151", marginBottom: 6 },
  input: {
    border: "1px solid #D1D5DB", borderRadius: 8, padding: "10px 14px", fontSize: 15,
    width: "100%", maxWidth: 560, outline: "none", boxSizing: "border-box",
    transition: "border-color 0.15s",
  },
  radioLabel: {
    fontSize: 15, cursor: "pointer", display: "flex", alignItems: "center", gap: 6,
    color: "#374151",
  },

  // Buttons
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

  // Progress
  progressTrack: { height: 8, background: "#E5E7EB", borderRadius: 4, overflow: "hidden" },
  progressFill: { height: "100%", background: "#2563EB", borderRadius: 4, transition: "width 0.4s ease" },
  errorText: { color: "#DC2626", fontSize: 14, marginTop: 16, fontWeight: 500 },

  // Image tabs
  tabBtn: {
    width: 36, height: 36, border: "1px solid #D1D5DB", background: "#fff", borderRadius: 8,
    cursor: "pointer", fontSize: 14, fontWeight: 600, color: "#6B7280",
    transition: "all 0.15s",
  },
  tabBtnActive: { background: "#2563EB", color: "#fff", borderColor: "#2563EB" },

  // Canvas
  canvasFrame: {
    border: "1px solid #E2E4E9", borderRadius: 12, overflow: "hidden", background: "#111",
  },

  // Sidebar
  sidebar: {
    width: 260, flexShrink: 0, background: "#FAFBFC", border: "1px solid #E2E4E9",
    borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", maxHeight: "72vh",
  },
  thumbGrid: {
    display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, overflowY: "auto", flex: 1,
  },
  sideThumb: {
    position: "relative", cursor: "pointer", borderRadius: 6, overflow: "hidden",
    border: "2px solid #2563EB",
  },
  sideThumbId: {
    position: "absolute", bottom: 3, left: 5, fontSize: 11, fontWeight: 600,
    color: "#fff", textShadow: "0 1px 4px rgba(0,0,0,0.8)",
  },

  // Result cards
  resultCard: {
    border: "1px solid #E2E4E9", borderRadius: 10, overflow: "hidden",
    background: "#fff", transition: "box-shadow 0.15s",
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

  // Controls bar
  controlBar: {
    display: "flex", gap: 24, alignItems: "flex-end", padding: "16px 20px",
    background: "#FAFBFC", borderRadius: 12, border: "1px solid #E2E4E9",
    flexWrap: "wrap", marginBottom: 24,
  },
  controlGroup: { display: "flex", flexDirection: "column", gap: 6 },
  controlLabel: { fontSize: 13, color: "#6B7280", fontWeight: 500 },
  slider: { width: 180, accentColor: "#2563EB" },

  // Gallery sections
  gallerySection: {},
  gallerySimilar: {
    padding: 20, background: "#F0FDF4", border: "1px solid #BBF7D0",
    borderRadius: 12,
  },
  galleryDissimilar: {
    padding: 20, background: "#FEF2F2", border: "1px solid #FECACA",
    borderRadius: 12,
  },
  galleryTitleGreen: {
    fontSize: 16, fontWeight: 700, margin: "0 0 14px", color: "#16A34A",
    letterSpacing: "-0.01em",
  },
  galleryTitleRed: {
    fontSize: 16, fontWeight: 700, margin: "0 0 14px", color: "#DC2626",
    letterSpacing: "-0.01em",
  },
  emptyMsg: {
    textAlign: "center", color: "#9CA3AF", padding: "32px 0", fontSize: 14,
  },

  // View toggle
  viewToggleBtn: {
    padding: "6px 16px", fontSize: 13, fontWeight: 500, border: "none",
    background: "transparent", color: "#6B7280", borderRadius: 6,
    cursor: "pointer", transition: "all 0.15s",
  },
  viewToggleBtnActive: {
    background: "#fff", color: "#111", fontWeight: 600,
    boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
  },
};
