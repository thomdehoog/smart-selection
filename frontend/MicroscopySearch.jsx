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
  search: (pos, neg, k, mode) =>
    apiFetch("/search", {
      method: "POST",
      body: JSON.stringify({ positive_ids: pos, negative_ids: neg, top_k: k, negative_alpha: 1.0, search_mode: mode || "weighted" }),
    }),
  listProjects: () => apiFetch("/list_projects"),
  loadProject: (name) =>
    apiFetch("/load_project", { method: "POST", body: JSON.stringify({ project_name: name }) }),
  saveProject: (name, posIds, negIds, searchMode, threshold, perImageCap) =>
    apiFetch("/save_project", {
      method: "POST",
      body: JSON.stringify({
        project_name: name,
        positive_ids: posIds && posIds.length > 0 ? posIds : null,
        negative_ids: negIds && negIds.length > 0 ? negIds : null,
        search_mode: searchMode, threshold, per_image_cap: perImageCap,
      }),
    }),
};

export function useAppState() {
  const [s, setS] = useState({
    step: 1, datasetId: null, numImages: 0, channelNames: [],
    imgIdx: 0, imageData: null, objects: [],
    classifierPos: new Set(), classifierNeg: new Set(),
    results: [],
    busy: false, progress: 0, msg: "", error: null,
    threshold: 0.0, perImageCap: 50, viewMode: "gallery", searchMode: "weighted",
    projectName: null, saving: false,
  });
  const set = useCallback((p) => setS((prev) => ({ ...prev, ...p })), []);
  const togglePos = useCallback((id) => {
    setS((prev) => {
      const next = new Set(prev.classifierPos);
      const neg = new Set(prev.classifierNeg);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        neg.delete(id); // can't be in both
      }
      return { ...prev, classifierPos: next, classifierNeg: neg };
    });
  }, []);
  const toggleNeg = useCallback((id) => {
    setS((prev) => {
      const next = new Set(prev.classifierNeg);
      const pos = new Set(prev.classifierPos);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        pos.delete(id); // can't be in both
      }
      return { ...prev, classifierNeg: next, classifierPos: pos };
    });
  }, []);
  return { s, set, togglePos, toggleNeg };
}

/* ─── Step 1: Load ──────────────────────────────────────────────────────── */

function ProjectPicker({ onLoad, onClose }) {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listProjects().then(d => { setProjects(d.projects || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  return (
    <div style={S.modalOverlay}>
      <div style={S.modalCard}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ ...S.h2, margin: 0 }}>Load Project</h2>
          <button style={{ ...S.linkBtn, color: "#6B7280", fontSize: 20, padding: 4 }} onClick={onClose}>&times;</button>
        </div>
        {loading && <p style={S.muted}>Loading projects...</p>}
        {!loading && projects.length === 0 && (
          <p style={S.muted}>No saved projects found. Start a new dataset to create one.</p>
        )}
        {projects.map(p => (
          <div key={p.project_name} onClick={() => onLoad(p.project_name)}
            style={S.projectRow}>
            <div>
              <div style={{ fontWeight: 600, fontSize: 15, color: "#111" }}>{p.project_name}</div>
              <div style={{ fontSize: 13, color: "#6B7280", marginTop: 2 }}>
                {p.num_images} images · {p.num_objects} cells
                {p.has_classifier && " · classifier saved"}
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#9CA3AF" }}>
              {p.created ? new Date(p.created).toLocaleDateString() : ""}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StepLoad({ s, set }) {
  const [dir, setDir] = useState("/Users/thomdehoog/Library/CloudStorage/Dropbox/Projects/smart-selection/microscopy-search/bbbc021_raw/Week1_22123/");
  const [n, setN] = useState(5);
  const [cropMode, setCropMode] = useState("single_cell");
  const [sizeInvariant, setSizeInvariant] = useState(true);
  const [rotationInvariant, setRotationInvariant] = useState(true);
  const [loading, setLoading] = useState(false);
  const [showPicker, setShowPicker] = useState(false);
  const pollRef = useRef(null);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const loadProject = async (projectName) => {
    setShowPicker(false);
    setLoading(true); set({ error: null });
    try {
      const r = await api.loadProject(projectName);
      if (r.error) { set({ error: r.error }); setLoading(false); return; }
      set({
        datasetId: r.dataset_id, numImages: r.num_images,
        channelNames: r.channel_names || ["DAPI", "Ch2", "Ch3"],
        projectName: projectName,
      });
      // Restore classifier state if saved
      const cls = r.classifier || {};
      if (cls.positive_ids && cls.positive_ids.length > 0) {
        set({
          classifierPos: new Set(cls.positive_ids),
          classifierNeg: new Set(cls.negative_ids || []),
          searchMode: cls.search_mode || "weighted",
          threshold: cls.threshold || 0.0,
          perImageCap: cls.per_image_cap || 50,
        });
      }
      const img = await api.image(0);
      const obj = await api.objects(0);
      set({ imageData: img, objects: obj.objects || [], imgIdx: 0, step: 2 });
      setLoading(false);
    } catch (e) { set({ error: e.message }); setLoading(false); }
  };

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
      {showPicker && <ProjectPicker onLoad={loadProject} onClose={() => setShowPicker(false)} />}

      <h2 style={S.h2}>Load Dataset</h2>
      <p style={S.muted}>Load a saved project or point to a BBBC021 plate directory for a new analysis.</p>

      <div style={{ marginBottom: 24 }}>
        <button style={S.btnSecondary} onClick={() => setShowPicker(true)} disabled={loading || s.busy}>
          Open Saved Project
        </button>
      </div>

      <div style={{ borderTop: "1px solid #E5E7EB", paddingTop: 20 }}>
        <h3 style={{ fontSize: 16, fontWeight: 600, margin: "0 0 12px", color: "#374151" }}>New Dataset</h3>
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

/* ─── Step 2: Select Exemplars ─────────────────────────────────────────── */

function StepSelect({ s, set, togglePos, toggleNeg }) {
  const canvasRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [posThumbs, setPosThumbs] = useState([]);
  const [negThumbs, setNegThumbs] = useState([]);
  const [maskData, setMaskData] = useState(null);
  const imgRef = useRef(null);
  const maskImgRef = useRef(null);

  useEffect(() => {
    const posIds = [...s.classifierPos];
    if (posIds.length === 0) setPosThumbs([]);
    else api.crops(posIds).then(d => setPosThumbs(d.crops || [])).catch(() => {});
  }, [s.classifierPos]);

  useEffect(() => {
    const negIds = [...s.classifierNeg];
    if (negIds.length === 0) setNegThumbs([]);
    else api.crops(negIds).then(d => setNegThumbs(d.crops || [])).catch(() => {});
  }, [s.classifierNeg]);

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

        const highlightIds = new Set([...s.classifierPos, ...s.classifierNeg]);
        if (hovered && !highlightIds.has(hovered)) highlightIds.add(hovered);

        if (highlightIds.size > 0) {
          for (let my = 0; my < s.imageData.height; my++) {
            for (let mx = 0; mx < s.imageData.width; mx++) {
              const idx = (my * s.imageData.width + mx) * 4;
              const cellId = mData[idx] + mData[idx + 1] * 256;
              if (cellId > 0 && highlightIds.has(cellId)) {
                const cx = Math.floor(mx * sx), cy = Math.floor(my * sy);
                const cw = Math.max(Math.ceil(sx), 1), ch = Math.max(Math.ceil(sy), 1);
                if (hovered === cellId && !s.classifierPos.has(cellId) && !s.classifierNeg.has(cellId)) {
                  clipCtx.fillStyle = "rgba(200,220,255,0.7)";
                } else if (s.classifierNeg.has(cellId)) {
                  clipCtx.fillStyle = "rgba(255,130,130,0.85)";
                } else {
                  clipCtx.fillStyle = "rgba(130,200,255,0.85)";
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
    img.src = s.imageData.thumbnail_base64;
  }, [s.imageData, s.objects, s.classifierPos, s.classifierNeg, hovered, maskData]);

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
        <h2 style={S.h2}>Select Exemplars</h2>
        <p style={{ ...S.muted, margin: 0 }}>
          {s.objects.length} cells detected · Left-click = positive · Right-click = negative
        </p>
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
              onClick={e => { const id = hitTest(e); if (id) togglePos(id); }}
              onContextMenu={e => { e.preventDefault(); const id = hitTest(e); if (id) toggleNeg(id); }}
              onMouseMove={e => setHovered(hitTest(e))}
              onMouseLeave={() => setHovered(null)} />
          </div>
        </div>

        <div style={S.sidebar}>
          {/* Positive exemplars */}
          <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 8, color: "#2563EB" }}>
            Positive ({s.classifierPos.size})
          </div>
          {s.classifierPos.size === 0 && <p style={{ ...S.caption, color: "#9CA3AF" }}>Left-click cells to add</p>}
          <div style={S.thumbGrid}>
            {posThumbs.map(t => (
              <div key={t.object_id} style={{ ...S.sideThumb, borderColor: "#2563EB" }} onClick={() => togglePos(t.object_id)}>
                <img src={t.thumbnail_base64} alt="" style={{ width: "100%", display: "block", borderRadius: 4 }} />
                <span style={S.sideThumbId}>#{t.object_id}</span>
              </div>
            ))}
          </div>

          {/* Negative exemplars */}
          <div style={{ fontWeight: 600, fontSize: 15, marginTop: 16, marginBottom: 8, color: "#DC2626" }}>
            Negative ({s.classifierNeg.size})
          </div>
          {s.classifierNeg.size === 0 && <p style={{ ...S.caption, color: "#9CA3AF" }}>Right-click cells to add</p>}
          <div style={S.thumbGrid}>
            {negThumbs.map(t => (
              <div key={t.object_id} style={{ ...S.sideThumb, borderColor: "#DC2626" }} onClick={() => toggleNeg(t.object_id)}>
                <img src={t.thumbnail_base64} alt="" style={{ width: "100%", display: "block", borderRadius: 4 }} />
                <span style={S.sideThumbId}>#{t.object_id}</span>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 16, paddingTop: 16, borderTop: "1px solid #E5E7EB" }}>
            <button style={S.btnSecondary} onClick={() => set({ classifierPos: new Set(), classifierNeg: new Set() })}>Clear</button>
            <button style={{ ...S.btnPrimary, flex: 1 }} disabled={s.classifierPos.size === 0}
              onClick={() => set({ step: 3 })}>Review</button>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─── Step 3: Review Classifier ────────────────────────────────────────── */

function StepReview({ s, set, togglePos, toggleNeg, onSave }) {
  const [posCrops, setPosCrops] = useState([]);
  const [negCrops, setNegCrops] = useState([]);

  useEffect(() => {
    const posIds = [...s.classifierPos];
    if (posIds.length === 0) setPosCrops([]);
    else api.crops(posIds).then(d => setPosCrops(d.crops || []));
  }, [s.classifierPos]);

  useEffect(() => {
    const negIds = [...s.classifierNeg];
    if (negIds.length === 0) setNegCrops([]);
    else api.crops(negIds).then(d => setNegCrops(d.crops || []));
  }, [s.classifierNeg]);

  const apply = async () => {
    set({ step: 4, results: [] });
    const sim = await api.search([...s.classifierPos], [...s.classifierNeg], 200, s.searchMode);
    set({ results: sim.results || [] });
  };

  return (
    <section style={S.card}>
      <h2 style={S.h2}>Review Classifier</h2>
      <p style={S.muted}>
        {s.classifierPos.size} positive{s.classifierNeg.size > 0 ? ` · ${s.classifierNeg.size} negative` : ""} exemplars.
        Remove any mistakes before applying.
      </p>

      {/* Positive exemplars */}
      <h3 style={{ ...S.galleryTitleGreen, marginTop: 16 }}>Positive Exemplars</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, margin: "0 0 20px" }}>
        {posCrops.map(c => (
          <div key={c.object_id} style={{ ...S.resultCard, borderColor: "#2563EB", borderWidth: 2 }}>
            <div style={S.thumbContainer}>
              <img src={c.thumbnail_base64} alt="" style={S.thumbImg} />
            </div>
            <div style={S.cardFooter}>
              <span style={{ color: "#2563EB", fontWeight: 600 }}>#{c.object_id}</span>
              <button style={S.linkBtn} onClick={() => togglePos(c.object_id)}>Remove</button>
            </div>
          </div>
        ))}
      </div>

      {/* Negative exemplars */}
      {negCrops.length > 0 && (
        <>
          <h3 style={{ ...S.galleryTitleRed, marginTop: 8 }}>Negative Exemplars</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 12, margin: "0 0 20px" }}>
            {negCrops.map(c => (
              <div key={c.object_id} style={{ ...S.resultCard, borderColor: "#DC2626", borderWidth: 2 }}>
                <div style={S.thumbContainer}>
                  <img src={c.thumbnail_base64} alt="" style={S.thumbImg} />
                </div>
                <div style={S.cardFooter}>
                  <span style={{ color: "#DC2626", fontWeight: 600 }}>#{c.object_id}</span>
                  <button style={S.linkBtn} onClick={() => toggleNeg(c.object_id)}>Remove</button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      <div style={{ display: "flex", gap: 12 }}>
        <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>Back</button>
        <button style={S.btnPrimary} onClick={apply} disabled={s.classifierPos.size === 0}>Apply Classifier</button>
        <button style={S.btnSave} onClick={onSave} disabled={s.saving}>
          {s.saving ? "Saving..." : "Save Project"}
        </button>
      </div>
    </section>
  );
}

/* ─── Step 4: Search ────────────────────────────────────────────────────── */

function MapView({ s, visible }) {
  const canvasRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  const [maskData, setMaskData] = useState(null);
  const [mapImageData, setMapImageData] = useState(null);
  const [mapImgIdx, setMapImgIdx] = useState(s.imgIdx);
  const imgRef = useRef(null);
  const maskImgRef = useRef(null);

  const resultScores = new Map();
  for (const r of visible) resultScores.set(r.object_id, r.similarity_score);

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

        const posSet = s.classifierPos;
        const negSet = s.classifierNeg;
        const allHighlight = new Set([...posSet, ...negSet]);
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
                if (hovered === cellId && !posSet.has(cellId) && !negSet.has(cellId) && !resultScores.has(cellId)) {
                  clipCtx.fillStyle = "rgba(200,220,255,0.7)";
                } else if (posSet.has(cellId)) {
                  clipCtx.fillStyle = "rgba(100,160,255,0.9)";
                } else if (negSet.has(cellId)) {
                  clipCtx.fillStyle = "rgba(255,130,130,0.85)";
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
  }, [mapImageData, maskData, hovered, s.classifierPos, s.classifierNeg, visible]);

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

  const hoveredResult = hovered ? visible.find(r => r.object_id === hovered) : null;
  const hoveredIsPos = hovered && s.classifierPos.has(hovered);
  const hoveredIsNeg = hovered && s.classifierNeg.has(hovered);

  return (
    <div style={{ display: "flex", gap: 20 }}>
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

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={S.canvasFrame}>
          <canvas ref={canvasRef}
            style={{ width: "100%", display: "block", cursor: "crosshair" }}
            onMouseMove={e => setHovered(hitTest(e))}
            onMouseLeave={() => setHovered(null)} />
        </div>
      </div>

      <div style={S.sidebar}>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 12, color: "#1A1A1A" }}>Legend</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: 4, background: "rgba(100,160,255,0.9)" }} />
            <span style={{ fontSize: 13, color: "#374151" }}>Positive exemplars</span>
          </div>
          {s.classifierNeg.size > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 16, height: 16, borderRadius: 4, background: "rgba(255,130,130,0.85)" }} />
              <span style={{ fontSize: 13, color: "#374151" }}>Negative exemplars</span>
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: 4, background: "rgba(80,220,100,0.85)" }} />
            <span style={{ fontSize: 13, color: "#374151" }}>Selected ({resultScores.size})</span>
          </div>
        </div>

        {(hoveredResult || hoveredIsPos || hoveredIsNeg || hovered) && (
          <div style={{ padding: 12, background: "#F9FAFB", borderRadius: 8, border: "1px solid #E5E7EB" }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: "#111" }}>Cell #{hovered}</div>
            {hoveredResult && (
              <div style={{ fontSize: 13, color: "#6B7280", marginTop: 4 }}>
                Similarity: <span style={{ fontWeight: 600, color: hoveredResult.similarity_score > 0.8 ? "#16A34A" : hoveredResult.similarity_score > 0.5 ? "#CA8A04" : "#DC2626" }}>
                  {(hoveredResult.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {hoveredIsPos && <div style={{ fontSize: 13, color: "#2563EB", marginTop: 4, fontWeight: 500 }}>Positive exemplar</div>}
            {hoveredIsNeg && <div style={{ fontSize: 13, color: "#DC2626", marginTop: 4, fontWeight: 500 }}>Negative exemplar</div>}
            {!hoveredResult && !hoveredIsPos && !hoveredIsNeg && (
              <div style={{ fontSize: 13, color: "#9CA3AF", marginTop: 4 }}>Not in selection</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StepSearch({ s, set, togglePos, toggleNeg, onSave }) {
  const [posCrops, setPosCrops] = useState([]);
  const [negCrops, setNegCrops] = useState([]);

  useEffect(() => {
    const ids = [...s.classifierPos];
    if (ids.length === 0) setPosCrops([]);
    else api.crops(ids).then(d => setPosCrops(d.crops || []));
  }, [s.classifierPos]);

  useEffect(() => {
    const ids = [...s.classifierNeg];
    if (ids.length === 0) setNegCrops([]);
    else api.crops(ids).then(d => setNegCrops(d.crops || []));
  }, [s.classifierNeg]);

  // Apply threshold + per-image cap
  const aboveThreshold = s.results.filter(r => r.similarity_score >= s.threshold);
  const visible = (() => {
    if (s.perImageCap <= 0) return aboveThreshold;
    const counts = {};
    return aboveThreshold.filter(r => {
      counts[r.image_index] = (counts[r.image_index] || 0) + 1;
      return counts[r.image_index] <= s.perImageCap;
    });
  })();

  const COLS = 5;

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
    </div>
  );

  return (
    <section style={S.card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div>
          <h2 style={S.h2}>Selection Results</h2>
          <p style={{ ...S.muted, margin: 0 }}>
            {s.classifierPos.size} positive{s.classifierNeg.size > 0 ? ` · ${s.classifierNeg.size} negative` : ""} exemplars · {visible.length} cells selected
          </p>
        </div>
        <div style={{ display: "flex", gap: 4, background: "#F3F4F6", borderRadius: 8, padding: 3 }}>
          <button onClick={() => set({ viewMode: "gallery" })}
            style={{ ...S.viewToggleBtn, ...(s.viewMode === "gallery" ? S.viewToggleBtnActive : {}) }}>Gallery</button>
          <button onClick={() => set({ viewMode: "map" })}
            style={{ ...S.viewToggleBtn, ...(s.viewMode === "map" ? S.viewToggleBtnActive : {}) }}>Map</button>
        </div>
      </div>

      {/* Classifier exemplars */}
      <div style={{ display: "flex", gap: 12, margin: "12px 0 20px", flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#2563EB", marginBottom: 6 }}>Positive exemplars</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {posCrops.map(c => (
              <div key={c.object_id} style={{ width: 56, height: 56, borderRadius: 6, overflow: "hidden", border: "2px solid #2563EB", cursor: "pointer", position: "relative" }}
                onClick={() => togglePos(c.object_id)}>
                <img src={c.thumbnail_base64} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              </div>
            ))}
          </div>
        </div>
        {negCrops.length > 0 && (
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#DC2626", marginBottom: 6 }}>Negative exemplars</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {negCrops.map(c => (
                <div key={c.object_id} style={{ width: 56, height: 56, borderRadius: 6, overflow: "hidden", border: "2px solid #DC2626", cursor: "pointer" }}
                  onClick={() => toggleNeg(c.object_id)}>
                  <img src={c.thumbnail_base64} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Controls: threshold + per-image cap + search mode */}
      <div style={S.controlBar}>
        <div style={S.controlGroup}>
          <label style={S.controlLabel}>Min similarity ({s.threshold.toFixed(2)})</label>
          <input type="range" min={0} max={1} step={0.05} value={s.threshold}
            onChange={e => set({ threshold: +e.target.value })} style={S.slider} />
        </div>
        <div style={S.controlGroup}>
          <label style={S.controlLabel}>Max per image ({s.perImageCap})</label>
          <input type="range" min={1} max={100} step={1} value={s.perImageCap}
            onChange={e => set({ perImageCap: +e.target.value })} style={S.slider} />
        </div>
        <div style={S.controlGroup}>
          <label style={S.controlLabel}>Search mode</label>
          <div style={{ display: "flex", gap: 4, background: "#F3F4F6", borderRadius: 8, padding: 3 }}>
            <button onClick={() => set({ searchMode: "weighted" })}
              style={{ ...S.viewToggleBtn, ...(s.searchMode === "weighted" ? S.viewToggleBtnActive : {}) }}>Weighted</button>
            <button onClick={() => set({ searchMode: "centroid" })}
              style={{ ...S.viewToggleBtn, ...(s.searchMode === "centroid" ? S.viewToggleBtnActive : {}) }}>Centroid</button>
          </div>
        </div>
      </div>

      {s.viewMode === "gallery" && (
        <div style={S.gallerySection}>
          <div style={S.gallerySimilar}>
            <h3 style={S.galleryTitleGreen}>Selected Cells ({visible.length})</h3>
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${COLS}, 1fr)`, gap: 10 }}>
              {visible.map(r => <ResultCard key={r.object_id} r={r} />)}
            </div>
            {visible.length === 0 && (
              <p style={S.emptyMsg}>No cells above threshold. Try lowering the minimum similarity.</p>
            )}
          </div>
        </div>
      )}

      {s.viewMode === "map" && <MapView s={s} visible={visible} />}

      <div style={{ marginTop: 24, display: "flex", gap: 12 }}>
        <button style={S.btnSecondary} onClick={() => set({ step: 2 })}>Edit Exemplars</button>
        <button style={S.btnSave} onClick={onSave} disabled={s.saving}>
          {s.saving ? "Saving..." : "Save Project"}
        </button>
      </div>
    </section>
  );
}

/* ─── App Shell ─────────────────────────────────────────────────────────── */

export default function App() {
  const { s, set, togglePos, toggleNeg } = useAppState();
  const steps = ["Load Data", "Exemplars", "Review", "Results"];

  const saveProject = async () => {
    const name = s.projectName || prompt("Project name:", s.datasetId || "my-project");
    if (!name) return;
    set({ saving: true });
    try {
      await api.saveProject(
        name,
        [...s.classifierPos], [...s.classifierNeg],
        s.searchMode, s.threshold, s.perImageCap,
      );
      set({ projectName: name, saving: false });
      alert(`Project "${name}" saved.`);
    } catch (e) {
      set({ saving: false });
      alert(`Save failed: ${e.message}`);
    }
  };

  return (
    <div style={S.root}>
      <header style={S.header}>
        <span style={S.logo}>Microscopy Search</span>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
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
          {s.step >= 2 && (
            <button style={S.btnSecondary} onClick={saveProject} disabled={s.saving}>
              {s.saving ? "Saving..." : "Save Project"}
            </button>
          )}
        </div>
      </header>
      <main style={S.main}>
        {s.step === 1 && <StepLoad s={s} set={set} />}
        {s.step === 2 && <StepSelect s={s} set={set} togglePos={togglePos} toggleNeg={toggleNeg} />}
        {s.step === 3 && <StepReview s={s} set={set} togglePos={togglePos} toggleNeg={toggleNeg} onSave={saveProject} />}
        {s.step === 4 && <StepSearch s={s} set={set} togglePos={togglePos} toggleNeg={toggleNeg} onSave={saveProject} />}
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
  btnSave: {
    background: "#F0FDF4", color: "#16A34A", border: "1px solid #BBF7D0", borderRadius: 8,
    padding: "10px 20px", fontSize: 14, fontWeight: 600, cursor: "pointer",
    transition: "background 0.15s", marginLeft: "auto",
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

  // Modal
  modalOverlay: {
    position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
    background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center",
    justifyContent: "center", zIndex: 1000,
  },
  modalCard: {
    background: "#fff", borderRadius: 16, padding: 32, maxWidth: 520, width: "90%",
    maxHeight: "70vh", overflowY: "auto", boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
  },
  projectRow: {
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "14px 16px", borderRadius: 10, border: "1px solid #E2E4E9",
    marginBottom: 8, cursor: "pointer", transition: "all 0.15s",
    background: "#FAFBFC",
  },
};
