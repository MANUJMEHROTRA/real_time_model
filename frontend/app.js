import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const SMOOTHING = 0.45; // 0 = frozen, 1 = raw landmarks (jittery)
const MIN_VISIBILITY = 0.5;
const TORSO_TO_SHOULDER_RATIO = 1.6; // used to guess hips when they're out of frame
const COUNTDOWN_SECONDS = 3;
const UPLOAD_MAX_SIDE = 1280;

// MediaPipe pose landmark indices.
const LM = { nose: 0, left_shoulder: 11, right_shoulder: 12, left_hip: 23, right_hip: 24,
             left_knee: 25, right_knee: 26, left_ankle: 27, right_ankle: 28 };

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const state = {
  garments: [],
  current: null, // { garment, img }
  images: new Map(),
  smoothed: null, // { anchorName: [x, y] } in video pixels
  busy: false,
  lastVideoTime: -1,
};

// ---------------------------------------------------------------- setup

async function main() {
  try {
    const [landmarker] = await Promise.all([loadPose(), startCamera(), loadCatalog()]);
    $("status").textContent = "";
    $("capture").disabled = false;
    requestAnimationFrame((t) => loop(landmarker, t));
  } catch (err) {
    console.error(err);
    $("status").textContent = `⚠️ ${err.message || err}`;
  }
}

async function loadPose() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  try {
    return await PoseLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch (e) {
    console.warn("GPU delegate unavailable, falling back to CPU", e);
    return PoseLandmarker.createFromOptions(fileset, opts("CPU"));
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function loadCatalog() {
  const res = await fetch("/api/catalog");
  if (!res.ok) throw new Error("Could not load catalog");
  state.garments = await res.json();

  const carousel = $("carousel");
  await Promise.all(state.garments.map(async (g, i) => {
    const img = new Image();
    img.src = g.image;
    await img.decode();
    state.images.set(g.id, img);

    const btn = document.createElement("button");
    btn.className = "item";
    btn.dataset.id = g.id;
    btn.title = g.name;
    btn.appendChild(img.cloneNode());
    btn.onclick = () => selectGarment(g.id);
    carousel.appendChild(btn);
  }));
  // Promise.all resolves out of order; restore catalog order.
  for (const g of state.garments) carousel.appendChild(carousel.querySelector(`[data-id="${g.id}"]`));
  if (state.garments.length) selectGarment(state.garments[0].id);
}

function selectGarment(id) {
  const garment = state.garments.find((g) => g.id === id);
  state.current = { garment, img: state.images.get(id) };
  $("garment-name").textContent = garment.name;
  document.querySelectorAll(".item").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
}

// ---------------------------------------------------------------- render loop

let fpsFrames = 0, fpsStart = performance.now();

function loop(landmarker, now) {
  if (video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, now);
    updateAnchors(result.landmarks?.[0]);
    render();
    tickFps(now);
  }
  requestAnimationFrame((t) => loop(landmarker, t));
}

function tickFps(now) {
  fpsFrames++;
  if (now - fpsStart > 1000) {
    $("fps").textContent = `${Math.round((fpsFrames * 1000) / (now - fpsStart))} fps`;
    fpsFrames = 0;
    fpsStart = now;
  }
}

function render() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (state.current && state.smoothed) drawGarment(ctx, state.current, state.smoothed);
}

// ---------------------------------------------------------------- pose → garment anchors

function updateAnchors(landmarks) {
  const needed = state.current ? Object.keys(state.current.garment.overlay.anchors) : [];
  const points = landmarks ? resolveAnchors(landmarks, needed) : null;
  $("hint").classList.toggle("hidden", !!points);

  if (!points) {
    state.smoothed = null;
    return;
  }
  if (!state.smoothed || Object.keys(points).some((k) => !state.smoothed[k])) {
    state.smoothed = points;
    return;
  }
  for (const [k, [x, y]] of Object.entries(points)) {
    const [sx, sy] = state.smoothed[k];
    state.smoothed[k] = [sx + (x - sx) * SMOOTHING, sy + (y - sy) * SMOOTHING];
  }
}

/** Map anchor names (e.g. "left_shoulder", "mid_hip") to pixel positions, or null if not visible. */
function resolveAnchors(landmarks, names) {
  const W = canvas.width, H = canvas.height;
  const px = (i) => [landmarks[i].x * W, landmarks[i].y * H];
  const visible = (i) => (landmarks[i].visibility ?? 1) >= MIN_VISIBILITY;

  const ls = LM.left_shoulder, rs = LM.right_shoulder;
  const shouldersOk = visible(ls) && visible(rs);

  // When the customer is close to the camera the hips drop out of frame; extrapolate them
  // perpendicular to the shoulder line so tops still fit.
  const guessedHip = (shoulderIdx) => {
    const [lx, ly] = px(ls), [rx, ry] = px(rs);
    const len = Math.hypot(lx - rx, ly - ry) * TORSO_TO_SHOULDER_RATIO;
    const [dx, dy] = [-(ly - ry), lx - rx].map((v) => v / Math.hypot(lx - rx, ly - ry));
    const [sx, sy] = px(shoulderIdx);
    return [sx + dx * len, sy + dy * len];
  };

  const point = (name) => {
    if (name.startsWith("mid_")) {
      const base = name.slice(4);
      const a = point(`left_${base}`), b = point(`right_${base}`);
      return a && b ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] : null;
    }
    const i = LM[name];
    if (i === undefined) throw new Error(`Unknown anchor '${name}'`);
    if (visible(i)) return px(i);
    if (name === "left_hip" && shouldersOk) return guessedHip(ls);
    if (name === "right_hip" && shouldersOk) return guessedHip(rs);
    return null;
  };

  const out = {};
  for (const n of names) {
    const p = point(n);
    if (!p) return null;
    out[n] = p;
  }
  return out;
}

/** Fit the garment with the affine transform that maps its 3 anchor points onto the body. */
function drawGarment(c, { garment, img }, body) {
  const names = Object.keys(garment.overlay.anchors).slice(0, 3);
  const src = names.map((n) => {
    const [u, v] = garment.overlay.anchors[n];
    return [u * img.naturalWidth, v * img.naturalHeight];
  });
  const dst = names.map((n) => body[n]);
  const m = affineFrom3Points(src, dst);
  if (!m) return;
  c.save();
  c.setTransform(...m);
  c.drawImage(img, 0, 0);
  c.restore();
}

/** Solve for [a,b,c,d,e,f] with x' = a·x + c·y + e, y' = b·x + d·y + f (canvas setTransform order). */
function affineFrom3Points([[x1, y1], [x2, y2], [x3, y3]], [[u1, v1], [u2, v2], [u3, v3]]) {
  const det = x1 * (y2 - y3) - y1 * (x2 - x3) + (x2 * y3 - x3 * y2);
  if (Math.abs(det) < 1e-6) return null;
  const solve = (r1, r2, r3) => [
    (r1 * (y2 - y3) - y1 * (r2 - r3) + (r2 * y3 - r3 * y2)) / det,
    (x1 * (r2 - r3) - r1 * (x2 - x3) + (x2 * r3 - x3 * r2)) / det,
    (x1 * (y2 * r3 - y3 * r2) - y1 * (x2 * r3 - x3 * r2) + r1 * (x2 * y3 - x3 * y2)) / det,
  ];
  const [a, c, e] = solve(u1, u2, u3);
  const [b, d, f] = solve(v1, v2, v3);
  return [a, b, c, d, e, f];
}

// ---------------------------------------------------------------- capture → photoreal try-on

async function capture() {
  if (state.busy || !state.current) return;
  state.busy = true;
  $("capture").disabled = true;

  for (let n = COUNTDOWN_SECONDS; n > 0; n--) {
    $("countdown").textContent = n;
    $("countdown").classList.remove("hidden");
    await sleep(1000);
  }
  $("countdown").classList.add("hidden");
  flash();

  const crop = portraitCrop();
  const person = frameToDataURL(video, crop);
  const preview = frameToDataURL(canvas, crop);
  showResult({ loading: true });

  try {
    const res = await fetch("/api/tryon", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ garment_id: state.current.garment.id, person_image: person, preview_image: preview }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
    console.info(`Try-on via ${data.backend} in ${data.elapsed_ms} ms`);
    showResult({ image: data.image });
  } catch (err) {
    showResult({ error: `Sorry, something went wrong: ${err.message}` });
  }
}

/** Try-on models are trained on 3:4 portrait photos; crop the (usually landscape) frame around the customer. */
function portraitCrop() {
  const W = canvas.width, H = canvas.height;
  const w = Math.min(W, Math.round((H * 3) / 4));
  const body = state.smoothed;
  const cx = body?.left_shoulder && body?.right_shoulder
    ? (body.left_shoulder[0] + body.right_shoulder[0]) / 2
    : W / 2;
  const x = Math.round(Math.min(Math.max(cx - w / 2, 0), W - w));
  return { x, y: 0, w, h: H };
}

function frameToDataURL(source, { x, y, w, h }) {
  const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(w, h));
  const off = document.createElement("canvas");
  off.width = Math.round(w * scale);
  off.height = Math.round(h * scale);
  off.getContext("2d").drawImage(source, x, y, w, h, 0, 0, off.width, off.height);
  return off.toDataURL("image/jpeg", 0.9);
}

function showResult({ loading = false, image = null, error = null }) {
  $("result").classList.remove("hidden");
  $("result-loading").classList.toggle("hidden", !loading);
  $("result-img").classList.toggle("hidden", !image);
  $("result-error").classList.toggle("hidden", !error);
  $("download").classList.toggle("hidden", !image);
  $("close").disabled = loading;
  if (image) {
    $("result-img").src = image;
    $("download").href = image;
  }
  $("result-error").textContent = error || "";
}

function closeResult() {
  $("result").classList.add("hidden");
  $("result-img").removeAttribute("src");
  state.busy = false;
  $("capture").disabled = false;
}

function flash() {
  const el = $("flash");
  el.classList.add("on");
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.remove("on")));
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- input

$("capture").onclick = capture;
$("close").onclick = closeResult;

// Keyboard / presenter-remote control for kiosks: ←/→ to browse, space/enter to capture, esc to close.
window.addEventListener("keydown", (e) => {
  if (!$("result").classList.contains("hidden")) {
    if (e.key === "Escape" && !$("close").disabled) closeResult();
    return;
  }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); capture(); }
  if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && state.current) {
    const i = state.garments.indexOf(state.current.garment);
    const next = (i + (e.key === "ArrowRight" ? 1 : -1) + state.garments.length) % state.garments.length;
    selectGarment(state.garments[next].id);
  }
});

main();
