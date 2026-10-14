"""h3_recipe_score.py plus a turn-tolerant face detector (YuNet) and a lip-sync measure. Every field of the original
scorer is computed exactly as before (Haar faces, same 8 sampled frames); the new fields are added next to them.

Old measures (unchanged): face texture, eye-band contrast, fine detail, saturation on 8 frames per clip (all of clip 1,
the last 46 frames of each continuation), audio RMS and spectral flatness per clip, skips, seams.

New per clip:
  faces_found_yunet     sampled frames (of the same 8) where YuNet finds the face (Haar count stays in faces_found)
  yunet_face_frames     frames of the clip (all of them, not just 8) with a tracked YuNet face
  face_texture_yunet / eye_contrast_yunet   the same two measures on the YuNet face box (eye band placed on the
                        detected eyes), so profile and turned faces get numbers where Haar gives None
  lipsync_r, lipsync_lag, lipsync_r0        MAIN lip-sync measure: mouth motion energy (mean |frame difference| of a
                        mouth patch held rigidly to the upper face and re-registered on the band above the lips, so
                        jaw and lips move inside it but head motion mostly does not) against the audio's
                        spectral flux (300-3400 Hz, positive log-spectrum change), Pearson r at the best lag in -6..+6
                        frames; lag > 0 = the sound comes that many frames after the mouth movement; r0 = r at lag 0
  lipsync_open_r, lipsync_open_lag          second measure: mouth opening (share of dark mouth-cavity pixels in the
                        patch) against the 300-3400 Hz RMS envelope; a sharper lag but it breaks when the head turns away
  speech_frames, lipsync_frames             frames with speech (envelope above 10% of the video's peak) / frames used
  Continuations use only their 85 new frames; clip 1 uses its 124. Fewer than 12 speech frames or 24 mouth frames in a
  clip gives None (nothing to line up). Frames without a face are left out (gaps up to 6 frames are interpolated);
  mouth differences are never taken across a shot cut.
New per video: "lipsync" (whole-video numbers: each clip's signals z-scored, then one common lag over the whole video;
  plus the lag curves), "cuts" (shot changes found), "face_detector".
New summary fields (second printed table): lipsync_r/lag, lipsync_open_r/lag (whole video), lipsync_r_vs_no_lora
  (difference to the reference arm), YuNet face-texture / eye-contrast change, Haar vs YuNet face counts.
How to read lip sync (numbers from the 11 test videos, see scorer2/REPORT.md): whole-video lipsync_r was 0.36-0.59 on
  real videos and about 0.13 (max 0.24) with the clips' audio shuffled; one clip needs r > 0.40 to beat another line's
  audio 95% of the time. A gap under ~0.1 between two single videos is within the clip-to-clip noise. A constant offset
  does not lower r (the lag search absorbs it); it shows in the lag: delaying the audio 2 or 4 frames moved both lags
  by exactly 2 or 4. The motion lag curve has a second peak near +4 in every video, so when the peak near 0 weakens the
  reported lipsync_lag jumps to +4; trust lipsync_open_lag (and r0) for the offset in frontal shots.

Usage: python h3_recipe_score2.py DIR NAME[,NAME...] [--out OUTDIR] [--ref NAME]
  NAME = <story>_<arm>; reads its last DIR/NAME_clip<k>.mp4; a single-clip video counts as one clip.
  Writes OUTDIR/scores.json and OUTDIR/sheet_<story>.jpg (OUTDIR defaults to DIR, like the original scorer).
  Each arm is compared with <story>_P0 unless --ref names the no-LoRA arm (e.g. --ref A for the ab_v3 set).
Needs face_detection_yunet_2023mar.onnx next to this script (opencv_zoo); without it (or OpenCV < 4.5.4) the
  tracker falls back to Haar boxes and the lower third of the box as the mouth.
"""
import argparse, json, os, pathlib, re, subprocess, sys
import cv2
import numpy as np

cv2.setNumThreads(4)  # OpenCV has its own thread pool; keep the scorer on a few cores
FF = os.environ.get("FFMPEG", "ffmpeg")  # set FFMPEG when ffmpeg is not on PATH
HERE = pathlib.Path(__file__).resolve().parent
YUNET_MODEL = HERE / "face_detection_yunet_2023mar.onnx"
ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("names")
ap.add_argument("--out", default=None, help="where scores.json and the sheets go (default: DIR)")
ap.add_argument("--ref", default=None, help="the no-LoRA arm every arm is compared with (default: <story>_P0)")
ARGS = ap.parse_args()
D = pathlib.Path(ARGS.dir)
OUT = pathlib.Path(ARGS.out) if ARGS.out else D
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ARGS.names.split(",")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
SEAMS = [124, 209, 294]
LAGS = list(range(-6, 7))
PW, PH = 80, 64  # mouth patch size


# ---------------------------------------------------------------- the original scorer's measures (unchanged)
def frames(path):
    err = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    w, h = map(int, re.search(r"Video: .*?, (\d{2,5})x(\d{2,5})", err).groups())
    raw = subprocess.check_output([FF, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def audio(path):
    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def segments(n, single=False):
    if single:
        return [(0, n)]
    bounds, start = [], 0
    for length in [124] + [85] * 10:
        if start >= n:
            break
        bounds.append((start, min(n, start + length)))
        start += length
    return bounds


def face_metrics(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (grey.shape[1] // 2, grey.shape[0] // 2))
    found = CASCADE.detectMultiScale(small, 1.1, 5, minSize=(40, 40))
    if len(found) == 0:
        return None
    x, y, w, h = [2 * v for v in max(found, key=lambda f: f[2] * f[3])]
    face = cv2.resize(grey[y:y + h, x:x + w], (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    texture = float(np.abs(cv2.Laplacian(face, cv2.CV_32F)).mean())
    eyes = face[28:60, 16:112]  # the band where the eyes sit in a Haar face box
    return texture, float(eyes.std())


def detail(img):
    size = (576, 320) if img.shape[1] > img.shape[0] else (320, 576)
    grey = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(grey, cv2.CV_32F)).mean())


def frame_diffs(f):
    size = (160, 90) if f.shape[2] > f.shape[1] else (90, 160)
    g = np.stack([cv2.cvtColor(cv2.resize(x, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for x in f]).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean((1, 2))
    local = np.array([np.median(d[max(0, i - 12):i + 13]) for i in range(len(d))])
    return d, d / np.maximum(local, 0.3)


def skips(f, diffs=None):
    d, ratio = diffs if diffs is not None else frame_diffs(f)
    jumps = [(i + 1, round(float(d[i]), 1), round(float(ratio[i]), 1))
             for i in range(len(d)) if ratio[i] > 3 and d[i] > 2.0 and i + 1 not in SEAMS]
    return jumps, {s: round(float(ratio[s - 1]), 1) for s in SEAMS if s - 1 < len(d)}


def last_video(name):
    return max(D.glob(f"{name}_clip*.mp4"), key=lambda p: int(re.search(r"_clip(\d+)\.mp4$", p.name).group(1)))


# ---------------------------------------------------------------- face tracking (YuNet, Haar fallback)
def yunet_detector(w, h):
    if not hasattr(cv2, "FaceDetectorYN") or not YUNET_MODEL.exists():
        return None
    try:
        return cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), 0.6, 0.3, 50)
    except cv2.error:
        return None


def track_faces(f):
    """One row per frame: x, y, w, h, 5 landmarks (right eye, left eye, nose, right/left mouth corner), score; NaN when
    no face. The largest face wins; a face under 40% of the video's median face width is treated as a false hit."""
    n, h, w = f.shape[:3]
    L = np.full((n, 15), np.nan)
    det = yunet_detector(w, h)
    if det is not None:
        for i in range(n):
            _, faces = det.detect(f[i])
            if faces is not None and len(faces):
                L[i] = max(faces, key=lambda r: r[2] * r[3])
        kind = "yunet"
    else:  # Haar fallback: landmarks placed on fixed fractions of the box (mouth = lower third)
        for i in range(n):
            grey = cv2.cvtColor(f[i], cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (w // 2, h // 2))
            found = CASCADE.detectMultiScale(small, 1.1, 5, minSize=(40, 40))
            if len(found):
                x, y, bw, bh = [2.0 * v for v in max(found, key=lambda r: r[2] * r[3])]
                fx = lambda a, b: (x + a * bw, y + b * bh)
                L[i] = [x, y, bw, bh, *fx(0.3, 0.4), *fx(0.7, 0.4), *fx(0.5, 0.6), *fx(0.35, 0.8), *fx(0.65, 0.8), 1.0]
        kind = "haar"
    med = np.nanmedian(L[:, 2]) if np.any(~np.isnan(L[:, 2])) else np.nan
    L[L[:, 2] < 0.4 * med] = np.nan
    return L, kind


def cut_runs(n, diffs):
    """Split the video at shot changes: a frame change > 8x the local motion and > 6 grey levels (the porch cut to a
    profile close-up scores 26-51x; ordinary skips 3-4x)."""
    d, ratio = diffs
    cuts = [i + 1 for i in range(len(d)) if ratio[i] > 8 and d[i] > 6.0]
    bounds = [0] + cuts + [n]
    return cuts, [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] > bounds[k]]


def movavg(x, k):
    """centred moving average along axis 0, NaN-aware, shrinking at the ends"""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x)
    n, h = len(x), k // 2
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.where(ok, x, 0), 0)])
    cv = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(ok, 0)])
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    den = cv[hi] - cv[lo]
    return np.where(den > 0, (cs[hi] - cs[lo]) / np.maximum(den, 1), np.nan)


def mouth_track(L, runs):
    """Per frame: smoothed eye midpoint and the slow (1 s median) eye-to-mouth vector. The mouth patch hangs on these,
    so it follows the head but not the jaw: opening and lip motion stay visible inside the patch."""
    n = len(L)
    pts = L[:, 4:14].reshape(-1, 5, 2)
    eye = (pts[:, 0] + pts[:, 1]) / 2
    mouth = (pts[:, 3] + pts[:, 4]) / 2
    eye_s = np.full((n, 2), np.nan); vec = np.full((n, 2), np.nan)
    for a, b in runs:
        e, m, bw = eye[a:b].copy(), mouth[a:b].copy(), L[a:b, 2]
        for i in range(b - a):  # single-frame glitches: eye midpoint far from its neighbours
            nb = [j for j in range(max(0, i - 3), min(b - a, i + 4)) if j != i and not np.isnan(e[j, 0])]
            if not np.isnan(e[i, 0]) and len(nb) >= 2 and np.hypot(*(e[i] - np.median(e[nb], 0))) > 0.25 * np.median(bw[nb]):
                e[i] = np.nan; m[i] = np.nan
        idx = np.arange(b - a); ok = ~np.isnan(e[:, 0])
        if ok.sum() < 2:
            continue
        prev = np.maximum.accumulate(np.where(ok, idx, -10 ** 6))
        nxt = np.minimum.accumulate(np.where(ok, idx, 10 ** 6)[::-1])[::-1]
        fill = ~ok & (nxt - prev <= 7)  # interpolate gaps of up to 6 frames
        for k in range(2):
            e[fill, k] = np.interp(idx[fill], idx[ok], e[ok, k])
            m[fill, k] = np.interp(idx[fill], idx[ok], m[ok, k])
        es = movavg(e, 5)
        es[np.isnan(e[:, 0])] = np.nan
        eye_s[a:b] = es
        dv = m - e
        for i in range(b - a):
            w = dv[max(0, i - 12):i + 13]
            w = w[~np.isnan(w[:, 0])]
            if len(w) and not np.isnan(e[i, 0]):
                vec[a + i] = np.median(w, 0)
    return eye_s, vec


def mouth_patches(f, eye_s, vec):
    """80x64 grey patch around the mouth, 1.3 x 1.04 eye-to-mouth distances, rotated with the face."""
    n = len(f)
    P = np.full((n, PH, PW), np.nan, np.float32)
    for i in range(n):
        if np.isnan(eye_s[i, 0]) or np.isnan(vec[i, 0]):
            continue
        s = float(np.hypot(*vec[i]))
        if s < 4:
            continue
        dn = vec[i] / s; rt = np.array([dn[1], -dn[0]])
        c = eye_s[i] + vec[i] * 1.12
        kx, ky = 1.3 * s / PW, 1.04 * s / PH
        M = np.array([[kx * rt[0], ky * dn[0], c[0] - kx * rt[0] * PW / 2 - ky * dn[0] * PH / 2],
                      [kx * rt[1], ky * dn[1], c[1] - kx * rt[1] * PW / 2 - ky * dn[1] * PH / 2]])
        grey = cv2.cvtColor(f[i], cv2.COLOR_BGR2GRAY)
        P[i] = cv2.warpAffine(grey, M, (PW, PH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return P


def mouth_signals(P, runs):
    """motion: mean |difference| of the (blurred) mouth zone from each frame to the next, never across a cut; the next
    patch is first shifted onto this one by phase correlation on the band above the lips (nose base, upper lip), so
    head motion the landmark smoothing missed does not count as mouth motion.
    opening: share of zone pixels darker than 0.55 x the patch median (the open mouth cavity)."""
    n = len(P)
    ok = ~np.isnan(P[:, 0, 0])
    Pb = np.stack([cv2.GaussianBlur(np.nan_to_num(p), (0, 0), 1.0) for p in P]).astype(np.float32)
    zr, zc = slice(int(0.25 * PH), int(0.85 * PH)), slice(int(0.2 * PW), int(0.8 * PW))
    Z = Pb[:, zr, zc]
    med = np.median(Pb.reshape(n, -1), axis=1)
    opening = np.where(ok, (Z < 0.55 * med[:, None, None]).mean((1, 2)), np.nan)
    band = int(0.35 * PH)
    hann = cv2.createHanningWindow((PW, band), cv2.CV_32F)
    motion = np.full(n, np.nan)  # motion[i] = change from frame i to i+1: centred where the sound of frame i is
    starts = {a for a, b in runs}
    for i in range(n - 1):
        if i + 1 not in starts and ok[i] and ok[i + 1]:
            (dx, dy), _ = cv2.phaseCorrelate(Pb[i][:band], Pb[i + 1][:band], hann)
            M = np.float32([[1, 0, -float(np.clip(dx, -4, 4))], [0, 1, -float(np.clip(dy, -4, 4))]])
            nxt = cv2.warpAffine(Pb[i + 1], M, (PW, PH), borderMode=cv2.BORDER_REPLICATE)
            motion[i] = float(np.abs(nxt[zr, zc] - Z[i]).mean())
    return motion, opening


def audio_features(a, n):
    """Per video frame i (the sound heard while frame i is shown, [i/24, (i+1)/24)): RMS of the 300-3400 Hz band
    and spectral flux (sum of positive log-magnitude changes, 300-3400 Hz, 32 ms windows every 10 ms)."""
    A = np.fft.rfft(a); fr = np.fft.rfftfreq(len(a), 1 / 16000)
    A[(fr < 300) | (fr > 3400)] = 0
    band = np.fft.irfft(A, len(a))
    spf = 16000 / 24
    env = np.full(n, np.nan)
    for i in range(n):
        s0, s1 = int(round(i * spf)), int(round((i + 1) * spf))
        if s1 <= len(band):
            env[i] = np.sqrt(np.mean(band[s0:s1] ** 2))
    win, hop = 512, 160
    flux = np.full(n, np.nan)
    if len(a) > win + hop:
        nfr = 1 + (len(a) - win) // hop
        idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
        X = np.abs(np.fft.rfft(a[idx] * np.hanning(win), axis=1))
        sel = (np.fft.rfftfreq(win, 1 / 16000) >= 300) & (np.fft.rfftfreq(win, 1 / 16000) <= 3400)
        lg = np.log1p(X[:, sel] / (np.percentile(X[:, sel], 99) + 1e-9) * 100)
        fl = np.maximum(lg[1:] - lg[:-1], 0).sum(1)
        which = np.floor((np.arange(1, nfr) * hop + win / 2) / spf).astype(int)
        for i in range(n):
            v = fl[which == i]
            if len(v):
                flux[i] = v.mean()
    return env, flux


def corr(x, y):
    good = ~(np.isnan(x) | np.isnan(y))
    if good.sum() < 24:
        return None
    x, y = x[good], y[good]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def lag_curve(m, e, t):
    """r(L) = corr(m[t], e[t + L]) over frames t; audio may reach outside the clip (it plays continuously)"""
    out = []
    for L in LAGS:
        u = t + L
        keep = (u >= 0) & (u < len(e))
        out.append(corr(m[t[keep]], e[u[keep]]))
    return out


def best(curve):
    vals = [(r, L) for r, L in zip(curve, LAGS) if r is not None]
    if not vals:
        return None, None, None
    r, L = max(vals)
    return round(r, 3), L, (round(curve[LAGS.index(0)], 3) if curve[LAGS.index(0)] is not None else None)


def zscore_by_clip(x, segs):
    z = np.full(len(x), np.nan)
    for a, b in segs:
        v = x[a:b]
        if np.sum(~np.isnan(v)) >= 3 and np.nanstd(v) > 1e-9:
            z[a:b] = (v - np.nanmean(v)) / np.nanstd(v)
    return z


def yunet_face_metrics(img, row):
    """face texture and eye-band contrast on the YuNet box resized to 128 px; the 32-row eye band is centred on the
    detected eyes (the Haar version assumes where they sit)"""
    if row is None or np.isnan(row[0]):
        return None
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = row[:4]
    x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
    x1, y1 = int(min(grey.shape[1], round(x + w))), int(min(grey.shape[0], round(y + h)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    face = cv2.resize(grey[y0:y1, x0:x1], (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    texture = float(np.abs(cv2.Laplacian(face, cv2.CV_32F)).mean())
    ey = int(round(((row[5] + row[7]) / 2 - y0) / (y1 - y0) * 128))
    ey = min(max(ey, 16), 112)
    # skin colour (2026-09-30): the central lower face (below the eyes, inside the cheeks), so redness is measured on
    # the skin itself rather than on the whole frame
    fw, fh = x1 - x0, y1 - y0
    skin = img[y0 + int(fh * 0.45):y1 - int(fh * 0.1), x0 + fw // 4:x1 - fw // 4]
    sat = float(cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)[..., 1].mean())
    red = float(cv2.cvtColor(skin, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32).mean()) - 128.0  # a*: + is red
    return texture, float(face[ey - 16:ey + 16, 16:112].std()), float(fh), sat, red


# ---------------------------------------------------------------- scoring
def score(path):
    f = frames(path)
    a = audio(path)
    n = len(f)
    single = path.name.endswith("_clip1.mp4")
    segs = segments(n, single)
    diffs = frame_diffs(f)
    L, kind = track_faces(f)
    cuts, runs = cut_runs(n, diffs)
    eye_s, vec = mouth_track(L, runs)
    motion, opening = mouth_signals(mouth_patches(f, eye_s, vec), runs)
    env, flux = audio_features(a, n)
    speech = env > 0.1 * np.nanpercentile(env, 99)
    rows = []
    for i, (s, e) in enumerate(segs):
        pick = np.linspace(s + (39 if i else 0), e - 1, 8).astype(int)
        faces = [m for m in (face_metrics(f[p]) for p in pick) if m]
        yunet = kind == "yunet"  # without YuNet the YuNet fields stay None (the Haar track still feeds lip sync)
        yfaces = [m for m in (yunet_face_metrics(f[p], L[p]) for p in pick) if m] if yunet else []
        seg = a[int(s / 24 * 16000):int(e / 24 * 16000)]
        spec = np.abs(np.fft.rfft(seg[: len(seg) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024), axis=1)) + 1e-9
        t = np.arange(s, e)
        n_speech = int(speech[s:e].sum())
        n_used = int((~np.isnan(motion[s:e]) & ~np.isnan(flux[s:e])).sum())
        enough = n_speech >= 12 and n_used >= 24
        r, lag, r0 = best(lag_curve(motion, flux, t)) if enough else (None, None, None)
        ro, lago, _ = best(lag_curve(opening, env, t)) if enough else (None, None, None)
        rows.append({"clip": i + 1, "frames": f"{s}-{e - 1}", "faces_found": len(faces),
                     "face_texture": round(float(np.mean([m[0] for m in faces])), 2) if faces else None,
                     "eye_contrast": round(float(np.mean([m[1] for m in faces])), 2) if faces else None,
                     "frame_detail": round(float(np.mean([detail(f[p]) for p in pick])), 2),
                     "saturation": round(float(np.mean([cv2.cvtColor(f[p], cv2.COLOR_BGR2HSV)[..., 1].mean() for p in pick])), 1),
                     "audio_rms": round(float(np.sqrt((seg ** 2).mean())), 4) if len(seg) else None,
                     "audio_flatness": round(float(np.mean(np.exp(np.log(spec).mean(1)) / spec.mean(1))), 4) if len(spec) else None,
                     "faces_found_yunet": len(yfaces) if yunet else None,
                     "yunet_face_frames": int((~np.isnan(L[s:e, 0])).sum()) if yunet else None,
                     "face_texture_yunet": round(float(np.mean([m[0] for m in yfaces])), 2) if yfaces else None,
                     "eye_contrast_yunet": round(float(np.mean([m[1] for m in yfaces])), 2) if yfaces else None,
                     "face_h_yunet": round(float(np.median([m[2] for m in yfaces])), 1) if yfaces else None,
                     "skin_saturation": round(float(np.mean([m[3] for m in yfaces])), 1) if yfaces else None,
                     "skin_redness": round(float(np.mean([m[4] for m in yfaces])), 2) if yfaces else None,
                     "_ht": [(m[2], m[0]) for m in yfaces],
                     "lipsync_r": r, "lipsync_lag": lag, "lipsync_r0": r0,
                     "lipsync_open_r": ro, "lipsync_open_lag": lago,
                     "speech_frames": n_speech, "lipsync_frames": n_used})
    ref_h = float(np.median([h for h, _ in rows[0]["_ht"]])) if rows and rows[0]["_ht"] else None
    for row in rows:
        ok = [t for h, t in row.pop("_ht") if ref_h and 0.8 * ref_h <= h <= 1.25 * ref_h]
        row["face_texture_sized"] = round(float(np.mean(ok)), 2) if len(ok) >= 2 else None
        row["sized_frames"] = len(ok)
    # whole video: z-score each clip's signals (clip-to-clip level changes are not sync), one common lag
    use = [(s, e) for (s, e), row in zip(segs, rows) if row["lipsync_r"] is not None]
    t = np.concatenate([np.arange(s, e) for s, e in use]) if use else np.arange(0)
    whole = {}
    for key, (m, au) in {"": (motion, flux), "open_": (opening, env)}.items():
        curve = lag_curve(zscore_by_clip(m, use), zscore_by_clip(au, use), t) if len(t) else [None] * len(LAGS)
        r, lag, r0 = best(curve)
        whole.update({f"{key}r": r, f"{key}lag": lag, f"{key}r0": r0,
                      f"{key}curve": [None if v is None else round(v, 3) for v in curve]})
    whole["clips_used"] = len(use)
    jumps, seams = skips(f, diffs)
    return f, {"clips": rows, "skips": jumps, "seams": seams, "cuts": cuts, "face_detector": kind,
               "lipsync": {"lags": LAGS, **whole}}


def change(rows, key):
    first, last = rows[0].get(key), rows[-1].get(key)
    return round(100 * (last / first - 1)) if first and last else None


def change_late(rows, key, window=3):
    """change from clip 1 to the latest of the last `window` clips that has a value; (pct, clip) or (None, None)"""
    first = rows[0].get(key)
    for r in reversed(rows[-window:]):
        if first and r.get(key) is not None:
            return round(100 * (r[key] / first - 1)), r["clip"]
    return None, None


def later_mean(rows, key):
    values = [r[key] for r in (rows[1:] or rows) if r.get(key) is not None]  # a single clip compares itself
    return float(np.mean(values)) if values else None


results, sheets = {}, {}
for name in NAMES:
    path = last_video(name)
    f, result = score(path)
    results[name] = result
    portrait = f.shape[1] > f.shape[2]
    tiles = []
    for k, (s, e) in enumerate(segments(len(f), path.name.endswith("_clip1.mp4"))):
        img = cv2.resize(f[e - 1], (240, 432) if portrait else (432, 240))
        cv2.putText(img, f"{name} clip {k + 1}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    sheets.setdefault(name.split("_")[0], []).append(np.hstack(tiles))
    del f
for story, rows in sheets.items():
    cv2.imwrite(str(OUT / f"sheet_{story}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

summary, new, skin = {}, {}, {}
for name, result in results.items():
    rows = result["clips"]
    ref_name = ARGS.ref or name.split("_")[0] + "_P0"
    ref = results.get(ref_name, {}).get("clips")
    rms, flat = later_mean(rows, "audio_rms"), later_mean(rows, "audio_flatness")
    summary[name] = {
        "face_texture_change_pct": change(rows, "face_texture"), "fine_detail_change_pct": change(rows, "frame_detail"),
        "eye_contrast_change_pct": change(rows, "eye_contrast"), "saturation_change_pct": change(rows, "saturation"),
        "audio_loudness_vs_no_lora_pct": round(100 * rms / later_mean(ref, "audio_rms")) if ref and rms else None,
        "audio_hiss_vs_no_lora_pct": round(100 * (flat / later_mean(ref, "audio_flatness") - 1)) if ref and flat else None,
        "audio_loudness_clips_2_4_vs_clip_1_pct": round(100 * rms / rows[0]["audio_rms"]) if rms and rows[0]["audio_rms"] else None,
        "skips": len(result["skips"]), "worst_seam_x_local_motion": max(result["seams"].values()) if result["seams"] else None}
    ls = result["lipsync"]
    ref_ls = results.get(ref_name, {}).get("lipsync", {})
    new[name] = {
        "lipsync_r": ls["r"], "lipsync_lag": ls["lag"], "lipsync_open_r": ls["open_r"], "lipsync_open_lag": ls["open_lag"],
        "lipsync_r_vs_no_lora": round(ls["r"] - ref_ls["r"], 3) if ref_ls.get("r") is not None and ls["r"] is not None else None,
        "face_texture_yunet_change_pct": change(rows, "face_texture_yunet"),
        "eye_contrast_yunet_change_pct": change(rows, "eye_contrast_yunet"),
        "faces_found_haar": f"{sum(r['faces_found'] for r in rows)}/{8 * len(rows)}",
        "faces_found_yunet": f"{sum(r['faces_found_yunet'] for r in rows)}/{8 * len(rows)}" if result["face_detector"] == "yunet" else None}
    sized, sized_clip = change_late(rows, "face_texture_sized")
    red0, red1 = rows[0].get("skin_redness"), rows[-1].get("skin_redness")
    skin[name] = {"face_texture_sized_change_pct": sized, "sized_clip": sized_clip,
                  "face_size_change_pct": change(rows, "face_h_yunet"),
                  "skin_saturation_change_pct": change(rows, "skin_saturation"),
                  "skin_redness_change": round(red1 - red0, 2) if red0 is not None and red1 is not None else None}
    result["summary"] = {**summary[name], **new[name], **skin[name]}
# merge (2026-09-25: each call used to replace scores.json and drop the chains scored earlier)
_prev = {}
if (OUT / "scores.json").exists():
    try:
        _prev = json.loads((OUT / "scores.json").read_text())
    except ValueError:
        _prev = {}
_tmp = OUT / "scores.json.tmp"
_tmp.write_text(json.dumps({**_prev, **results}, indent=1))
os.replace(_tmp, OUT / "scores.json")
keys = list(next(iter(summary.values())).keys())
print("arm | " + " | ".join(keys))
for name, s in summary.items():
    print(f"{name} | " + " | ".join(str(s[k]) for k in keys))
for name, result in results.items():
    if result["skips"]:
        print(f"{name} skips (frame, size, x local motion): {result['skips']}")
keys = list(next(iter(new.values())).keys())
print("\narm | " + " | ".join(keys))
for name, s in new.items():
    print(f"{name} | " + " | ".join(str(s[k]) for k in keys))
keys = list(next(iter(skin.values())).keys())
print("\nface burn at the same face size + skin colour | " + " | ".join(keys))
for name, s in skin.items():
    print(f"{name} | " + " | ".join(str(s[k]) for k in keys))
print("\nlip sync per clip, r@lag (motion vs flux) / open r@lag (opening vs envelope); lag > 0 = sound after the mouth:")
for name, result in results.items():
    cells = [f"c{r['clip']} {r['lipsync_r']}@{r['lipsync_lag']} / {r['lipsync_open_r']}@{r['lipsync_open_lag']}" for r in result["clips"]]
    print(f"{name}: " + " | ".join(cells) + (f"  (shot cuts at {result['cuts']})" if result["cuts"] else ""))
