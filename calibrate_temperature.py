"""Fit a confidence-calibration temperature for the cls ensemble.

Standard temperature scaling (Guo et al. 2017): find the scalar T that
minimizes NLL of the ensemble's per-pixel 4-class softmax on a held-out
xBD split, evaluated on ground-truth building pixels. The plugin then
divides cls logits by T before softmax (see BuildingDamageEngine.
_cls_softmax), which leaves every argmax decision unchanged but makes the
`confidence` attribute mean what analysts assume it means — so "review
buildings below confidence X" becomes a statistically meaningful triage.

Dev tool — runs OUTSIDE QGIS. Requires: numpy, onnxruntime, Pillow.

Usage:
    python calibrate_temperature.py --val-dir <xbd_split_dir> [options]

Expected xBD layout under --val-dir:
    images/<scene>_pre_disaster.png
    images/<scene>_post_disaster.png
    targets/<scene>_post_disaster_target.png    (uint8 labels 0..4)

Output: calibration.json next to this script, e.g.
    {"cls_temperature": 1.32, ...}
Ship it in the plugin folder; the engine picks it up on load_model().
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
ENCODER_STRIDE = 32

# Same member set load_model() discovers, in the same order.
CLS_CANDIDATES = ['cls_best_fp16.onnx', 'cls_m2_fp16.onnx',
                  'cls_best.onnx', 'cls_m2.onnx']


def log(msg):
    print(msg, flush=True)


def load_sessions(model_dir):
    import onnxruntime as ort
    sessions = []
    seen_stems = set()
    for name in CLS_CANDIDATES:
        stem = name.replace('_fp16', '').replace('.onnx', '')
        path = os.path.join(model_dir, name)
        if stem in seen_stems or not os.path.exists(path):
            continue
        sess = ort.InferenceSession(path,
                                    providers=['CPUExecutionProvider'])
        dtype = (np.float16 if 'float16' in str(sess.get_inputs()[0].type)
                 else np.float32)
        sessions.append((name, sess, dtype))
        seen_stems.add(stem)
        log(f"  loaded {name} ({np.dtype(dtype).name} input)")
    return sessions


def normalize(img_u8):
    return ((img_u8.astype(np.float32) / 255.0) - MEAN) / STD


def pad_to_stride(arr):
    h, w = arr.shape[:2]
    hp = ((h + ENCODER_STRIDE - 1) // ENCODER_STRIDE) * ENCODER_STRIDE
    wp = ((w + ENCODER_STRIDE - 1) // ENCODER_STRIDE) * ENCODER_STRIDE
    if hp == h and wp == w:
        return arr
    return np.pad(arr, ((0, hp - h), (0, wp - w), (0, 0)), mode='reflect')


def member_logits(sessions, pre_u8, post_u8):
    """Run each cls member on one pre/post pair. Returns (M, 4, H, W) fp32."""
    h, w = pre_u8.shape[:2]
    pre = pad_to_stride(normalize(pre_u8)).transpose(2, 0, 1)[None]
    post = pad_to_stride(normalize(post_u8)).transpose(2, 0, 1)[None]
    out = []
    for _name, sess, dtype in sessions:
        pair = np.ascontiguousarray(
            np.concatenate([pre, post], axis=1), dtype=dtype)
        logits = sess.run(['logits'], {'input': pair})[0][0]
        out.append(logits[:, :h, :w].astype(np.float32))
    return np.stack(out, axis=0)


def find_pairs(val_dir):
    img_dir = os.path.join(val_dir, 'images')
    tgt_dir = os.path.join(val_dir, 'targets')
    pairs = []
    for pre_path in sorted(glob.glob(
            os.path.join(img_dir, '*_pre_disaster.png'))):
        stem = os.path.basename(pre_path)[:-len('_pre_disaster.png')]
        post_path = os.path.join(img_dir, f'{stem}_post_disaster.png')
        tgt_path = os.path.join(tgt_dir, f'{stem}_post_disaster_target.png')
        if os.path.exists(post_path) and os.path.exists(tgt_path):
            pairs.append((stem, pre_path, post_path, tgt_path))
    return pairs


def ensemble_probs(logit_samples, t):
    """logit_samples (M, N, 4) -> mean-of-softmax probs (N, 4) at temp t.
    Mirrors the engine: T applied per member, before the probability mean."""
    z = logit_samples / t
    z = z - z.max(axis=2, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=2, keepdims=True)
    return p.mean(axis=0)


def nll(logit_samples, labels, t):
    p = ensemble_probs(logit_samples, t)
    return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())


def ece(logit_samples, labels, t, n_bins=15):
    """Expected calibration error (equal-width bins on max-prob)."""
    p = ensemble_probs(logit_samples, t)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        err += (n / total) * abs(correct[sel].mean() - conf[sel].mean())
    return float(err)


def fit_temperature(logit_samples, labels, t_lo=0.25, t_hi=4.0):
    """Coarse log-grid scan + golden-section refinement (no scipy needed)."""
    grid = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), 25))
    losses = [nll(logit_samples, labels, t) for t in grid]
    i = int(np.argmin(losses))
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]

    invphi = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = (nll(logit_samples, labels, np.exp(c)),
              nll(logit_samples, labels, np.exp(d)))
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = nll(logit_samples, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = nll(logit_samples, labels, np.exp(d))
        if abs(b - a) < 1e-4:
            break
    return float(np.exp((a + b) / 2))


def main():
    ap = argparse.ArgumentParser(
        description="Fit cls-ensemble temperature on a held-out xBD split.")
    ap.add_argument('--val-dir', required=True,
                    help="xBD split dir containing images/ + targets/")
    ap.add_argument('--model-dir', default=os.path.dirname(
        os.path.abspath(__file__)),
        help="Directory holding the cls .onnx files (default: script dir)")
    ap.add_argument('--max-pairs', type=int, default=200,
                    help="Cap on validation pairs (default 200)")
    ap.add_argument('--pixels-per-image', type=int, default=20000,
                    help="Building pixels sampled per image (default 20000)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', default=None,
                    help="Output path (default: <model-dir>/calibration.json)")
    args = ap.parse_args()

    from PIL import Image

    log(f"Loading cls models from {args.model_dir} ...")
    sessions = load_sessions(args.model_dir)
    if not sessions:
        log("ERROR: no cls .onnx files found. Run from the plugin folder "
            "or pass --model-dir.")
        return 1

    pairs = find_pairs(args.val_dir)
    if not pairs:
        log(f"ERROR: no xBD pre/post/target triplets under {args.val_dir} "
            f"(expected images/*_pre_disaster.png + targets/).")
        return 1
    rng = np.random.default_rng(args.seed)
    if len(pairs) > args.max_pairs:
        idx = rng.choice(len(pairs), size=args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    log(f"Calibrating on {len(pairs)} validation pairs, "
        f"{len(sessions)} ensemble member(s).")

    sampled_logits = []     # each (M, k, 4)
    sampled_labels = []     # each (k,)
    for n_done, (stem, pre_path, post_path, tgt_path) in enumerate(pairs, 1):
        pre = np.asarray(Image.open(pre_path).convert('RGB'))
        post = np.asarray(Image.open(post_path).convert('RGB'))
        target = np.asarray(Image.open(tgt_path))
        if target.ndim == 3:
            target = target[..., 0]
        ys, xs = np.nonzero((target >= 1) & (target <= 4))
        if len(ys) == 0:
            continue
        if len(ys) > args.pixels_per_image:
            sel = rng.choice(len(ys), size=args.pixels_per_image,
                             replace=False)
            ys, xs = ys[sel], xs[sel]
        logits = member_logits(sessions, pre, post)       # (M, 4, H, W)
        sampled_logits.append(
            logits[:, :, ys, xs].transpose(0, 2, 1))      # (M, k, 4)
        sampled_labels.append(target[ys, xs].astype(np.int64) - 1)
        if n_done % 10 == 0 or n_done == len(pairs):
            log(f"  {n_done}/{len(pairs)} pairs "
                f"({sum(len(l) for l in sampled_labels):,} pixels)")

    if not sampled_labels:
        log("ERROR: no building pixels found in the targets.")
        return 1

    logit_samples = np.concatenate(sampled_logits, axis=1)   # (M, N, 4)
    labels = np.concatenate(sampled_labels)                  # (N,)
    log(f"Fitting temperature on {len(labels):,} building pixels ...")

    t_fit = fit_temperature(logit_samples, labels)
    stats = {
        'nll_before': nll(logit_samples, labels, 1.0),
        'nll_after': nll(logit_samples, labels, t_fit),
        'ece_before': ece(logit_samples, labels, 1.0),
        'ece_after': ece(logit_samples, labels, t_fit),
    }
    log(f"\n  cls_temperature = {t_fit:.4f}")
    log(f"  NLL  {stats['nll_before']:.4f} -> {stats['nll_after']:.4f}")
    log(f"  ECE  {stats['ece_before']:.4f} -> {stats['ece_after']:.4f}")

    out_path = args.output or os.path.join(args.model_dir, 'calibration.json')
    payload = {
        'cls_temperature': round(t_fit, 4),
        'method': 'temperature scaling (Guo et al. 2017), per-member T '
                  'before ensemble probability mean',
        'fitted_on': os.path.abspath(args.val_dir),
        'n_pairs': len(pairs),
        'n_pixels': int(len(labels)),
        'members': [name for name, _s, _d in sessions],
        'metrics': {k: round(v, 5) for k, v in stats.items()},
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    log(f"\nWrote {out_path} — ship it next to the .onnx weights and the "
        f"plugin applies it automatically.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
