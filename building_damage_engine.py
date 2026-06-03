"""AI building damage detection engine. Outputs damage-classified building polygons."""
import numpy as np
import onnxruntime as ort
from qgis.core import (QgsGeometry, QgsPointXY, QgsMessageLog, Qgis,
                       QgsRasterLayer, QgsProject)
from scipy import ndimage
from PIL import Image as PILImage
import os
import tempfile
import time
import json


# Backend provider chain. ONNX Runtime picks the first available; the
# engine logs which one is active. CUDA needs onnxruntime-gpu wheel,
# DirectML needs onnxruntime-directml (Windows GPU without CUDA Toolkit),
# CoreML is auto-available on macOS, CPU is the universal fallback.
_PROVIDER_PREFERENCE = [
    'CUDAExecutionProvider',
    'DmlExecutionProvider',
    'CoreMLExecutionProvider',
    'CPUExecutionProvider',
]

_PROVIDER_LABELS = {
    'CUDAExecutionProvider':   'CUDA',
    'DmlExecutionProvider':    'DirectML',
    'CoreMLExecutionProvider': 'CoreML',
    'CPUExecutionProvider':    'CPU',
}


def _select_providers():
    """Pick the best provider chain available on this machine. Returns the
    full chain ordered by preference (each session falls back through it
    until one succeeds). Always includes CPU at the end."""
    available = set(ort.get_available_providers())
    chain = [p for p in _PROVIDER_PREFERENCE if p in available]
    if 'CPUExecutionProvider' not in chain:
        chain.append('CPUExecutionProvider')
    return chain


def _np_sigmoid(x):
    """Numerically stable sigmoid for numpy arrays."""
    # Split positive/negative branches so neither np.exp argument grows
    # past ~88 (the fp32 overflow threshold for np.exp).
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def _np_softmax(x, axis=0):
    """Numerically stable softmax along `axis`."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class _DeviceShim:
    """Mimics enough of `torch.device` for the rest of the engine (which
    reads `.type`) without needing torch installed at runtime."""
    __slots__ = ('type', 'provider')

    def __init__(self, provider):
        self.provider = provider
        # 'cpu' if CPU-only, 'cuda' for any GPU provider. This keeps the
        # existing `self.device.type == 'cpu'` checks working untouched.
        self.type = 'cpu' if provider == 'CPUExecutionProvider' else 'cuda'

    def __str__(self):
        return _PROVIDER_LABELS.get(self.provider, self.provider)


class _InferenceProfiler:
    """Per-stage timing, throughput, and CPU memory tracking for one detect()
    call. Output feeds the QGIS log summary and the optional JSON sidecar.
    GPU peak memory is not tracked because ONNX Runtime's per-provider
    memory accounting is not exposed through the public API."""

    _psutil_hint_shown = False    # one-time install hint per QGIS session

    def __init__(self, device):
        self.device = device
        # GPU memory tracking is unavailable through onnxruntime, so we
        # report only CPU metrics regardless of the active provider.
        self.use_cuda = False
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            self._psutil = None
            if not _InferenceProfiler._psutil_hint_shown:
                try:
                    py = os.path.basename(__import__('sys').executable)
                    QgsMessageLog.logMessage(
                        f"[Damage AI] psutil not installed — CPU memory "
                        f"tracking disabled. To enable, run from cmd.exe:\n"
                        f"  \"<QGIS_python_path>\\{py}\" -m pip install psutil",
                        'Damage AI', level=Qgis.Info)
                except Exception:
                    pass
                _InferenceProfiler._psutil_hint_shown = True
        self.reset()

    def reset(self):
        self.stages = {}      # stage name -> list of durations (sec)
        self.counts = {}      # event name -> count
        self.scalars = {}     # name -> any scalar (img dims, n_models, ...)
        self.start_time = None
        self.end_time = None
        self._cpu_baseline = self._cpu_rss()

    def _cpu_rss(self):
        """Current process RSS in bytes, or 0 if psutil isn't available."""
        if self._psutil is None:
            return 0
        try:
            return self._psutil.Process().memory_info().rss
        except Exception:
            return 0

    def _cuda_sync(self):
        # No-op under ONNX Runtime: ORT auto-synchronizes per session.run().
        pass

    def start_run(self):
        self._cuda_sync()
        self.start_time = time.perf_counter()

    def end_run(self):
        self._cuda_sync()
        self.end_time = time.perf_counter()

    def stage(self, name):
        """Context manager for timing a code block. Usage:
            with engine.profiler.stage('inference'):
                ... do work ...
        """
        return _StageContext(self, name)

    def count(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + n

    def scalar(self, name, value):
        self.scalars[name] = value

    def get_report(self):
        """Structured dict of every metric, used by the log + JSON sidecar."""
        total = ((self.end_time - self.start_time)
                 if (self.start_time and self.end_time) else 0.0)

        stage_summary = {}
        for name, durations in self.stages.items():
            if not durations:
                continue
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            stage_summary[name] = {
                'count':   n,
                'total_s': float(sum(durations_sorted)),
                'mean_ms': 1000.0 * sum(durations_sorted) / n,
                'p50_ms':  1000.0 * durations_sorted[n // 2],
                'p95_ms':  1000.0 * durations_sorted[min(int(n * 0.95), n - 1)],
                'max_ms':  1000.0 * durations_sorted[-1],
                'min_ms':  1000.0 * durations_sorted[0],
                'pct_total': 100.0 * sum(durations_sorted) / max(total, 1e-9),
            }

        # GPU peak tracking unavailable through ONNX Runtime's public API.
        gpu_peak_mb = 0.0

        # CPU peak as DELTA from baseline — QGIS already uses 1+ GB at startup.
        cpu_peak_mb = 0.0
        if self._psutil is not None:
            try:
                cur = self._cpu_rss()
                cpu_peak_mb = float(cur - self._cpu_baseline) / 1024.0**2
                cpu_peak_mb = max(0.0, cpu_peak_mb)
            except Exception:
                pass

        n_pixels = int(self.scalars.get('image_pixels', 0))
        n_tiles = int(self.counts.get('tiles', 0))
        n_forward = int(self.counts.get('forward_passes', 0))
        n_polygons = int(self.scalars.get('n_polygons', 0))
        inv_total = 1.0 / max(total, 1e-9)

        return {
            'total_wall_time_s': round(total, 3),
            'device': str(self.device),
            'gpu_peak_mb': round(gpu_peak_mb, 1),
            'cpu_delta_mb': round(cpu_peak_mb, 1),
            'scalars': dict(self.scalars),
            'counts': dict(self.counts),
            'stages': stage_summary,
            'throughput': {
                'megapixels_per_sec': round(n_pixels / 1e6 * inv_total, 2),
                'tiles_per_sec':      round(n_tiles * inv_total, 2),
                'forwards_per_sec':   round(n_forward * inv_total, 2),
                'polygons_per_sec':   round(n_polygons * inv_total, 2),
            },
        }

    def format_summary(self):
        """Human-readable report for the QGIS log panel."""
        r = self.get_report()
        sc = r['scalars']
        L = []
        L.append('=' * 72)
        L.append('INFERENCE PROFILE')
        L.append('=' * 72)
        L.append(f"  device:      {r['device']}")
        L.append(f"  wall time:   {r['total_wall_time_s']:.2f} s")
        # ORT does not expose per-EP GPU memory through the public API, so
        # we can't report a real number. Show the device label instead so
        # users can tell at a glance whether GPU was actually used.
        dev = r.get('device', '?')
        if 'CPU' in dev:
            gpu_line = "  GPU peak:    N/A (CPU run)"
        else:
            gpu_line = (f"  GPU peak:    N/A "
                        f"(running on {dev}, ORT does not expose GPU memory)")
        L.append(gpu_line)
        L.append(f"  CPU delta:   {r['cpu_delta_mb']:.0f} MB"
                 if r['cpu_delta_mb'] > 0
                 else "  CPU delta:   N/A (psutil unavailable)")
        if 'image_w' in sc and 'image_h' in sc:
            mp = sc.get('image_pixels', 0) / 1e6
            L.append(f"  scene:       {sc['image_w']} x {sc['image_h']}"
                     f"  ({mp:.2f} MP)")
        if 'cls_models' in sc:
            L.append(f"  cls models:  {sc['cls_models']}")
        if 'tta_mode' in sc:
            L.append(f"  TTA mode:    {sc['tta_mode']}")
        if 'n_polygons' in sc:
            L.append(f"  polygons:    {sc['n_polygons']}")
        tp = r['throughput']
        L.append('')
        L.append(f"  throughput:  {tp['megapixels_per_sec']:>7.2f} MP/s   "
                 f"{tp['tiles_per_sec']:>6.2f} tiles/s   "
                 f"{tp['forwards_per_sec']:>6.2f} forwards/s   "
                 f"{tp['polygons_per_sec']:>6.2f} polys/s")
        L.append('')
        L.append('  Per-stage breakdown (sorted by total time):')
        L.append(f"  {'stage':<28} {'calls':>5} {'total':>9} "
                 f"{'mean':>9} {'p95':>9} {'%total':>7}")
        L.append('  ' + '-' * 70)
        for name, st in sorted(r['stages'].items(),
                               key=lambda kv: kv[1]['total_s'],
                               reverse=True):
            L.append(f"  {name:<28} {st['count']:>5} "
                     f"{st['total_s']:>7.2f} s {st['mean_ms']:>7.0f} ms "
                     f"{st['p95_ms']:>7.0f} ms {st['pct_total']:>6.1f}%")
        L.append('=' * 72)
        return '\n'.join(L)


class _StageContext:
    """Context manager from _InferenceProfiler.stage(). CUDA-syncs before/after
    so timings reflect kernel time, not queue submission."""
    __slots__ = ('p', 'name', '_t0')

    def __init__(self, profiler, name):
        self.p = profiler
        self.name = name
        self._t0 = 0.0

    def __enter__(self):
        self.p._cuda_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.p._cuda_sync()
        dt = time.perf_counter() - self._t0
        self.p.stages.setdefault(self.name, []).append(dt)
        return False





class BuildingDamageEngine:
    """AI damage detection - outputs building-shaped polygons"""
    
    DAMAGE_NAMES = {
        0: 'Background',
        1: 'No Damage',
        2: 'Minor Damage',
        3: 'Major Damage',
        4: 'Destroyed'
    }
    
    def __init__(self):
        # Two-network pipeline:
        #   loc_model  : pre-image -> binary building mask
        #   cls_models : pre+post  -> 4-class damage. List of one or more
        #                SiameseUNets; softmax outputs are averaged when >1.
        # The first cls model is always cls_best.pth; optional ensemble
        # members come from OPTIONAL_CLS_FILES.
        self.loc_model = None      # ort.InferenceSession for the loc U-Net
        self.cls_models = []       # list of ort.InferenceSession for cls members
        self.cls_model = None      # back-compat alias for the first member
        self.providers = _select_providers()
        self.device = _DeviceShim(self.providers[0])
        self.model_loaded = False
        self.img_size = 512

        # Loc gate: pixels above this loc-prob threshold are assigned a damage
        # class; below, background.
        self.loc_threshold = 0.4

        # Per-building object voting: collapses each connected building into
        # one damage class via argmax of summed per-pixel probabilities
        # (ChangeOS, Zheng et al., RSE 2021). Produces clean polygon outputs
        # and avoids the wide-area-scan failure mode of per-pixel mode.
        self.object_voting = True

        # Class upgrade rules applied AFTER per-building voting.
        # Each rule (from_class, to_class, min_ratio): a building voted as
        # from_class with a summed to_class score >= min_ratio * from_class
        # score gets reclassified as to_class. Classes: 1=NoDmg, 2=Minor,
        # 3=Major, 4=Destroyed. Set to [] to disable all rules.
        self.class_upgrade_rules = [(3, 4, 0.45)]

        # Watershed split: separates touching buildings into distinct polygon
        # instances. Uses both shape (distance transform) and confidence (loc
        # probability peaks) so footprints merge cleanly in dense rows.
        self.watershed_split = True
        self.watershed_min_distance = 7
        self.loc_core_threshold = 0.6
        self.morph_erode_radius = 1

        # GSD normalization: resample the input pair to the model's training
        # GSD (~0.5 m/px) before inference. Wrong-scale buildings are the #1
        # failure mode on non-xBD imagery. Geo-referencing is preserved.
        self.gsd_normalize = True
        self.target_gsd_m = 0.5
        self.gsd_tolerance = 1.25    # only resample if >25% off target
        self.gsd_max_scale = 4.0

        # ImageNet normalization. float32 dtype is required — float64 would
        # upcast the input image to ~3 GB on a 100+ MP scene.
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.profiler = _InferenceProfiler(self.device)

        # CPU Fast Mode (dialog checkbox). When True AND device is CPU:
        # ensemble members and TTA are skipped (~8× speedup, ~0.005 F1 cost).
        # Ignored on CUDA.
        self.cpu_fast_mode = False

        # Batched TTA: stack the 4 augmented orientations of one tile into
        # a batch=4 input and dispatch ONE session.run per cls model
        # instead of 4 sequential calls. Skips ~9 × per-call ORT overhead.
        # Auto-falls-back to per-pass on any memory error (see
        # _predict_tile_maybe_tta). Bounded peak memory: 4 × single-tile
        # working set, no growth across tiles.
        self.tta_batched = True
        self._tta_batched_fallback_logged = False

        # Diagnose the provider chain. If no GPU provider is compiled into
        # the current onnxruntime build, log a hint about which package
        # variant the user should install to enable GPU acceleration.
        available = ort.get_available_providers()
        self.log(f"Engine initialized on {self.device}")
        self.log(f"  ONNX Runtime providers available: {', '.join(available)}")
        gpu_providers = {'CUDAExecutionProvider', 'DmlExecutionProvider',
                         'CoreMLExecutionProvider', 'ROCMExecutionProvider'}
        if not (gpu_providers & set(available)):
            self.log(
                "  No GPU provider is compiled into this onnxruntime build. "
                "Inference will run on CPU.\n"
                "  To enable GPU, install ONE of (and uninstall plain "
                "'onnxruntime' first — they cannot coexist):\n"
                "    • Windows, any DX12 GPU (Intel / AMD / NVIDIA, no CUDA "
                "Toolkit needed):\n"
                "        python -m pip uninstall -y onnxruntime\n"
                "        python -m pip install onnxruntime-directml\n"
                "    • NVIDIA CUDA (requires matching CUDA Toolkit at OS "
                "level):\n"
                "        python -m pip uninstall -y onnxruntime\n"
                "        python -m pip install onnxruntime-gpu\n"
                "  Run the install commands from the OSGeo4W shell so the "
                "package lands in QGIS's Python environment.",
                Qgis.Warning)
    
    def log(self, message, level=Qgis.Info):
        QgsMessageLog.logMessage(message, 'Damage AI', level=level)
        print(f"[Damage AI] {message}")
    
    # Public weights distribution via GitHub Releases. Plain HTTPS GET, no
    # authentication required — anyone can download. After each weight
    # update: bump _RELEASE_TAG, upload the new files to the release page,
    # and refresh the SHA-256 hashes below.
    #     python -c "import hashlib;print(hashlib.sha256(open('loc_best_fp16.onnx','rb').read()).hexdigest())"
    _RELEASE_TAG = "v1.0.0"
    _RELEASE_BASE = (
        f"https://github.com/azeldev/beaconGIS/releases/download/{_RELEASE_TAG}"
    )
    WEIGHTS_HOMEPAGE = "https://github.com/azeldev/beaconGIS/releases"

    # Filenames of the on-disk model artifacts. The plugin ships fp16
    # weights for ~2× DirectML speedup + half the download size. The
    # engine auto-detects the on-disk dtype, so swapping these to the
    # fp32 names (loc_best.onnx, cls_best.onnx, cls_m2.onnx) and pointing
    # MODEL_FILES at the fp32 hashes still works — useful for users on
    # older onnxruntime builds where fp16 kernels are buggy.
    LOC_FILENAME = "loc_best_fp16.onnx"
    CLS_PRIMARY_FILENAME = "cls_best_fp16.onnx"

    MODEL_FILES = {
        "loc_best_fp16.onnx": (
            f"{_RELEASE_BASE}/loc_best_fp16.onnx",
            "cbe525b545bb35c511213d7805e33f2a40f12b2fb0cb1c461ec8b8820088518b",
        ),
        "cls_best_fp16.onnx": (
            f"{_RELEASE_BASE}/cls_best_fp16.onnx",
            "2e3e25b7b3c007e11f45037bf412d993f88017cf825a779807c0686b6bb2f724",
        ),
        "cls_m2_fp16.onnx": (
            f"{_RELEASE_BASE}/cls_m2_fp16.onnx",
            "9532d60121f68f6c8f439a8ad63c9c50d318e9e8c9ffb8e555fbc16ccc149125",
        ),
    }

    # Additional cls ensemble members loaded if present locally but NOT
    # auto-downloaded. cls_m2_fp16.onnx was promoted to MODEL_FILES because
    # the published benchmark uses it; cls_m3 stays opt-in for power users
    # who train their own third member.
    OPTIONAL_CLS_FILES = ['cls_m3_fp16.onnx', 'cls_m3.onnx']

    def _ensure_model_files(self, plugin_dir):
        """Download missing checkpoints from the configured public URLs and
        verify SHA-256 hashes. Atomic rename: the .onnx file only lands on
        disk after the full download succeeds + hash verifies."""
        import hashlib
        import urllib.request
        import urllib.error
        import ssl

        missing = [
            (fname, url, expected_sha)
            for fname, (url, expected_sha) in self.MODEL_FILES.items()
            if not os.path.exists(os.path.join(plugin_dir, fname))
        ]
        if not missing:
            return

        headers = {"User-Agent": "qgis-building-damage-plugin/1.0"}
        ctx = ssl.create_default_context()
        total_mb = sum(int(s) if isinstance(s, int) else 0 for s in [0])
        self.log(f"First-run weights download: {len(missing)} file(s) from "
                 f"{self.WEIGHTS_HOMEPAGE}")

        for fname, url, expected_sha in missing:
            path = os.path.join(plugin_dir, fname)
            self.log(f"  downloading {fname} from {url}...")
            tmp = path + ".part"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    sha = hashlib.sha256()
                    bytes_read = 0
                    # Log each 10% bucket exactly once. Without this guard
                    # the inner `pct % 10 == 0` test matches for EVERY 64KB
                    # chunk that arrives while pct rounds to e.g. 10.x,
                    # which spams hundreds of log lines per tier and can
                    # freeze the QGIS UI.
                    last_logged_bucket = 0
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = resp.read(1024 * 64)
                            if not chunk:
                                break
                            out.write(chunk)
                            sha.update(chunk)
                            bytes_read += len(chunk)
                            if total:
                                bucket = int((100 * bytes_read / total) // 10) * 10
                                if bucket > last_logged_bucket:
                                    last_logged_bucket = bucket
                                    self.log(f"    {bucket}% "
                                             f"({bytes_read // (1024*1024)} "
                                             f"of {total // (1024*1024)} MB)")
                got_sha = sha.hexdigest()
                if not expected_sha.startswith("REPLACE_"):
                    if got_sha != expected_sha:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass
                        raise RuntimeError(
                            f"SHA-256 mismatch for {fname}.\n"
                            f"  expected: {expected_sha}\n"
                            f"  got:      {got_sha}\n"
                            f"The downloaded file is corrupt or has been "
                            f"tampered with. Refusing to use it.")
                else:
                    self.log(f"    [{fname}] SHA-256 verification skipped "
                             f"(no baked-in hash). Got: {got_sha}",
                             Qgis.Warning)
                os.replace(tmp, path)
                self.log(f"    saved {fname}")
            except urllib.error.HTTPError as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                if e.code == 404:
                    raise FileNotFoundError(
                        f"Model file '{fname}' not found at the release URL.\n"
                        f"  URL: {url}\n\n"
                        f"This is likely a plugin packaging bug. Either:\n"
                        f"  1. The plugin's release tag is wrong, or\n"
                        f"  2. The release upload is incomplete.\n"
                        f"Visit {self.WEIGHTS_HOMEPAGE} to download manually "
                        f"and place the file in:\n  {plugin_dir}"
                    ) from e
                raise FileNotFoundError(
                    f"Could not download '{fname}'.\n"
                    f"  HTTP {e.code}: {e.reason}\n\n"
                    f"To download manually, visit:\n  {self.WEIGHTS_HOMEPAGE}"
                ) from e
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                msg = (f"Could not download '{fname}'.\n"
                       f"  {type(e).__name__}: {e}\n\n"
                       f"Check internet connectivity and try again. To "
                       f"download manually, visit:\n  {self.WEIGHTS_HOMEPAGE}\n"
                       f"and place the file in:\n  {plugin_dir}")
                self.log(msg, Qgis.Critical)
                raise FileNotFoundError(msg) from e

    def load_model(self):
        """Build ONNX Runtime inference sessions for the loc + cls networks.
        Downloads missing .onnx files from the gated HF repo on first run.
        Raises with an actionable message on any failure."""
        if self.model_loaded:
            return True

        plugin_dir = os.path.dirname(__file__)
        loc_path = os.path.join(plugin_dir, self.LOC_FILENAME)
        cls_path = os.path.join(plugin_dir, self.CLS_PRIMARY_FILENAME)

        self._ensure_model_files(plugin_dir)

        missing = [p for p in (loc_path, cls_path) if not os.path.exists(p)]
        if missing:
            msg = ("Missing model file(s):\n  " +
                   "\n  ".join(missing) +
                   f"\n\nDrop the files into:\n  {plugin_dir}")
            self.log(msg, Qgis.Critical)
            raise FileNotFoundError(msg)

        # Discover cls ensemble members. CPU Fast Mode loads only the primary.
        # Two sources:
        #   1. Auto-downloaded ensemble members in MODEL_FILES (e.g. cls_m2)
        #   2. User-supplied opt-in members in OPTIONAL_CLS_FILES (e.g. cls_m3)
        # Order matters: MODEL_FILES first so the published benchmark
        # configuration is the default ensemble.
        cls_paths = [(self.CLS_PRIMARY_FILENAME, cls_path)]
        # Fast Mode skips ensemble + TTA regardless of active device. Even
        # on a GPU, single-model can be ~2× faster than 2-model ensemble
        # when the user prefers latency over the ~0.5pt F1 boost the
        # ensemble provides.
        fast_skip = self.cpu_fast_mode
        if not fast_skip:
            for fname in self.MODEL_FILES:
                if fname.startswith('cls_') and fname != self.CLS_PRIMARY_FILENAME:
                    fpath = os.path.join(plugin_dir, fname)
                    if os.path.exists(fpath):
                        cls_paths.append((fname, fpath))
            for opt_name in self.OPTIONAL_CLS_FILES:
                opt_path = os.path.join(plugin_dir, opt_name)
                if os.path.exists(opt_path):
                    cls_paths.append((opt_name, opt_path))

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Match the active provider; CPU benefits from intra-op threading.
        if self.device.type == 'cpu':
            sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        else:
            # GPU providers: disable memory-pattern caching. The arena
            # allocator otherwise grows to peak and never shrinks across
            # many tile calls, which on multi-tile scenes (100+ tiles)
            # accumulates into multi-GB allocations and eventually OOMs.
            # CPU benefits from mem_pattern so we leave it on there.
            sess_opts.enable_mem_pattern = False

        try:
            self.loc_model = ort.InferenceSession(
                loc_path, sess_options=sess_opts, providers=self.providers)
            self.cls_models = []
            for _fname, fpath in cls_paths:
                self.cls_models.append(ort.InferenceSession(
                    fpath, sess_options=sess_opts, providers=self.providers))
            self.cls_model = self.cls_models[0]
        except Exception as e:
            msg = (f"ONNX Runtime session creation failed.\n"
                   f"  {type(e).__name__}: {e}\n\n"
                   f"This usually means a corrupt download. Try removing "
                   f".onnx files from the plugin folder so they re-download.")
            self.log(msg, Qgis.Critical)
            raise RuntimeError(msg) from e

        # Inspect the provider that actually got assigned (the chain falls
        # back if e.g. CUDA was listed but no GPU was found at runtime).
        actual = self.loc_model.get_providers()[0] if self.loc_model.get_providers() else 'CPUExecutionProvider'
        self.device = _DeviceShim(actual)
        # Update the profiler's device reference too — it was captured in
        # __init__ before load_model discovered the actual provider, so
        # without this re-bind the report would show the requested device,
        # not the one that actually ran.
        self.profiler.device = self.device

        # Detect each session's expected input dtype. Plugin ships fp16
        # weights for ~2× DirectML speedup; older fp32 deploys still work
        # because we cast input to whatever the session wants. Output
        # logits get cast back to fp32 for numerically-stable post-
        # processing (sigmoid/softmax in fp16 is unstable on extreme
        # logits and unnecessary anyway — the post-processing time is
        # tiny compared to inference).
        def _session_dtype(sess):
            type_str = str(sess.get_inputs()[0].type)
            return np.float16 if 'float16' in type_str else np.float32
        self.loc_input_dtype = _session_dtype(self.loc_model)
        self.cls_input_dtype = _session_dtype(self.cls_models[0])
        precision = ('fp16' if (self.loc_input_dtype == np.float16
                                or self.cls_input_dtype == np.float16)
                     else 'fp32')

        self.model_loaded = True

        n_cls = len(self.cls_models)
        cls_names = ', '.join(n for n, _ in cls_paths)
        cls_desc = (f"{n_cls}-model ensemble ({cls_names})" if n_cls > 1
                    else f"single model ({cls_names})")
        self.log(f"Loaded: {os.path.basename(loc_path)} + {cls_desc} "
                 f"on {self.device} ({precision})")
        if fast_skip:
            self.log("  Fast Mode active — TTA and ensemble members skipped")

        # If a GPU provider was in our preference list but didn't actually
        # bind to the session, surface that loudly — common cause is the
        # GPU package being installed without the matching driver/toolkit
        # (e.g. onnxruntime-gpu without CUDA Toolkit at the OS level).
        gpu_in_chain = [p for p in self.providers
                        if p != 'CPUExecutionProvider']
        if gpu_in_chain and actual == 'CPUExecutionProvider':
            self.log(
                f"  Warning: GPU providers {gpu_in_chain} are compiled into "
                f"this onnxruntime build but none bound to the session. "
                f"Falling back to CPU.\n"
                f"  Likely cause: the GPU package is installed but the "
                f"matching driver/toolkit is missing or version-mismatched.\n"
                f"  • onnxruntime-gpu needs CUDA Toolkit installed at the "
                f"OS level (https://developer.nvidia.com/cuda-downloads).\n"
                f"  • onnxruntime-directml needs an up-to-date GPU driver "
                f"with DirectX 12 support.",
                Qgis.Warning)

        return True

    def detect(self, before_layer, unused1, after_layer, unused2,
               sensitivity=50, shape_type='square',
               use_tta=False, tta_mode=None,
               tile_size=512, overlap=64):
        """Detect damage and return building-shaped polygon features.

        Args:
            tta_mode: 'off' | '4' | '8'. Overrides legacy `use_tta` when given.
            tile_size: sliding-window edge in px (must match training res).
            overlap: tile overlap in px.
        """
        # Fresh batched-TTA fallback log per detect() call so users see the
        # warning once per scene if it fires (rather than once per session).
        self._tta_batched_fallback_logged = False

        if tta_mode is None:
            tta_mode = '4' if use_tta else 'off'
        tta_mode = str(tta_mode).lower()
        if tta_mode not in ('off', '4', '8'):
            self.log(f"  Unknown tta_mode={tta_mode!r}; falling back to 'off'",
                     Qgis.Warning)
            tta_mode = 'off'

        # Fast Mode forces TTA off (no augmented passes), in addition to
        # the single-model load_model() configures.
        if self.cpu_fast_mode and tta_mode != 'off':
            self.log(f"   Fast Mode: forcing tta_mode='off' (was '{tta_mode}')")
            tta_mode = 'off'

        self.log("=" * 60)
        self.log("Building damage detection")
        self.log("=" * 60)

        self.profiler.reset()
        self.profiler.start_run()
        self.profiler.scalar('tta_mode', tta_mode)
        self.profiler.scalar('tile_size', tile_size)
        self.profiler.scalar('overlap', overlap)

        with self.profiler.stage('load_model'):
            self.load_model()
        self.profiler.scalar('cls_models', len(self.cls_models))

        # Read inputs
        with self.profiler.stage('read_rgb_pre'):
            before_rgb = self.read_rgb(before_layer)
        with self.profiler.stage('read_rgb_post'):
            after_rgb = self.read_rgb(after_layer)
        if before_rgb is None or after_rgb is None:
            raise Exception("Failed to read images")

        # Harmonize shapes — Siamese cls expects identical (H, W) for pre+post.
        # Prefer geographic warp; fall back to pixel resize.
        if after_rgb.shape[:2] != before_rgb.shape[:2]:
            with self.profiler.stage('warp_post_to_pre'):
                warped = self._warp_to_match(after_layer, before_layer)
                if warped is not None:
                    after_rgb = warped
                else:
                    import cv2
                    bh, bw = before_rgb.shape[:2]
                    after_rgb = cv2.resize(after_rgb, (bw, bh),
                                           interpolation=cv2.INTER_LINEAR)
                    self.log("  pre/post shapes differ — no CRS, used pixel resize",
                             Qgis.Warning)

        with self.profiler.stage('gsd_normalize'):
            before_rgb, after_rgb = self._normalize_gsd(
                before_rgb, after_rgb, before_layer)

        h, w = before_rgb.shape[:2]
        self.profiler.scalar('image_w', w)
        self.profiler.scalar('image_h', h)
        self.profiler.scalar('image_pixels', w * h)

        ref_extent = before_layer.extent()
        px_w = ref_extent.width() / w
        px_h = ref_extent.height() / h

        # One-line scene summary
        tta_label = {'off': 'off', '4': '4-way', '8': '8-way'}.get(
            tta_mode, tta_mode)
        self.log(f"Scene: {w}×{h} px  ·  tile {tile_size}/{overlap}  ·  "
                 f"TTA {tta_label}  ·  cls models: {len(self.cls_models)}")

        # Inference
        with self.profiler.stage('predict_total'):
            damage_mask, confidence_map = self.predict(
                before_rgb, after_rgb,
                tta_mode=tta_mode,
                tile_size=tile_size,
                overlap=overlap,
            )

        # Cache per-pixel artifacts for downstream mask/JSON sidecar export.
        self._last_damage_mask = damage_mask
        self._last_confidence_map = confidence_map
        self._last_extent = ref_extent
        self._last_crs = before_layer.crs() if before_layer is not None else None

        # Per-class distribution — single packed line.
        unique, counts = np.unique(damage_mask, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        total = int(damage_mask.size)
        building_pixels = total - dist.get(0, 0)
        if building_pixels == 0:
            self.log("No buildings detected.")
            return []
        dist_pct = {self.DAMAGE_NAMES.get(c, str(c)):
                    100.0 * dist.get(c, 0) / max(building_pixels, 1)
                    for c in (1, 2, 3, 4)}
        mean_conf = float(confidence_map[damage_mask > 0].mean())
        self.log(
            f"Damage (% of building pixels): "
            f"NoDmg {dist_pct['No Damage']:.1f}%  "
            f"Minor {dist_pct['Minor Damage']:.1f}%  "
            f"Major {dist_pct['Major Damage']:.1f}%  "
            f"Destroyed {dist_pct['Destroyed']:.1f}%  "
            f"(mean conf {mean_conf:.2f})")

        # Polygon extraction
        with self.profiler.stage('extract_polygons'):
            features = self.extract_contour_polygons(
                damage_mask, confidence_map, ref_extent, px_w, px_h, sensitivity)

        self.profiler.scalar('n_polygons', len(features))
        self.profiler.end_run()
        self._last_profile = self.profiler.get_report()
        self.log(self.profiler.format_summary())

        self.log("=" * 60)
        self.log(f"Detected {len(features)} buildings.")
        self.log("=" * 60)

        return features

    # ------------------------------------------------------------------
    # Tile-based inference with optional TTA
    # ------------------------------------------------------------------
    def predict(self, before_rgb, after_rgb,
                use_tta=True, tta_mode=None,
                tile_size=512, overlap=64):
        """Sliding-window Hanning-blended inference. Returns (damage_mask,
        confidence_map). tta_mode in {'off', '4', '8'} (overrides use_tta)."""
        if tta_mode is None:
            tta_mode = '4' if use_tta else 'off'
        tta_mode = str(tta_mode).lower()
        if tta_mode not in ('off', '4', '8'):
            tta_mode = '4'
        h, w = before_rgb.shape[:2]
        num_classes = 5     # background + 4 damage classes

        before_norm = self._normalize_image(before_rgb)
        after_norm = self._normalize_image(after_rgb)

        # Single-shot fast path when the image fits in one tile.
        fits_in_one_tile = (h <= tile_size) and (w <= tile_size)
        asymmetric_small = (h < tile_size) or (w < tile_size)
        if fits_in_one_tile or asymmetric_small:
            pad_h = max(0, tile_size - h)
            pad_w = max(0, tile_size - w)
            before_p = np.pad(before_norm, ((0, pad_h), (0, pad_w), (0, 0)))
            after_p = np.pad(after_norm, ((0, pad_h), (0, pad_w), (0, 0)))
            with self.profiler.stage('predict_tile'):
                prob = self._predict_tile_maybe_tta(before_p, after_p, tta_mode)
            tta_mult_sm = {'off': 1, '4': 4, '8': 8}.get(tta_mode, 4)
            self.profiler.count('tiles', 1)
            self.profiler.count('forward_passes',
                                tta_mult_sm * max(1, len(self.cls_models)))
            prob = prob[:, :h, :w]
            del before_norm, after_norm
            with self.profiler.stage('finalize_prediction'):
                damage_mask, confidence_map = self._finalize_prediction(prob)
            del prob
            return damage_mask, confidence_map

        positions = self._compute_positions(h, w, tile_size, overlap)
        weight = self._make_weight_map(tile_size).astype(np.float32)

        prob_accum = np.zeros((num_classes, h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)

        n_tiles = len(positions)
        tta_mult = {'off': 1, '4': 4, '8': 8}.get(tta_mode, 4)

        # forwards_per_tile = TTA augmentations × cls ensemble members
        forwards_per_tile = tta_mult * max(1, len(self.cls_models))
        log_every = max(1, n_tiles // 5)   # ~5 progress lines per scene

        for idx, (y0, x0) in enumerate(positions):
            y1 = y0 + tile_size
            x1 = x0 + tile_size
            before_tile = before_norm[y0:y1, x0:x1]
            after_tile = after_norm[y0:y1, x0:x1]
            with self.profiler.stage('predict_tile'):
                prob_tile = self._predict_tile_maybe_tta(
                    before_tile, after_tile, tta_mode)
            self.profiler.count('tiles', 1)
            self.profiler.count('forward_passes', forwards_per_tile)

            prob_accum[:, y0:y1, x0:x1] += prob_tile * weight[None, ...]
            weight_accum[y0:y1, x0:x1] += weight

            if (idx + 1) % log_every == 0 or idx + 1 == n_tiles:
                self.log(f"  tiles {idx + 1}/{n_tiles}")

        weight_accum = np.maximum(weight_accum, 1e-6)
        prob_accum /= weight_accum[None, ...]

        # Tile loop is done — free the normalized RGB buffers + accumulators.
        # On a 100+ MP scene these are several GB each; freeing them before
        # _finalize_prediction leaves enough headroom for the voting path's
        # per-building probability sums to allocate without OOM.
        del before_norm, after_norm, weight_accum

        with self.profiler.stage('finalize_prediction'):
            damage_mask, confidence_map = self._finalize_prediction(prob_accum)
        # prob_accum is the engine's largest single buffer (~5 × H × W float32 =
        # ~2.7 GB on a 136 MP scene). _finalize_prediction has already extracted
        # damage_mask + confidence_map from it; release the local binding so
        # polygon extraction has memory to work with.
        del prob_accum
        return damage_mask, confidence_map

    def _finalize_prediction(self, prob):
        """Convert (5, H, W) blended probability into (damage_mask, confidence)
        using a hard localization gate.

        prob[0] = 1 - loc_prob (background channel); prob[1..4] = loc * cls
        (damage channels). Pixels with loc_prob > loc_threshold are assigned
        a damage class; the rest stay background. Object voting (default)
        then collapses each building to a single class; per-pixel mode
        (object_voting=False) keeps the per-pixel argmax.
        """
        loc_building = 1.0 - prob[0]
        dmg_stack = prob[1:]
        building = loc_building > self.loc_threshold

        self._inst_labels = None
        self._inst_class = None

        if getattr(self, 'object_voting', True):
            return self._object_vote(building, dmg_stack, prob)

        # Per-pixel fallback
        dmg_argmax = np.argmax(dmg_stack, axis=0).astype(np.uint8) + 1
        damage_mask = np.where(building, dmg_argmax, 0).astype(np.uint8)
        chosen = np.take_along_axis(prob, damage_mask[None, ...], axis=0)[0]
        confidence_map = np.where(building, chosen, prob[0]).astype(np.float32)
        self._store_perpixel_instances(building, damage_mask, loc_building)
        return damage_mask, confidence_map

    def _object_vote(self, building, dmg_stack, prob):
        """Collapse each connected building to one damage class via argmax of
        summed per-pixel softmax probabilities (ChangeOS, Zheng et al. 2021).

        Pipeline: watershed split -> per-building probability sums ->
        argmax vote -> class upgrade rules -> per-pixel mask repaint.
        """
        from scipy import ndimage
        H, W = building.shape
        damage_mask = np.zeros((H, W), dtype=np.uint8)
        confidence_map = prob[0].astype(np.float32).copy()    # bg conf default

        if getattr(self, 'watershed_split', True):
            loc_prob = 1.0 - prob[0]
            labeled, n = self._split_touching_buildings(building, loc_prob=loc_prob)
        else:
            labeled, n = ndimage.label(building)
        if n == 0:
            return damage_mask, confidence_map

        idx = np.arange(1, n + 1)
        # Summed damage probability per building per class -> (n, 4).
        class_sums = np.stack(
            [ndimage.sum(dmg_stack[c], labeled, idx) for c in range(4)], axis=1
        )
        comp_total = class_sums.sum(axis=1) + 1e-6

        # Argmax of summed per-pixel softmax probabilities per building.
        comp_class = class_sums.argmax(axis=1).astype(np.uint8) + 1
        comp_conf = (class_sums.max(axis=1) / comp_total).astype(np.float32)

        # Class upgrade rules — see __init__ for the rule format.
        for rule in getattr(self, 'class_upgrade_rules', []):
            try:
                from_c, to_c, min_ratio = rule
            except (TypeError, ValueError):
                self.log(f"   skipping malformed upgrade rule: {rule!r}",
                         Qgis.Warning)
                continue
            if not (1 <= from_c <= 4 and 1 <= to_c <= 4):
                self.log(f"   skipping out-of-range upgrade rule: {rule!r}",
                         Qgis.Warning)
                continue
            from_idx = from_c - 1
            to_idx   = to_c - 1
            is_from = (comp_class == from_c)
            if not is_from.any():
                continue
            ratio = class_sums[:, to_idx] / np.maximum(class_sums[:, from_idx],
                                                       1e-6)
            upgrade = is_from & (ratio >= float(min_ratio))
            n_up = int(upgrade.sum())
            if n_up > 0:
                comp_class[upgrade] = to_c
                new_conf = (class_sums[upgrade, to_idx]
                            / comp_total[upgrade]).astype(np.float32)
                comp_conf[upgrade] = new_conf
                names = {1: 'No Damage', 2: 'Minor', 3: 'Major', 4: 'Destroyed'}
                self.log(
                    f"   class upgrade {names[from_c]}->{names[to_c]} "
                    f"(ratio>={min_ratio:.2f}): {n_up}/{int(is_from.sum())} "
                    f"borderline buildings reclassified")

        self.log(f"   Object voting: {n} buildings -> one class each")

        # Paint per-building decisions back to a per-pixel mask + confidence.
        cls_lut = np.zeros(n + 1, dtype=np.uint8)
        cls_lut[1:] = comp_class
        conf_lut = np.zeros(n + 1, dtype=np.float32)
        conf_lut[1:] = comp_conf
        damage_mask = cls_lut[labeled]
        bld = labeled > 0
        confidence_map[bld] = conf_lut[labeled][bld]

        # Expose instances for polygon extraction.
        self._inst_labels = labeled.astype(np.int32, copy=False)
        self._inst_class = cls_lut
        return damage_mask, confidence_map

    def _store_perpixel_instances(self, building, damage_mask, loc_prob):
        """Label building instances and tag each with its majority per-pixel
        damage class. Used by the per-pixel finalize_prediction fallback
        when object_voting is False. The damage mask is left unchanged."""
        self._inst_labels = None
        self._inst_class = None
        if not building.any():
            return
        from scipy import ndimage
        if getattr(self, 'watershed_split', True):
            labeled, n = self._split_touching_buildings(building, loc_prob=loc_prob)
        else:
            labeled, n = ndimage.label(building)
        if n == 0:
            return

        # Per-instance majority class from the final per-pixel mask, vectorized:
        # bincount over (instance_label * 5 + class) -> argmax across classes.
        flat = labeled.ravel().astype(np.int64)
        dm = damage_mask.ravel().astype(np.int64)
        sel = flat > 0
        keys = flat[sel] * 5 + dm[sel]
        counts = np.bincount(keys, minlength=(n + 1) * 5).reshape(n + 1, 5)
        cls_lut = counts[:, 1:].argmax(axis=1).astype(np.uint8) + 1   # 1..4
        cls_lut[0] = 0
        self._inst_labels = labeled.astype(np.int32, copy=False)
        self._inst_class = cls_lut

    def _split_touching_buildings(self, building, loc_prob=None,
                                  min_distance=None):
        """Marker-controlled watershed to separate touching buildings. Uses
        both shape (distance transform peaks) and loc-confidence peaks as
        markers, so the split works on both xBD-style and fuzzy non-xBD masks.
        Falls back to plain CC labeling if scikit-image is missing or the
        watershed step OOMs on huge scenes. Returns (labeled, n_components)."""
        from scipy import ndimage as ndi
        b = building.astype(bool)
        if not b.any():
            return np.zeros(building.shape, dtype=np.int32), 0

        if min_distance is None:
            min_distance = getattr(self, 'watershed_min_distance', 7)

        try:
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
        except ImportError:
            if not getattr(self, '_warned_no_skimage', False):
                self.log("   watershed split needs scikit-image (pip install "
                         "scikit-image); falling back to merged components",
                         Qgis.Warning)
                self._warned_no_skimage = True
            lbl, n = ndi.label(b)
            return lbl.astype(np.int32, copy=False), int(n)

        merged_n = int(ndi.label(b)[1])

        # Distance transform (shape cue). Wrapped in a MemoryError safety
        # net so a truly huge scan falls back to plain CC labeling instead
        # of crashing the plugin. Touching buildings will merge into one
        # polygon when this fallback fires; the per-pixel damage mask is
        # unaffected.
        try:
            dist = ndi.distance_transform_edt(b).astype(np.float32)
        except MemoryError:
            scene_mp = (b.shape[0] * b.shape[1]) / 1e6
            self.log(f"   Watershed distance transform OOM on {scene_mp:.1f} MP "
                     f"scene; falling back to plain connected-components "
                     f"labeling. Touching buildings will merge into one "
                     f"polygon on this scene.", Qgis.Warning)
            lbl, n = ndi.label(b)
            return lbl.astype(np.int32, copy=False), int(n)

        # Marker-search region: erode so touching cores separate. The full mask
        # `b` is kept for flooding, so footprints stay intact. Revert if erosion
        # wipes everything (scene of tiny buildings).
        r = int(getattr(self, 'morph_erode_radius', 1))
        core_region = b
        if r > 0:
            eroded = ndi.binary_erosion(b, iterations=r)
            if eroded.any():
                core_region = eroded

        prob_aware = loc_prob is not None
        if prob_aware:
            prob = np.asarray(loc_prob, dtype=np.float32)
            prob_s = ndi.gaussian_filter(prob, sigma=1.0)
            # Combined surface: confident AND central pixels score highest.
            dnorm = dist / (float(dist.max()) + 1e-6)
            surface = 0.5 * dnorm + 0.5 * prob_s
            # Free the intermediates — on a 136 MP scene each is ~545 MB and
            # they're not needed past this point. Gives watershed below room
            # to run its internal float64 cast.
            del prob_s, dnorm, dist
            # Seed only from confident cores; fall back to the eroded region if
            # nothing clears the bar (uniformly low-confidence non-xBD scene).
            core_thr = float(getattr(self, 'loc_core_threshold', 0.6))
            core = core_region & (prob >= core_thr)
            marker_region = core if core.any() else core_region
            del prob
        else:
            surface = dist
            marker_region = core_region

        coords = peak_local_max(surface, min_distance=min_distance,
                                labels=marker_region)
        if len(coords) <= 1:
            # 0 or 1 center -> nothing to split.
            lbl, n = ndi.label(b)
            return lbl.astype(np.int32, copy=False), int(n)

        peak_mask = np.zeros(surface.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(peak_mask)
        del peak_mask

        # Flood from markers over the inverted surface (valleys = the gaps
        # between buildings), constrained to the full building mask. Wrapped
        # in a MemoryError safety net — skimage.watershed casts the surface
        # to float64 internally (~1 GB on a 136 MP scene), which can OOM
        # even after the distance transform succeeded. Fall back to plain
        # CC labeling in that case.
        try:
            labels = watershed(-surface, markers, mask=b)
        except MemoryError:
            scene_mp = (b.shape[0] * b.shape[1]) / 1e6
            self.log(f"   Watershed flood OOM on {scene_mp:.1f} MP scene "
                     f"(skimage's float64 internal cast). Falling back to "
                     f"plain connected-components labeling — touching "
                     f"buildings will merge into one polygon.", Qgis.Warning)
            lbl, n = ndi.label(b)
            return lbl.astype(np.int32, copy=False), int(n)
        n = int(labels.max())
        mode = 'prob-aware' if prob_aware else 'shape'
        self.log(f"   Watershed split ({mode}): {n} building instances "
                 f"(from {merged_n} merged components)")
        return labels.astype(np.int32, copy=False), n

    def _compute_positions(self, h, w, tile_size, overlap):
        """Top-left (y, x) positions for a sliding window, last tile snapped
        to the edge so the whole image is covered without exceeding it."""
        stride = tile_size - overlap
        ys = list(range(0, max(1, h - tile_size + 1), stride))
        if not ys or ys[-1] != h - tile_size:
            ys.append(max(0, h - tile_size))
        xs = list(range(0, max(1, w - tile_size + 1), stride))
        if not xs or xs[-1] != w - tile_size:
            xs.append(max(0, w - tile_size))
        return [(y, x) for y in ys for x in xs]

    def _make_weight_map(self, tile_size):
        """2D Hanning window — zero at borders, one in centre — so overlapping
        tiles blend without visible seams."""
        hann = np.hanning(tile_size + 2)[1:-1]  # drop the zeros at the ends
        return np.outer(hann, hann)

    def _predict_tile_maybe_tta(self, before_tile, after_tile, tta_mode):
        """Run one tile with optional geometric TTA.
        Modes: 'off' (single pass), '4' (Klein four-group), '8' (full D4).

        For mode '4' with `self.tta_batched=True`, stacks the 4 orientations
        into a batch=4 input and dispatches ONE session.run per model
        instead of 4 sequential calls — saves ~9× per-call ORT overhead on
        GPU. Auto-falls-back to per-pass on memory errors.
        """
        if isinstance(tta_mode, bool):
            tta_mode = '4' if tta_mode else 'off'
        tta_mode = str(tta_mode).lower()

        if tta_mode == 'off':
            return self._predict_tile(before_tile, after_tile)

        # Fast path: batched TTA-4. Falls back to per-pass on OOM.
        if tta_mode == '4' and getattr(self, 'tta_batched', True):
            try:
                return self._predict_tile_batched_tta4(before_tile, after_tile)
            except Exception as e:
                msg = str(e).lower()
                is_mem = any(t in msg for t in
                             ('memory', 'alloc', 'oom', 'out of'))
                if not is_mem:
                    # Not a memory error — surface real bugs immediately.
                    raise
                if not getattr(self, '_tta_batched_fallback_logged', False):
                    self.log(
                        f"  Batched TTA hit a memory error; falling back to "
                        f"per-pass TTA for the rest of this scene. "
                        f"({type(e).__name__}: {str(e)[:120]})",
                        Qgis.Warning)
                    self._tta_batched_fallback_logged = True
                # Fall through to per-pass path below.

        probs = []
        if tta_mode == '4':
            probs.append(self._predict_tile(before_tile, after_tile))
            p = self._predict_tile(before_tile[:, ::-1], after_tile[:, ::-1])
            probs.append(p[:, :, ::-1])
            p = self._predict_tile(before_tile[::-1, :], after_tile[::-1, :])
            probs.append(p[:, ::-1, :])
            p = self._predict_tile(before_tile[::-1, ::-1],
                                   after_tile[::-1, ::-1])
            probs.append(p[:, ::-1, ::-1])
        elif tta_mode == '8':
            for k in range(4):
                b_rot = np.rot90(before_tile, k=k, axes=(0, 1))
                a_rot = np.rot90(after_tile, k=k, axes=(0, 1))
                p = self._predict_tile(b_rot, a_rot)
                probs.append(np.rot90(p, k=-k, axes=(1, 2)))
                p = self._predict_tile(b_rot[:, ::-1], a_rot[:, ::-1])
                probs.append(np.rot90(p[:, :, ::-1], k=-k, axes=(1, 2)))
        else:
            return self._predict_tile(before_tile, after_tile)

        return np.mean(np.stack(probs, axis=0), axis=0)

    def _predict_tile_batched_tta4(self, before_tile, after_tile):
        """Batched 4-way TTA: identity + hflip + vflip + 180-rot stacked
        into one batch=4 input. Skips ~9× per-call ORT overhead vs the
        sequential path. Returns the same (5, H, W) probability map as
        _predict_tile_maybe_tta('4') — bit-equivalent except for fp32
        reduction order on the final mean."""
        H_in, W_in = before_tile.shape[:2]

        # Pad H, W up to encoder-stride multiple (same as _predict_tile).
        stride = self.ENCODER_STRIDE
        H_pad = ((H_in + stride - 1) // stride) * stride
        W_pad = ((W_in + stride - 1) // stride) * stride
        pad_h, pad_w = H_pad - H_in, W_pad - W_in

        def _maybe_pad(arr):
            if not (pad_h or pad_w):
                return arr
            mode = 'reflect' if (pad_h < H_in and pad_w < W_in) else 'edge'
            return np.pad(arr,
                          ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

        # Build 4 augmented (pre, post) pairs of the SAME tile.
        # Inverse transforms applied to outputs preserve the per-pixel
        # mapping back to identity orientation.
        variants = [
            (before_tile,               after_tile),                 # identity
            (before_tile[:, ::-1],      after_tile[:, ::-1]),        # hflip
            (before_tile[::-1, :],      after_tile[::-1, :]),        # vflip
            (before_tile[::-1, ::-1],   after_tile[::-1, ::-1]),     # 180
        ]

        # HWC -> CHW + batch stack. Cast to the model's expected input
        # dtype (fp16 for the shipped weights, fp32 if older weights are
        # in use). Logits cast back to fp32 for stable post-processing.
        pre_batch = np.ascontiguousarray(np.stack(
            [_maybe_pad(b).transpose(2, 0, 1) for b, _ in variants],
            axis=0), dtype=self.loc_input_dtype)
        post_batch_fp = np.ascontiguousarray(np.stack(
            [_maybe_pad(a).transpose(2, 0, 1) for _, a in variants],
            axis=0), dtype=self.cls_input_dtype)
        # pair input may need fp32-vs-fp16-cast pre too; build separately
        # so we don't double-cast unnecessarily.
        pre_for_pair = (pre_batch if self.cls_input_dtype == self.loc_input_dtype
                        else pre_batch.astype(self.cls_input_dtype))
        pair_batch = np.concatenate([pre_for_pair, post_batch_fp], axis=1)
        # shapes:  pre_batch (4, 3, H_pad, W_pad),  pair_batch (4, 6, ..)

        # One loc call (batch=4) and one cls call per ensemble member.
        loc_logits = self.loc_model.run(['logits'], {'input': pre_batch})[0]
        # loc_logits: (4, 1, H_pad, W_pad). Cast to fp32 for stable sigmoid.
        loc_prob = _np_sigmoid(loc_logits[:, 0].astype(np.float32, copy=False))
        del loc_logits, pre_batch, pre_for_pair

        if len(self.cls_models) == 1:
            cls_logits = self.cls_models[0].run(
                ['logits'], {'input': pair_batch})[0]   # (4, 4, ...)
            cls_prob = _np_softmax(
                cls_logits.astype(np.float32, copy=False), axis=1)
            del cls_logits
        else:
            per_member = []
            for m in self.cls_models:
                logits = m.run(['logits'], {'input': pair_batch})[0]
                logits = logits.astype(np.float32, copy=False)
                per_member.append(_np_softmax(logits, axis=1))
                del logits
            cls_prob = np.mean(np.stack(per_member, axis=0), axis=0)
            del per_member
        del pair_batch

        # Build full 5-channel prob per orientation (batch=4): bg + 4*damage.
        bg = (1.0 - loc_prob)[:, None, ...]              # (4, 1, ...)
        damage = loc_prob[:, None, ...] * cls_prob       # (4, 4, ...)
        full_batch = np.concatenate([bg, damage], axis=1)  # (4, 5, ...)
        del bg, damage, loc_prob, cls_prob

        # Crop padded outputs back to original tile size.
        if pad_h or pad_w:
            full_batch = full_batch[:, :, :H_in, :W_in]

        # Inverse-transform each orientation, then average.
        # CHW axes: 0=channel, 1=H, 2=W
        averaged = (full_batch[0]                       # identity
                    + full_batch[1, :, :, ::-1]         # undo hflip
                    + full_batch[2, :, ::-1, :]         # undo vflip
                    + full_batch[3, :, ::-1, ::-1]      # undo 180
                    ) * 0.25
        return np.ascontiguousarray(averaged)

    # ENCODER_STRIDE: the SeResNeXt-50 encoder downsamples by 2 at five
    # stages (max effective stride = 32). The ONNX export traced the
    # model's runtime shape-matching `if` as a constant for a clean
    # 512x512 input, so the graph cannot self-correct for non-multiple-
    # of-32 spatial dims at runtime — it'll Concat-mismatch by 1 px.
    # We pre-pad input H/W up to a multiple of ENCODER_STRIDE and crop
    # the output back. Constant edge padding is used (rather than
    # zero-padding in normalized space) so the padded strip doesn't look
    # like a wall of background to the network.
    ENCODER_STRIDE = 32

    def _predict_tile(self, before_tile, after_tile):
        """Single ONNX Runtime forward pass on one normalized tile. Returns
        a (5, H, W) probability map:
            channel 0     = 1 - loc_prob (background)
            channels 1..4 = loc_prob * cls_prob (damage classes)
        With multiple cls models their softmax outputs are arithmetic-mean
        averaged before the loc gate."""
        # Pad H, W up to the next multiple of ENCODER_STRIDE so all internal
        # feature-map dimensions divide cleanly. Reflection padding matches
        # the local image statistics better than constant zero; fall back
        # to edge padding if the pad amount would exceed the input dim - 1
        # (np.pad mode='reflect' requires pad <= dim - 1).
        H_in, W_in = before_tile.shape[:2]
        stride = self.ENCODER_STRIDE
        H_pad = ((H_in + stride - 1) // stride) * stride
        W_pad = ((W_in + stride - 1) // stride) * stride
        pad_h, pad_w = H_pad - H_in, W_pad - W_in
        if pad_h or pad_w:
            mode = ('reflect'
                    if (pad_h < H_in and pad_w < W_in)
                    else 'edge')
            before_tile = np.pad(before_tile,
                                  ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
            after_tile = np.pad(after_tile,
                                 ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

        # HWC -> CHW, add batch dim. Cast to each session's expected input
        # dtype (fp16 for the shipped weights, fp32 if older weights are
        # in use). Logits get cast back to fp32 for stable post-processing.
        pre = np.ascontiguousarray(
            before_tile.transpose(2, 0, 1)[None, ...], dtype=self.loc_input_dtype)
        post = np.ascontiguousarray(
            after_tile.transpose(2, 0, 1)[None, ...], dtype=self.cls_input_dtype)
        # cls input is [pre | post] concat; if loc and cls use different
        # dtypes we need a separately-cast pre for the cls path.
        pre_for_cls = (pre if self.cls_input_dtype == self.loc_input_dtype
                       else pre.astype(self.cls_input_dtype))
        pair = np.concatenate([pre_for_cls, post], axis=1)

        loc_logits = self.loc_model.run(['logits'], {'input': pre})[0]
        loc_prob = _np_sigmoid(
            loc_logits[0, 0].astype(np.float32, copy=False))     # (H_pad, W_pad)

        if len(self.cls_models) == 1:
            cls_logits = self.cls_models[0].run(['logits'], {'input': pair})[0]
            cls_prob = _np_softmax(
                cls_logits[0].astype(np.float32, copy=False), axis=0)
        else:
            per_member = []
            for m in self.cls_models:
                logits = m.run(['logits'], {'input': pair})[0]
                per_member.append(_np_softmax(
                    logits[0].astype(np.float32, copy=False), axis=0))
            cls_prob = np.mean(np.stack(per_member, axis=0), axis=0)

        bg = (1.0 - loc_prob)[None, ...]                   # (1, H_pad, W_pad)
        damage = loc_prob[None, ...] * cls_prob            # (4, H_pad, W_pad)
        full = np.concatenate([bg, damage], axis=0)        # (5, H_pad, W_pad)

        # Crop padded outputs back to the caller's original H_in, W_in.
        if pad_h or pad_w:
            full = full[:, :H_in, :W_in]
        return full

    def _normalize_image(self, rgb):
        """Scale to [0,1] and apply ImageNet mean/std. Input may be uint8 or
        float32 in [0,255]."""
        img = rgb.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = (img - self.mean) / self.std
        return img.astype(np.float32)
    
    def extract_contour_polygons(self, mask, confidence_map, extent, px_w, px_h, sensitivity):
        """Extract building polygons with mean softmax confidence per region.
        Prefers instance-aware extraction when watershed instance labels are
        available; falls back to per-class contour tracing otherwise."""
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
            self.log("   OpenCV not available, using scipy")

        inst_labels = getattr(self, '_inst_labels', None)
        inst_class = getattr(self, '_inst_class', None)
        if has_cv2 and inst_labels is not None and inst_class is not None:
            feats = self.extract_instance_polygons(
                inst_labels, inst_class, confidence_map,
                extent, px_w, px_h, sensitivity)
            self.log(f"   Instance-aware extraction — total buildings: {len(feats)}")
            return feats

        features = []
        if has_cv2:
            self.log("   Using OpenCV for contour extraction (per-class fallback)")
        for damage_class in [1, 2, 3, 4]:
            class_mask = (mask == damage_class).astype(np.uint8)
            if np.sum(class_mask) == 0:
                continue
            if has_cv2:
                class_features = self.extract_cv2_contours(
                    class_mask, confidence_map, damage_class,
                    extent, px_w, px_h, sensitivity)
            else:
                class_features = self.extract_scipy_polygons(
                    class_mask, confidence_map, damage_class,
                    extent, px_w, px_h, sensitivity)
            features.extend(class_features)
        self.log(f"   Total buildings: {len(features)}")
        return features

    def extract_instance_polygons(self, inst_labels, inst_class, confidence_map,
                                  extent, px_w, px_h, sensitivity):
        """One polygon per building instance. No MORPH_CLOSE so neighbours
        are never bridged. Each feature carries geometry, area, class,
        and mean per-region confidence."""
        import cv2
        from scipy import ndimage
        features = []
        n = int(inst_labels.max())
        if n == 0:
            return features
        min_pixels = max(20, 80 - sensitivity)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        slices = ndimage.find_objects(inst_labels)
        for k in range(1, n + 1):
            sl = slices[k - 1] if (k - 1) < len(slices) else None
            if sl is None:
                continue
            dc = int(inst_class[k]) if k < len(inst_class) else 0
            if dc < 1 or dc > 4:
                continue
            y0, x0 = sl[0].start, sl[1].start
            sub = (inst_labels[sl] == k)
            if int(sub.sum()) < min_pixels:
                continue
            # Light open to despeckle; no close (closing would re-merge footprints).
            m = cv2.morphologyEx((sub.astype(np.uint8)) * 255,
                                 cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conf_sub = confidence_map[sl]
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                if area_pixels < min_pixels:
                    continue
                region = np.zeros(sub.shape, dtype=np.uint8)
                cv2.drawContours(region, [contour], -1, color=1, thickness=-1)
                rp = region.astype(bool)
                mean_conf = float(conf_sub[rp].mean()) if rp.any() else 0.0
                epsilon = 0.015 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                if len(simplified) < 3:
                    continue
                pts = []
                for point in simplified:
                    col, row = point[0]
                    x = extent.xMinimum() + (x0 + col) * px_w
                    y = extent.yMaximum() - (y0 + row) * px_h
                    pts.append(QgsPointXY(x, y))
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                geom = QgsGeometry.fromPolygonXY([pts])
                if geom.isEmpty() or not geom.isGeosValid():
                    continue
                area_m2 = area_pixels * px_w * px_h
                features.append({
                    'geometry': geom,
                    'intensity': dc / 4.0,
                    'area': float(area_m2),
                    'damage_class': dc,
                    'damage_name': self.DAMAGE_NAMES.get(dc, 'Unknown'),
                    'confidence': mean_conf,
                })
        return features

    def extract_cv2_contours(self, binary_mask, confidence_map, damage_class,
                             extent, px_w, px_h, sensitivity):
        """OpenCV per-class contour fallback (used when no instance labels)."""
        import cv2

        features = []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask * 255, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_pixels = max(20, 80 - sensitivity)
        H, W = binary_mask.shape

        for contour in contours:
            area_pixels = cv2.contourArea(contour)

            if area_pixels < min_pixels:
                continue

            # Rasterize the raw contour for exact pixel support.
            region = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(region, [contour], -1, color=1, thickness=-1)
            region_pixels = region.astype(bool)
            if region_pixels.any():
                mean_conf = float(confidence_map[region_pixels].mean())
            else:
                mean_conf = 0.0

            epsilon = 0.015 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            if len(simplified) < 3:
                continue

            pts = []
            for point in simplified:
                col, row = point[0]
                x = extent.xMinimum() + col * px_w
                y = extent.yMaximum() - row * px_h
                pts.append(QgsPointXY(x, y))

            if pts[0] != pts[-1]:
                pts.append(pts[0])

            geom = QgsGeometry.fromPolygonXY([pts])

            if geom.isEmpty() or not geom.isGeosValid():
                continue

            area_m2 = area_pixels * px_w * px_h

            features.append({
                'geometry': geom,
                'intensity': damage_class / 4.0,
                'area': float(area_m2),
                'damage_class': damage_class,
                'damage_name': self.DAMAGE_NAMES.get(damage_class, 'Unknown'),
                'confidence': mean_conf,
            })

        return features

    def extract_scipy_polygons(self, binary_mask, confidence_map, damage_class,
                               extent, px_w, px_h, sensitivity):
        """No-OpenCV fallback: scipy convex hulls per connected region."""
        from scipy import ndimage
        from scipy.spatial import ConvexHull

        features = []

        labeled, n_regions = ndimage.label(binary_mask)

        min_pixels = max(20, 80 - sensitivity)

        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            region_size = int(np.sum(region_mask))

            if region_size < min_pixels:
                continue

            mean_conf = float(confidence_map[region_mask].mean())

            rows, cols = np.where(region_mask)

            if len(rows) < 3:
                continue

            points = np.column_stack((cols, rows))

            try:
                if len(points) >= 3:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]

                    pts = []
                    for col, row in hull_points:
                        x = extent.xMinimum() + col * px_w
                        y = extent.yMaximum() - row * px_h
                        pts.append(QgsPointXY(x, y))

                    pts.append(pts[0])

                    geom = QgsGeometry.fromPolygonXY([pts])

                    if not geom.isEmpty():
                        area_m2 = region_size * px_w * px_h

                        features.append({
                            'geometry': geom,
                            'intensity': damage_class / 4.0,
                            'area': float(area_m2),
                            'damage_class': damage_class,
                            'damage_name': self.DAMAGE_NAMES.get(damage_class, 'Unknown'),
                            'confidence': mean_conf,
                        })
            except Exception:
                continue

        return features
    
    def resize(self, image, size):
        pil = PILImage.fromarray(image.astype(np.uint8))
        resized = pil.resize((size, size), PILImage.BILINEAR)
        return np.array(resized).astype(np.float32)
    
    def resize_mask(self, mask, target_shape):
        pil = PILImage.fromarray(mask.astype(np.uint8))
        resized = pil.resize((target_shape[1], target_shape[0]), PILImage.NEAREST)
        return np.array(resized)
    
    def read_rgb(self, layer):
        try:
            from osgeo import gdal
            
            ds = gdal.Open(layer.source())
            if not ds:
                return None
            
            n_bands = ds.RasterCount
            h, w = ds.RasterYSize, ds.RasterXSize
            
            self.log(f"   Reading {n_bands} bands, size {w}x{h}")
            
            if n_bands >= 3:
                rgb = np.zeros((h, w, 3), dtype=np.float32)
                for i in range(3):
                    band = ds.GetRasterBand(i + 1).ReadAsArray()
                    if band is not None:
                        rgb[:, :, i] = band.astype(np.float32)
            else:
                band = ds.GetRasterBand(1).ReadAsArray()
                if band is None:
                    return None
                rgb = np.stack([band, band, band], axis=2).astype(np.float32)
            
            ds = None

            for i in range(3):
                channel = rgb[:, :, i]
                if channel.max() > channel.min():
                    p2, p98 = np.percentile(channel, (2, 98))
                    channel = np.clip((channel - p2) / (p98 - p2 + 1e-6) * 255, 0, 255)
                    rgb[:, :, i] = channel
            
            return rgb
            
        except Exception as e:
            self.log(f"ERROR: {e}", Qgis.Critical)
            return None

    def _layer_gsd_meters(self, layer):
        """Ground sample distance (m/px) as (gsd_x, gsd_y), or None.
        Handles projected and geographic (degrees) CRSs."""
        try:
            from qgis.core import (QgsDistanceArea, QgsUnitTypes, QgsPointXY,
                                   QgsProject)
        except Exception:
            return None
        try:
            ext = layer.extent()
            w, h = layer.width(), layer.height()
            if w <= 0 or h <= 0:
                return None
            crs = layer.crs()
            if not crs.isValid():
                return None

            # Projected: convert map-units-per-pixel to meters.
            units = crs.mapUnits()
            if units != QgsUnitTypes.DistanceDegrees:
                try:
                    f = QgsUnitTypes.fromUnitToUnitFactor(
                        units, QgsUnitTypes.DistanceMeters)
                except Exception:
                    f = 0.0
                if f and f > 0:
                    return (ext.width() / w) * f, (ext.height() / h) * f

            # Geographic / unknown: measure on the ellipsoid.
            da = QgsDistanceArea()
            da.setSourceCrs(crs, QgsProject.instance().transformContext())
            da.setEllipsoid('WGS84')
            cy = (ext.yMinimum() + ext.yMaximum()) / 2.0
            cx = (ext.xMinimum() + ext.xMaximum()) / 2.0
            width_m = da.measureLine(QgsPointXY(ext.xMinimum(), cy),
                                     QgsPointXY(ext.xMaximum(), cy))
            height_m = da.measureLine(QgsPointXY(cx, ext.yMinimum()),
                                      QgsPointXY(cx, ext.yMaximum()))
            if width_m <= 0 or height_m <= 0:
                return None
            return width_m / w, height_m / h
        except Exception as e:
            self.log(f"   GSD estimate failed: {e}", Qgis.Warning)
            return None

    def _normalize_gsd(self, before_rgb, after_rgb, before_layer):
        """Resample pre & post to the model's training scale (~target_gsd_m).
        Geographic extent is unchanged; only the array shape changes."""
        if not getattr(self, 'gsd_normalize', True):
            return before_rgb, after_rgb

        gsd = self._layer_gsd_meters(before_layer)
        if gsd is None:
            self.log("   GSD unknown (no CRS / not georeferenced); "
                     "skipping resolution normalization")
            return before_rgb, after_rgb

        gsd_x, gsd_y = gsd
        target = float(self.target_gsd_m)
        tol = float(self.gsd_tolerance)
        gsd_mean = (gsd_x + gsd_y) / 2.0
        ratio = gsd_mean / target if target > 0 else 1.0
        self.log(f"   Input GSD ~{gsd_mean:.3f} m/px "
                 f"(target {target:.2f}); ratio {ratio:.2f}x")

        if target <= 0 or (1.0 / tol) <= ratio <= tol:
            self.log("   GSD within tolerance; no resampling")
            return before_rgb, after_rgb

        # scale > 1 = upsample (coarser input); scale < 1 = downsample.
        cap = float(self.gsd_max_scale)
        sx_raw, sy_raw = gsd_x / target, gsd_y / target
        sx = min(max(sx_raw, 1.0 / cap), cap)
        sy = min(max(sy_raw, 1.0 / cap), cap)
        capped = (sx != sx_raw) or (sy != sy_raw)

        if ratio > tol:
            self.log("   WARNING: input is much coarser than the training "
                     "resolution; buildings may be too small for reliable "
                     "results even after upsampling.", Qgis.Warning)
        if capped:
            self.log(f"   resample factor capped at {cap:.0f}x", Qgis.Warning)

        h, w = before_rgb.shape[:2]
        new_w = max(8, int(round(w * sx)))
        new_h = max(8, int(round(h * sy)))
        if (new_w, new_h) == (w, h):
            self.log("   GSD resample is a no-op after rounding; skipping")
            return before_rgb, after_rgb

        import cv2
        shrinking = (new_w * new_h) < (w * h)
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        before_rs = cv2.resize(before_rgb, (new_w, new_h), interpolation=interp)
        after_rs = cv2.resize(after_rgb, (new_w, new_h), interpolation=interp)
        self.log(f"   GSD normalize: {w}x{h} -> {new_w}x{new_h} "
                 f"(now ~{target:.2f} m/px)")
        return before_rs, after_rs

    def _warp_to_match(self, src_layer, ref_layer):
        """Warp src_layer onto ref_layer's CRS/grid via GDAL. Returns
        (H, W, 3) float32 RGB, or None if CRS/extent unusable."""
        try:
            from osgeo import gdal, osr
        except ImportError:
            return None

        try:
            ref_crs = ref_layer.crs()
            src_crs = src_layer.crs()
            if not ref_crs.isValid() or not src_crs.isValid():
                return None

            ref_ext = ref_layer.extent()
            ref_ds = gdal.Open(ref_layer.source())
            src_ds = gdal.Open(src_layer.source())
            if ref_ds is None or src_ds is None:
                return None

            ref_w, ref_h = ref_ds.RasterXSize, ref_ds.RasterYSize
            x_res = ref_ext.width() / ref_w
            y_res = ref_ext.height() / ref_h

            warped = gdal.Warp(
                '', src_ds, format='MEM',
                outputBounds=(ref_ext.xMinimum(), ref_ext.yMinimum(),
                              ref_ext.xMaximum(), ref_ext.yMaximum()),
                xRes=x_res, yRes=y_res,
                dstSRS=ref_crs.toWkt(),
                resampleAlg='bilinear',
            )
            ref_ds = None
            src_ds = None
            if warped is None:
                return None

            n_bands = warped.RasterCount
            h, w = warped.RasterYSize, warped.RasterXSize
            if n_bands < 1:
                return None

            rgb = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(min(3, n_bands)):
                band = warped.GetRasterBand(i + 1).ReadAsArray()
                if band is not None:
                    rgb[:, :, i] = band.astype(np.float32)
            if n_bands == 1:
                rgb[:, :, 1] = rgb[:, :, 0]
                rgb[:, :, 2] = rgb[:, :, 0]
            warped = None

            # Same percentile stretch as read_rgb (match normalization range).
            for i in range(3):
                ch = rgb[:, :, i]
                if ch.max() > ch.min():
                    p2, p98 = np.percentile(ch, (2, 98))
                    rgb[:, :, i] = np.clip((ch - p2) / (p98 - p2 + 1e-6) * 255,
                                           0, 255)
            return rgb
        except Exception as e:
            self.log(f"   warp_to_match failed: {e}", Qgis.Warning)
            return None
