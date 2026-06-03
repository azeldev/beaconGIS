"""Plugin dependency bootstrap.

Runs in classFactory() BEFORE the heavy plugin modules import (which would
fail on a fresh QGIS without onnxruntime/numpy/etc installed). Uses ONLY
Python stdlib + QGIS PyQt5 — both of which are guaranteed available in
any QGIS install — so the bootstrap module itself loads cleanly even when
the plugin's actual runtime dependencies are missing.

Flow:
  1. classFactory imports this module + calls ensure_dependencies(iface).
  2. ensure_dependencies returns READY (deps OK), RESTART_NEEDED
     (deps just installed), or DECLINED (user said no / install failed).
  3. classFactory uses the result to decide what to return — the real
     plugin class, a restart-hint shell, or a dummy that exposes a
     "retry install" action.
"""
import importlib
import os
import subprocess
import sys


# Bootstrap outcomes
READY = "ready"                # All deps importable; load the real plugin.
RESTART_NEEDED = "restart"     # Deps installed; user must restart QGIS.
DECLINED = "declined"          # User skipped, or install failed.


def _find_python_executable():
    """Locate the real Python interpreter — NOT sys.executable.

    Inside QGIS on Windows, sys.executable points to qgis-bin.exe (the
    QGIS launcher) because QGIS embeds Python. Running pip via that
    executable would re-launch QGIS instead of invoking pip, which is
    exactly the bug that opens new QGIS windows on every pip call.

    The real interpreter is at sys.exec_prefix/python.exe on Windows or
    sys.exec_prefix/bin/python on Unix. We probe a few likely locations
    and return the first one that exists. Returns None if nothing is
    found — caller should surface a clear error in that case."""
    if sys.platform == 'win32':
        candidates = [
            os.path.join(sys.exec_prefix, 'python.exe'),
            os.path.join(sys.base_exec_prefix, 'python.exe'),
            os.path.join(sys.prefix, 'python.exe'),
            os.path.join(os.path.dirname(sys.executable), 'python.exe'),
        ]
    else:
        candidates = [
            os.path.join(sys.exec_prefix, 'bin', 'python3'),
            os.path.join(sys.exec_prefix, 'bin', 'python'),
            os.path.join(sys.base_exec_prefix, 'bin', 'python3'),
            os.path.join(sys.base_exec_prefix, 'bin', 'python'),
        ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


# Windows: suppress the cmd console window pip would otherwise pop up.
_SUBPROCESS_CREATE_FLAGS = (
    subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)


def _required_dependencies():
    """Return [(import_name, pip_install_name, friendly_label), ...].

    On Windows the default ONNX Runtime variant is the DirectML build so
    users get GPU acceleration on any DX12 card without installing CUDA
    Toolkit. Mac/Linux get plain onnxruntime."""
    is_win = sys.platform == 'win32'
    return [
        ('onnxruntime',
         'onnxruntime-directml' if is_win else 'onnxruntime',
         'ONNX Runtime' + (' (DirectML)' if is_win else '')),
        ('numpy',  'numpy',         'NumPy'),
        ('PIL',    'Pillow',        'Pillow'),
        ('scipy',  'scipy',         'SciPy'),
        ('cv2',    'opencv-python', 'OpenCV'),
    ]


def _check_missing():
    """Probe each required dep. Returns (missing_pip_names, missing_labels)."""
    missing_pip = []
    missing_labels = []
    for import_name, pip_name, label in _required_dependencies():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_pip.append(pip_name)
            missing_labels.append(label)
    return missing_pip, missing_labels


def _show_manual_instructions(iface, pip_cmd):
    from qgis.PyQt.QtWidgets import QMessageBox
    QMessageBox.information(
        iface.mainWindow(), "Manual install",
        "Open the <b>OSGeo4W Shell</b> from your Windows Start menu "
        "(not regular Command Prompt — OSGeo4W ships its own Python) "
        "and paste:<br><br>"
        f"<pre style='background:#eee;padding:6px;'>{pip_cmd}</pre><br>"
        "Restart QGIS after the install finishes. The plugin's toolbar "
        "icons will appear automatically.")


def _run_pip(python_exe, missing_pip, extra_flags=None):
    """Returns subprocess.CompletedProcess. Times out at 10 minutes.
    Uses the discovered python_exe (NOT sys.executable, which is the
    QGIS binary on Windows). CREATE_NO_WINDOW suppresses the cmd console
    that would otherwise flash up on every pip invocation."""
    extra_flags = extra_flags or []
    cmd = ([python_exe, '-m', 'pip', 'install',
            '--disable-pip-version-check']
           + extra_flags + missing_pip)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        creationflags=_SUBPROCESS_CREATE_FLAGS)


def ensure_dependencies(iface):
    """Main entry point. Returns READY / RESTART_NEEDED / DECLINED."""
    missing_pip, missing_labels = _check_missing()
    if not missing_pip:
        return READY

    # Defer Qt imports until we know we need the dialog (avoids any cost
    # for the common fast-path where everything's already installed).
    from qgis.PyQt.QtWidgets import QMessageBox, QApplication
    from qgis.core import Qgis

    # Find the actual Python interpreter (NOT sys.executable, which is
    # qgis-bin.exe on Windows and would re-launch QGIS on every pip call).
    python_exe = _find_python_executable()
    if python_exe is None:
        # Can't auto-install without finding the interpreter. Show manual
        # instructions and bail out.
        QMessageBox.critical(
            iface.mainWindow(), "beaconGIS — could not locate Python",
            "Could not find the Python interpreter inside your QGIS install. "
            "Auto-install is unavailable.<br><br>"
            "Open the <b>OSGeo4W Shell</b> from your Windows Start menu, then "
            "paste:<br><br>"
            f"<pre style='background:#eee;padding:6px;'>"
            f"python -m pip install {' '.join(missing_pip)}</pre><br>"
            "Restart QGIS after the install finishes.")
        return DECLINED

    pip_cmd = (f'"{python_exe}" -m pip install '
               + ' '.join(missing_pip))

    # First-run dialog. Three buttons: Install now / Show manual / Skip.
    msg = QMessageBox(iface.mainWindow())
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("beaconGIS — install dependencies?")
    msg.setText("<b>beaconGIS</b> needs these Python packages:")
    msg.setInformativeText(
        "  • " + "<br>  • ".join(missing_labels) + "<br><br>"
        "These are not bundled with QGIS. <b>Auto-install them now?</b><br><br>"
        "Install runs via pip; takes 1–3 min depending on connection. "
        "QGIS will be briefly unresponsive. Restart QGIS afterwards so the "
        "plugin can pick up the new libraries.")
    msg.setStandardButtons(
        QMessageBox.Yes | QMessageBox.Cancel | QMessageBox.Help)
    msg.button(QMessageBox.Yes).setText("Install now")
    msg.button(QMessageBox.Help).setText("Show manual command")
    msg.button(QMessageBox.Cancel).setText("Skip for now")
    msg.setDefaultButton(QMessageBox.Yes)
    reply = msg.exec_()

    if reply == QMessageBox.Help:
        _show_manual_instructions(iface, pip_cmd)
        return DECLINED
    if reply != QMessageBox.Yes:
        return DECLINED

    # Notify + force repaint before the blocking pip call.
    iface.messageBar().pushMessage(
        "beaconGIS",
        f"Installing {len(missing_pip)} package(s) via pip — "
        f"QGIS will pause for 1–3 min...",
        level=Qgis.Info, duration=0)
    QApplication.processEvents()

    try:
        result = _run_pip(python_exe, missing_pip)
        # Permission / externally-managed fallbacks → retry with --user
        if result.returncode != 0:
            err = (result.stderr or '').lower()
            if any(t in err for t in (
                    'permission denied', 'access is denied',
                    'errno 13', 'externally-managed-environment',
                    'no write permission')):
                result = _run_pip(python_exe, missing_pip, ['--user'])

        iface.messageBar().clearWidgets()

        if result.returncode == 0:
            QMessageBox.information(
                iface.mainWindow(), "beaconGIS — install complete",
                "Dependencies installed successfully.<br><br>"
                "<b>Please restart QGIS</b> so the plugin can load the "
                "new libraries.<br><br>"
                "After restart, the toolbar icons will appear automatically.")
            return RESTART_NEEDED

        tail = (result.stderr or result.stdout or '')[-1200:]
        QMessageBox.critical(
            iface.mainWindow(), "Install failed",
            f"pip exited with code {result.returncode}.<br><br>"
            f"<b>Last output:</b><br>"
            f"<pre style='background:#eee;padding:6px;'>{tail}</pre><br>"
            f"Try manually from the OSGeo4W shell:<br>"
            f"<pre style='background:#eee;padding:6px;'>{pip_cmd}</pre>")
        return DECLINED

    except subprocess.TimeoutExpired:
        iface.messageBar().clearWidgets()
        QMessageBox.critical(
            iface.mainWindow(), "Install timeout",
            f"pip install timed out after 10 minutes. Try manually from "
            f"the OSGeo4W shell:<br>"
            f"<pre style='background:#eee;padding:6px;'>{pip_cmd}</pre>")
        return DECLINED
    except Exception as e:
        iface.messageBar().clearWidgets()
        QMessageBox.critical(
            iface.mainWindow(), "Install error",
            f"<b>{type(e).__name__}:</b> {e}<br><br>"
            f"Try manually from the OSGeo4W shell:<br>"
            f"<pre style='background:#eee;padding:6px;'>{pip_cmd}</pre>")
        return DECLINED


# ---------------------------------------------------------------------------
# Plugin shells used when the real plugin can't be loaded.
# ---------------------------------------------------------------------------
class RestartHintPlugin:
    """Shown after a successful auto-install, until QGIS is restarted."""

    def __init__(self, iface):
        self.iface = iface
        self.action = None

    def initGui(self):
        from qgis.PyQt.QtWidgets import QAction, QMessageBox
        from qgis.PyQt.QtGui import QIcon

        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            'beaconGIS — restart QGIS to finish setup',
            self.iface.mainWindow())
        self.action.triggered.connect(self._show_hint)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu('&beaconGIS', self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginRasterMenu('&beaconGIS', self.action)

    def _show_hint(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.information(
            self.iface.mainWindow(), "beaconGIS — pending restart",
            "Dependencies have been installed but QGIS must be restarted "
            "before the plugin can use them.<br><br>"
            "<b>Please restart QGIS.</b>")


class DummyPlugin:
    """Shown when the user declined auto-install. Exposes a single
    'Retry dependency install' action so they can change their mind."""

    def __init__(self, iface):
        self.iface = iface
        self.action = None

    def initGui(self):
        from qgis.PyQt.QtWidgets import QAction
        from qgis.PyQt.QtGui import QIcon

        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            'beaconGIS — install dependencies',
            self.iface.mainWindow())
        self.action.triggered.connect(self._retry)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu('&beaconGIS', self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginRasterMenu('&beaconGIS', self.action)

    def _retry(self):
        result = ensure_dependencies(self.iface)
        if result == RESTART_NEEDED:
            # User just installed; they'll see the restart prompt from
            # ensure_dependencies. Nothing more to do here.
            pass
