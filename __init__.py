"""beaconGIS — plugin entry point.

The bootstrap import below uses ONLY Python stdlib + QGIS PyQt5, both of
which are guaranteed to be available. It runs the dependency check + the
optional one-click pip install BEFORE the heavy plugin modules import.
This prevents ModuleNotFoundError on a fresh QGIS without numpy /
onnxruntime / Pillow / scipy / opencv installed.
"""


def classFactory(iface):
    from . import _bootstrap

    result = _bootstrap.ensure_dependencies(iface)

    if result == _bootstrap.READY:
        # All deps importable — load the real plugin.
        from .change_detector import ChangeDetector
        return ChangeDetector(iface)
    elif result == _bootstrap.RESTART_NEEDED:
        # Deps just installed; QGIS needs a restart before they're usable.
        return _bootstrap.RestartHintPlugin(iface)
    else:
        # User declined install or install failed. Show a toolbar shell
        # with one action: "install dependencies" (retry).
        return _bootstrap.DummyPlugin(iface)
