def classFactory(iface):
    from .change_detector import ChangeDetector
    return ChangeDetector(iface)