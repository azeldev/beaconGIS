from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog, QFileDialog
from qgis.core import QgsProject, QgsRasterLayer, QgsMapLayerProxyModel
import os

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'change_detector_dialog_base.ui'))


class ChangeDetectorDialog(QDialog, FORM_CLASS):
    """Building Damage Assessment dialog."""

    def __init__(self, iface, parent=None):
        super(ChangeDetectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface

        self.damage_before_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.damage_after_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        if hasattr(self, 'aoi_combo'):
            self.aoi_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
            self.aoi_combo.setAllowEmptyLayer(True)

        self.damage_before_button.clicked.connect(self.load_damage_before)
        self.damage_after_button.clicked.connect(self.load_damage_after)

        if hasattr(self, 'segmentation_slider'):
            self.segmentation_slider.valueChanged.connect(self._on_segmentation_changed)
            self._on_segmentation_changed(self.segmentation_slider.value())

    def _seg_min_distance(self, value):
        """Slider 0..100 -> watershed min-distance px (higher slider = more
        splits = smaller min-distance). 0->24px, 50->14px, 100->4px."""
        return int(round(4 + (100 - value) / 100.0 * 20))

    def _on_segmentation_changed(self, value):
        if hasattr(self, 'seg_value_label'):
            self.seg_value_label.setText(f"{value}%")

    def load_damage_before(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Before Disaster RGB Image", "",
            "GeoTIFF (*.tif *.tiff);;All Files (*.*)")
        if file_path:
            layer = QgsRasterLayer(file_path, "Before Disaster")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.damage_before_combo.setLayer(layer)

    def load_damage_after(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select After Disaster RGB Image", "",
            "GeoTIFF (*.tif *.tiff);;All Files (*.*)")
        if file_path:
            layer = QgsRasterLayer(file_path, "After Disaster")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                self.damage_after_combo.setLayer(layer)

    def get_damage_before_layer(self):
        return self.damage_before_combo.currentLayer()

    def get_damage_after_layer(self):
        return self.damage_after_combo.currentLayer()

    def get_use_tta(self):
        if hasattr(self, 'tta_checkbox'):
            return bool(self.tta_checkbox.isChecked())
        return False

    def get_tta_mode(self):
        if hasattr(self, 'tta_checkbox'):
            return '4' if self.tta_checkbox.isChecked() else 'off'
        return 'off'

    def get_segmentation_min_distance(self):
        if hasattr(self, 'segmentation_slider'):
            return self._seg_min_distance(self.segmentation_slider.value())
        return None

    def get_aoi_layer(self):
        if hasattr(self, 'aoi_combo'):
            return self.aoi_combo.currentLayer()
        return None

    def get_save_gpkg(self):
        if hasattr(self, 'save_gpkg_checkbox'):
            return bool(self.save_gpkg_checkbox.isChecked())
        return False

    def get_save_mask(self):
        if hasattr(self, 'save_mask_checkbox'):
            return bool(self.save_mask_checkbox.isChecked())
        return False

    def get_cpu_fast_mode(self):
        if hasattr(self, 'cpu_fast_mode_checkbox'):
            return bool(self.cpu_fast_mode_checkbox.isChecked())
        return False
