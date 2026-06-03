import os
import datetime
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, QgsField, QgsFields,
                       QgsFillSymbol, QgsCategorizedSymbolRenderer, QgsRendererCategory,
                       QgsGeometry, QgsCoordinateTransform, QgsVectorFileWriter,
                       QgsMessageLog, Qgis)
from qgis.PyQt.QtCore import QVariant
from .change_detector_dialog import ChangeDetectorDialog
from .building_damage_engine import BuildingDamageEngine
from .assistant import AssessmentReportsDock
from .satellite_downloader import SatelliteDownloaderDock


class ChangeDetector:
    """QGIS plugin for AI building damage detection from pre/post imagery."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.dialog = None
        self.assistant_dock = None
        self.satellite_dock = None
        self.actions = []
        self.menu = '&Building Damage Assessment'
        self.toolbar = self.iface.addToolBar('BuildingDamage')
        self.toolbar.setObjectName('BuildingDamage')

    def add_action(self, icon_path, text, callback, enabled_flag=True,
                   add_to_menu=True, add_to_toolbar=True, status_tip=None,
                   whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)
        if whats_this is not None:
            action.setWhatsThis(whats_this)
        if add_to_toolbar:
            self.toolbar.addAction(action)
        if add_to_menu:
            self.iface.addPluginToRasterMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.add_action(icon_path, text='Detect Building Damage',
                        callback=self.run, parent=self.iface.mainWindow())

        self.add_action(icon_path, text='Assessment Reports',
                        callback=self.show_assistant, parent=self.iface.mainWindow())

        self.add_action(icon_path, text='Satellite Downloader',
                        callback=self.show_satellite_downloader, parent=self.iface.mainWindow())

    def unload(self):
        for action in self.actions:
            self.iface.removePluginRasterMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)

        if self.assistant_dock:
            self.iface.removeDockWidget(self.assistant_dock)
            self.assistant_dock.deleteLater()
            self.assistant_dock = None

        if self.satellite_dock:
            self.iface.removeDockWidget(self.satellite_dock)
            self.satellite_dock.deleteLater()
            self.satellite_dock = None

        if self.dialog is not None:
            self.dialog.deleteLater()
            self.dialog = None

        # ONNX Runtime sessions release their memory when their references
        # go out of scope, so no explicit cache flush is needed here.
        del self.toolbar

    def show_assistant(self):
        if self.assistant_dock is None:
            self.assistant_dock = AssessmentReportsDock(self.iface, self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.assistant_dock)
        else:
            if self.assistant_dock.isVisible():
                self.assistant_dock.hide()
            else:
                self.assistant_dock.show()
                self.assistant_dock.refresh_statistics()

    def show_satellite_downloader(self):
        if self.satellite_dock is None:
            self.satellite_dock = SatelliteDownloaderDock(self.iface, self.iface.mainWindow())
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.satellite_dock)
        else:
            if self.satellite_dock.isVisible():
                self.satellite_dock.hide()
            else:
                self.satellite_dock.show()

    def run(self):
        if self.dialog is None:
            self.dialog = ChangeDetectorDialog(self.iface)

        self.dialog.show()
        result = self.dialog.exec_()

        if result:
            self.process_damage_detection()

    def _check_dependencies(self):
        """Verify Python deps are installed. Returns (ok, missing_list)."""
        missing = []
        for mod_name, friendly_name in [
            ("onnxruntime", "ONNX Runtime"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("scipy", "SciPy"),
            ("cv2", "OpenCV"),
        ]:
            try:
                __import__(mod_name)
            except ImportError:
                missing.append(friendly_name)
        return (len(missing) == 0, missing)

    def process_damage_detection(self):
        # Dependency check first — clear error instead of deep traceback.
        ok, missing = self._check_dependencies()
        if not ok:
            QMessageBox.critical(
                None, "Missing dependencies",
                f"This plugin needs Python libraries that aren't installed "
                f"in your QGIS environment:\n\n"
                f"  • " + "\n  • ".join(missing) + "\n\n"
                f"Install them by opening the OSGeo4W shell (or your QGIS "
                f"Python environment) and running:\n\n"
                f"    python -m pip install onnxruntime numpy pillow scipy opencv-python\n\n"
                f"For GPU acceleration, install one of:\n"
                f"  • Windows (any DX12 GPU): pip install onnxruntime-directml\n"
                f"  • NVIDIA CUDA:           pip install onnxruntime-gpu\n\n"
                f"After installing, restart QGIS."
            )
            return

        try:
            before_image = self.dialog.get_damage_before_layer()
            after_image = self.dialog.get_damage_after_layer()

            if not all([before_image, after_image]):
                QMessageBox.warning(None, "Error",
                    "Please select both images:\n"
                    "- Before Disaster RGB Image\n"
                    "- After Disaster RGB Image")
                return

            # Cache the engine across runs (model load is 3-4 s). Cache is
            # invalidated when CPU Fast Mode changes since that changes the
            # loaded model set.
            new_fast_mode = (self.dialog.get_cpu_fast_mode()
                             if hasattr(self.dialog, 'get_cpu_fast_mode')
                             else False)
            cached = getattr(self, '_cached_engine', None)
            cached_fast = getattr(self, '_cached_engine_fast_mode', None)
            if cached is None or cached_fast != new_fast_mode:
                engine = BuildingDamageEngine()
                engine.cpu_fast_mode = new_fast_mode
                self._cached_engine = engine
                self._cached_engine_fast_mode = new_fast_mode
            else:
                engine = cached
                # Profiler state must be fresh per detect; the engine's
                # cached models stay loaded.
                engine.profiler.reset()

            # Auto-timestamped output layer name so multiple runs in the same
            # session don't collide on top of each other.
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            layer_name = f"Building Damage {timestamp}"

            self.iface.messageBar().pushMessage(
                "Building Damage",
                "Analyzing building damage with AI...",
                level=0, duration=0)

            try:
                tta_mode = 'off'
                if hasattr(self.dialog, 'get_tta_mode'):
                    tta_mode = self.dialog.get_tta_mode()
                elif hasattr(self.dialog, 'get_use_tta'):
                    tta_mode = '4' if self.dialog.get_use_tta() else 'off'

                if hasattr(self.dialog, 'get_segmentation_min_distance'):
                    seg_md = self.dialog.get_segmentation_min_distance()
                    if seg_md:
                        engine.watershed_min_distance = int(seg_md)

                change_features = engine.detect(
                    before_image, None,
                    after_image, None,
                    sensitivity=50,
                    shape_type='square',
                    tta_mode=tta_mode,
                )
            except Exception as e:
                self.iface.messageBar().clearWidgets()
                QMessageBox.critical(
                    None, "Model Error",
                    f"Could not run AI building damage model.\n\n"
                    f"Details:\n{str(e)}\n\n"
                    f"Check the 'Damage AI' log panel (View → Panels → Log Messages) "
                    f"for the full traceback."
                )
                import traceback
                traceback.print_exc()
                return

            if not change_features:
                self.iface.messageBar().clearWidgets()
                QMessageBox.information(None, "Result",
                    "No buildings detected in the post-disaster image.")
                return

            # AOI clip: drop polygons outside the user-supplied region.
            aoi_layer = None
            if hasattr(self.dialog, 'get_aoi_layer'):
                aoi_layer = self.dialog.get_aoi_layer()
            if aoi_layer is not None:
                before_n = len(change_features)
                change_features = self._clip_to_aoi(
                    change_features, aoi_layer, before_image)
                print(f"[Damage AI] AOI clip: {before_n} -> {len(change_features)} polygons")
                if not change_features:
                    self.iface.messageBar().clearWidgets()
                    QMessageBox.information(None, "Result",
                        "No buildings inside the AOI.")
                    return

            self.create_damage_layer(
                change_features, before_image, layer_name)

            # Optional file exports (Output Options checkboxes).
            try:
                self._save_outputs_if_requested(
                    engine, change_features, before_image, after_image,
                    layer_name)
            except Exception as e:
                # Non-critical: notify but keep the detection result on screen.
                print(f"[Damage AI] output export failed: "
                      f"{type(e).__name__}: {e}")
                self.iface.messageBar().pushMessage(
                    "Output export",
                    f"Could not write all requested output files: "
                    f"{type(e).__name__}: {e}",
                    level=1, duration=8)

            # Open the assistant panel and pass results (handles its own
            # stats + CSV export).
            self.show_assistant()
            if self.assistant_dock:
                self.assistant_dock.set_assessment_data(
                    change_features, layer_name)

            self.iface.messageBar().clearWidgets()
            self.iface.messageBar().pushMessage(
                "Success",
                f"Detected {len(change_features)} buildings!",
                level=3, duration=5)

        except Exception as e:
            self.iface.messageBar().clearWidgets()
            QMessageBox.critical(None, "Error", f"An error occurred:\n\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _clip_to_aoi(self, features, aoi_layer, raster_layer):
        """Keep only polygons that intersect the AOI layer. Reprojects AOI
        to raster CRS if needed."""
        raster_crs = raster_layer.crs()
        aoi_crs = aoi_layer.crs()

        if aoi_crs != raster_crs:
            transform = QgsCoordinateTransform(
                aoi_crs, raster_crs, QgsProject.instance())
        else:
            transform = None

        aoi_geoms = []
        for feat in aoi_layer.getFeatures():
            geom = QgsGeometry(feat.geometry())
            if transform is not None:
                try:
                    geom.transform(transform)
                except Exception:
                    continue
            if geom and not geom.isEmpty():
                aoi_geoms.append(geom)
        if not aoi_geoms:
            return features
        aoi_union = QgsGeometry.unaryUnion(aoi_geoms)

        kept = []
        for feat in features:
            g = feat.get('geometry')
            if g is None:
                continue
            if aoi_union.intersects(g):
                kept.append(feat)
        return kept

    def create_damage_layer(self, change_features, reference_layer, layer_name):
        """Create the damage vector layer."""
        crs = reference_layer.crs()
        crs_string = crs.authid()

        layer = QgsVectorLayer(
            f"Polygon?crs={crs_string}",
            layer_name,
            "memory")

        provider = layer.dataProvider()

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("damage_class", QVariant.Int))
        fields.append(QgsField("damage_name", QVariant.String))
        fields.append(QgsField("area_m2", QVariant.Double))
        fields.append(QgsField("confidence", QVariant.Double))
        provider.addAttributes(fields)
        layer.updateFields()

        features = []
        for idx, feature_data in enumerate(change_features):
            feature = QgsFeature()
            feature.setGeometry(feature_data['geometry'])
            feature.setAttributes([
                idx + 1,
                feature_data.get('damage_class', 4),
                feature_data.get('damage_name', 'Unknown'),
                round(feature_data['area'], 2),
                round(float(feature_data.get('confidence', 0.0)), 4),
            ])
            features.append(feature)

        provider.addFeatures(features)
        layer.updateExtents()

        # Categorized renderer by damage class
        categories = []

        symbol1 = QgsFillSymbol.createSimple({
            'color': '0,255,0,255',
            'outline_color': '#00aa00',
            'outline_width': '0.5'
        })
        categories.append(QgsRendererCategory(1, symbol1, 'No Damage'))

        symbol2 = QgsFillSymbol.createSimple({
            'color': '255,255,0,255',
            'outline_color': '#cccc00',
            'outline_width': '0.5'
        })
        categories.append(QgsRendererCategory(2, symbol2, 'Minor Damage'))

        symbol3 = QgsFillSymbol.createSimple({
            'color': '255,165,0,255',
            'outline_color': '#cc8800',
            'outline_width': '0.5'
        })
        categories.append(QgsRendererCategory(3, symbol3, 'Major Damage'))

        symbol4 = QgsFillSymbol.createSimple({
            'color': '255,0,0,255',
            'outline_color': '#cc0000',
            'outline_width': '0.5'
        })
        categories.append(QgsRendererCategory(4, symbol4, 'Destroyed'))

        renderer = QgsCategorizedSymbolRenderer('damage_class', categories)
        layer.setRenderer(renderer)
        layer.setOpacity(0.3)

        QgsProject.instance().addMapLayer(layer)
        self.iface.mapCanvas().setExtent(layer.extent())
        self.iface.mapCanvas().refresh()

        # Cache so file exports don't have to re-find the layer by name.
        self._last_damage_layer = layer

    def _save_outputs_if_requested(self, engine, change_features,
                                    before_layer, after_layer, layer_name):
        """Dispatch GPKG / GeoTIFF / JSON sidecar exports per the dialog
        checkboxes. Each is wrapped so one failure doesn't block the others.

        Output names (when the post image is a real file):
            <post_basename>_damage.gpkg
            <post_basename>_mask.tif
            <post_basename>_meta.json"""
        if not (hasattr(self.dialog, 'get_save_gpkg')
                or hasattr(self.dialog, 'get_save_mask')):
            return

        want_gpkg = self.dialog.get_save_gpkg() if hasattr(
            self.dialog, 'get_save_gpkg') else False
        want_mask = self.dialog.get_save_mask() if hasattr(
            self.dialog, 'get_save_mask') else False
        if not (want_gpkg or want_mask):
            return

        out_dir = self._resolve_output_dir(after_layer)
        if not out_dir:
            self.iface.messageBar().pushMessage(
                "Output export",
                "Could not determine an output directory from the post "
                "image source; falling back to user home.", level=1, duration=6)
            out_dir = os.path.expanduser('~')

        base = self._derive_output_stem(after_layer, layer_name)

        saved = []
        if want_gpkg:
            try:
                gpkg_path = os.path.join(out_dir, f"{base}_damage.gpkg")
                self._save_layers_as_gpkg(gpkg_path)
                saved.append(os.path.basename(gpkg_path))
            except Exception as e:
                print(f"[Damage AI] GPKG export failed: "
                      f"{type(e).__name__}: {e}")
                self.iface.messageBar().pushMessage(
                    "Output export",
                    f"GeoPackage export failed: {type(e).__name__}: {e}",
                    level=2, duration=8)
        if want_mask:
            try:
                tif_path = os.path.join(out_dir, f"{base}_mask.tif")
                self._save_damage_mask_as_geotiff(engine, tif_path)
                saved.append(os.path.basename(tif_path))
            except Exception as e:
                print(f"[Damage AI] mask export failed: "
                      f"{type(e).__name__}: {e}")
                self.iface.messageBar().pushMessage(
                    "Output export",
                    f"Mask GeoTIFF export failed: {type(e).__name__}: {e}",
                    level=2, duration=8)
        # Provenance + perf sidecars accompany whichever files were written.
        if want_gpkg or want_mask:
            try:
                json_path = os.path.join(out_dir, f"{base}_meta.json")
                self._save_metadata_sidecar(
                    engine, before_layer, after_layer, json_path,
                    layer_name, len(change_features))
                saved.append(os.path.basename(json_path))
            except Exception as e:
                print(f"[Damage AI] metadata sidecar export failed: "
                      f"{type(e).__name__}: {e}")
            try:
                profile = getattr(engine, '_last_profile', None)
                if profile is not None:
                    import json as _json
                    perf_path = os.path.join(out_dir, f"{base}_perf.json")
                    with open(perf_path, 'w', encoding='utf-8') as f:
                        _json.dump(profile, f, indent=2)
                    saved.append(os.path.basename(perf_path))
            except Exception as e:
                print(f"[Damage AI] perf sidecar export failed: "
                      f"{type(e).__name__}: {e}")
        if saved:
            self.iface.messageBar().pushMessage(
                "Output export",
                f"Saved {', '.join(saved)} to {out_dir}",
                level=3, duration=8)

    def _derive_output_stem(self, after_layer, layer_name):
        """Filename stem for exports: prefer the post image's basename,
        falling back to a sanitized layer name."""
        try:
            src = after_layer.source()
            if src and '|' in src:
                src = src.split('|', 1)[0]
            if src and os.path.exists(src):
                stem = os.path.splitext(os.path.basename(src))[0]
                # Strip common post-disaster suffixes so the stem is shared
                # with pre + mask, enabling glob-by-stem in trainers.
                for suffix in ('_post_disaster', '_post', '-post'):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                return self._sanitize_filename(stem)
        except Exception:
            pass
        return self._sanitize_filename(layer_name)

    def _save_metadata_sidecar(self, engine, before_layer, after_layer,
                                json_path, layer_name, n_features):
        """JSON provenance sidecar for the GPKG/mask exports."""
        import json as _json
        import datetime as _dt

        def _src_of(layer):
            try:
                s = layer.source() if layer is not None else None
                if s and '|' in s:
                    s = s.split('|', 1)[0]
                return s
            except Exception:
                return None

        meta = {
            'format': 'change-detector damage-mask v1',
            'created_at_utc': _dt.datetime.utcnow().isoformat() + 'Z',
            'layer_name': layer_name,
            'n_buildings_detected': int(n_features),
            'pre_image':  _src_of(before_layer),
            'post_image': _src_of(after_layer),
            'label_format': 'xBD',
            'label_definitions': {
                '0': 'background',
                '1': 'no_damage',
                '2': 'minor_damage',
                '3': 'major_damage',
                '4': 'destroyed',
            },
            'mask_geotiff': {
                'dtype': 'uint8',
                'nodata': 0,
                'colormap': {
                    '0': 'transparent', '1': 'green',
                    '2': 'yellow', '3': 'orange', '4': 'red',
                },
            },
            'ensemble': {
                'cls_members': list(
                    getattr(engine, 'cls_member_names', None) or []),
                'cls_member_count': len(getattr(engine, 'cls_models', [])),
            },
            'post_processing': {
                'loc_threshold': float(
                    getattr(engine, 'loc_threshold', 0.4)),
                'object_voting': bool(
                    getattr(engine, 'object_voting', True)),
                'class_upgrade_rules': [
                    list(r) for r in
                    getattr(engine, 'class_upgrade_rules', [])
                ],
                'watershed_split': bool(
                    getattr(engine, 'watershed_split', True)),
                'watershed_min_distance': int(
                    getattr(engine, 'watershed_min_distance', 7)),
                'gsd_normalize': bool(
                    getattr(engine, 'gsd_normalize', True)),
                'target_gsd_m': float(
                    getattr(engine, 'target_gsd_m', 0.5)),
            },
            'notes': (
                "Pixel values in the *_mask.tif are integer class labels "
                "(0..4) matching the xBD damage-class convention. Use them "
                "directly as training targets for a per-pixel classifier; "
                "pair with the pre_image and post_image referenced above. "
                "The embedded ColorTable is purely for visual rendering and "
                "does not affect the raw label values."),
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            _json.dump(meta, f, indent=2, ensure_ascii=False)

    def _resolve_output_dir(self, raster_layer):
        """Directory of the raster's source file, or '' if it isn't a real path."""
        try:
            src = raster_layer.source()
            if src and os.path.exists(src):
                return os.path.dirname(os.path.abspath(src))
            if src and '|' in src:
                p = src.split('|', 1)[0]
                if os.path.exists(p):
                    return os.path.dirname(os.path.abspath(p))
        except Exception:
            pass
        return ''

    def _sanitize_filename(self, name):
        """Strip cross-platform unsafe filename characters."""
        bad = '<>:"/\\|?*\n\r\t'
        out = ''.join('_' if c in bad else c for c in name).strip()
        return out or 'damage_assessment'

    def _save_layers_as_gpkg(self, gpkg_path):
        """Write the main damage layer into a .gpkg ('damage_buildings' table)."""
        main = getattr(self, '_last_damage_layer', None)
        if main is None or not main.isValid():
            raise RuntimeError("No damage layer in memory to export.")

        # Remove existing file so the writer doesn't append prior-run tables.
        for suffix in ('', '-shm', '-wal', '-journal'):
            p = gpkg_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Write main layer first (CreateOrOverwriteFile), then append review.
        opts_main = QgsVectorFileWriter.SaveVectorOptions()
        opts_main.driverName = 'GPKG'
        opts_main.layerName = 'damage_buildings'
        opts_main.actionOnExistingFile = (
            QgsVectorFileWriter.CreateOrOverwriteFile)
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            main, gpkg_path,
            QgsProject.instance().transformContext(), opts_main)
        if result and result[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(
                f"GeoPackage write failed: {result[1] if len(result) > 1 else result[0]}")

    def _save_damage_mask_as_geotiff(self, engine, tif_path):
        """Write the engine's cached damage mask as a single-band uint8 GeoTIFF.
        Pixel values follow the xBD convention:
          0=background, 1=NoDmg, 2=Minor, 3=Major, 4=Destroyed."""
        try:
            from osgeo import gdal, osr
        except ImportError:
            raise RuntimeError(
                "GDAL is not available; cannot write GeoTIFF. "
                "Install gdal in the QGIS Python env to use this option.")

        mask = getattr(engine, '_last_damage_mask', None)
        extent = getattr(engine, '_last_extent', None)
        crs = getattr(engine, '_last_crs', None)
        if mask is None:
            raise RuntimeError(
                "Engine has no cached damage mask — did detection complete?")

        import numpy as _np
        mask = _np.asarray(mask, dtype=_np.uint8)
        h, w = mask.shape[:2]

        driver = gdal.GetDriverByName('GTiff')
        # Single-band uint8, raw labels 0..4 (xBD / SpaceNet convention).
        # LZW+PREDICTOR=2 + tiled 256x256 for compact, partial-read-friendly output.
        ds = driver.Create(
            tif_path, w, h, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'PREDICTOR=2',
                     'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
        if ds is None:
            raise RuntimeError(f"GDAL could not create {tif_path}")

        if extent is not None and w > 0 and h > 0:
            # GDAL geotransform: top-left origin, y resolution negative.
            x_min = extent.xMinimum()
            y_max = extent.yMaximum()
            px_w = extent.width() / float(w)
            px_h = extent.height() / float(h)
            ds.SetGeoTransform([x_min, px_w, 0.0, y_max, 0.0, -px_h])

        if crs is not None and crs.isValid():
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs.toWkt())
            ds.SetProjection(srs.ExportToWkt())

        band = ds.GetRasterBand(1)
        band.WriteArray(mask)
        band.SetNoDataValue(0)

        ds.SetMetadata({
            'CLASS_0': 'background',
            'CLASS_1': 'no_damage',
            'CLASS_2': 'minor_damage',
            'CLASS_3': 'major_damage',
            'CLASS_4': 'destroyed',
            'LABEL_FORMAT': 'xBD',
        })
        band.SetDescription('Damage class (0=bg, 1=NoDmg, 2=Minor, 3=Major, 4=Destroyed)')
        ds.FlushCache()
        ds = None
