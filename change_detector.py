import os
import datetime
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.core import (QgsProject, QgsVectorLayer, QgsFeature, QgsField, QgsFields,
                       QgsFillSymbol, QgsCategorizedSymbolRenderer, QgsRendererCategory,
                       QgsGeometry, QgsCoordinateTransform, QgsVectorFileWriter,
                       QgsCoordinateReferenceSystem, QgsRectangle,
                       QgsTask, QgsApplication,
                       QgsMessageLog, Qgis)
from qgis.PyQt.QtCore import QVariant
from .change_detector_dialog import ChangeDetectorDialog
from .building_damage_engine import BuildingDamageEngine, DetectionCancelledError
from .assistant import AssessmentReportsDock
from .satellite_downloader import SatelliteDownloaderDock
from .processing_provider import BeaconGISProcessingProvider


class _LayerSnapshot:
    """Thread-safe stand-in for a QgsRasterLayer, captured on the main
    thread before the detection task starts. The engine only calls these
    five accessors (pixel data is read directly via GDAL on the source
    path), so the background task never touches a live QGIS layer object
    from the worker thread."""

    def __init__(self, layer):
        self._source = layer.source()
        self._crs = QgsCoordinateReferenceSystem(layer.crs())
        self._extent = QgsRectangle(layer.extent())
        self._width = layer.width()
        self._height = layer.height()

    def source(self):
        return self._source

    def crs(self):
        return self._crs

    def extent(self):
        return QgsRectangle(self._extent)

    def width(self):
        return self._width

    def height(self):
        return self._height


class DamageDetectionTask(QgsTask):
    """Background damage detection. run() executes in QGIS's task thread
    pool (first-run weights download + inference + polygonization), with
    progress and cancellation wired into the engine; finished() hands the
    results back to the plugin on the main thread."""

    def __init__(self, description, engine, before_snap, after_snap,
                 tta_mode, on_done):
        super().__init__(description, QgsTask.CanCancel)
        self.engine = engine
        self.before_snap = before_snap
        self.after_snap = after_snap
        self.tta_mode = tta_mode
        self.on_done = on_done
        self.features = None
        self.error = None
        self.was_cancelled = False

    def run(self):
        try:
            self.engine.progress_callback = self.setProgress
            self.engine.cancel_callback = self.isCanceled
            self.features = self.engine.detect(
                self.before_snap, self.after_snap,
                sensitivity=50,
                tta_mode=self.tta_mode,
            )
            return True
        except DetectionCancelledError:
            self.was_cancelled = True
            return False
        except Exception as e:
            import traceback
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            return False
        finally:
            self.engine.progress_callback = None
            self.engine.cancel_callback = None

    def finished(self, result):
        # Runs on the main thread — safe to touch the GUI and QgsProject.
        self.on_done(self, result)


class ChangeDetector:
    """QGIS plugin for AI building damage detection from pre/post imagery."""

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.dialog = None
        self.assistant_dock = None
        self.satellite_dock = None
        self.actions = []
        self.provider = None
        self._active_task = None
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

    def initProcessing(self):
        self.provider = BeaconGISProcessingProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.add_action(icon_path, text='Detect Building Damage',
                        callback=self.run, parent=self.iface.mainWindow())

        self.add_action(icon_path, text='Assessment Reports',
                        callback=self.show_assistant, parent=self.iface.mainWindow())

        self.add_action(icon_path, text='Satellite Downloader',
                        callback=self.show_satellite_downloader, parent=self.iface.mainWindow())

    def unload(self):
        # A still-running background task must not outlive the plugin.
        if self._active_task is not None:
            try:
                self._active_task.cancel()
            except Exception:
                pass
            self._active_task = None

        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

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
        if self._active_task is not None:
            self.iface.messageBar().pushMessage(
                "Building Damage",
                "A detection is already running — watch its progress in the "
                "task manager (bottom status bar) or cancel it there first.",
                level=Qgis.Warning, duration=6)
            return

        if self.dialog is None:
            self.dialog = ChangeDetectorDialog(self.iface)

        if self.dialog.exec():
            self.start_damage_detection()

    def start_damage_detection(self):
        """Validate inputs, snapshot everything the worker needs, and hand
        detection to a background QgsTask. The UI stays responsive; results
        come back via _on_detection_done on the main thread."""
        before_layer = self.dialog.get_damage_before_layer()
        after_layer = self.dialog.get_damage_after_layer()

        if not all([before_layer, after_layer]):
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

        # Captured on the main thread so the worker never has to touch
        # QgsProject.instance().
        engine._transform_context = QgsProject.instance().transformContext()

        tta_mode = 'off'
        if hasattr(self.dialog, 'get_tta_mode'):
            tta_mode = self.dialog.get_tta_mode()
        elif hasattr(self.dialog, 'get_use_tta'):
            tta_mode = '4' if self.dialog.get_use_tta() else 'off'

        if hasattr(self.dialog, 'get_segmentation_min_distance'):
            seg_md = self.dialog.get_segmentation_min_distance()
            if seg_md:
                engine.watershed_min_distance = int(seg_md)

        # Auto-timestamped output layer name so multiple runs in the same
        # session don't collide on top of each other.
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        layer_name = f"Building Damage {timestamp}"

        aoi_layer = None
        if hasattr(self.dialog, 'get_aoi_layer'):
            aoi_layer = self.dialog.get_aoi_layer()

        # Snapshot dialog state NOW — the user may reopen the dialog and
        # change selections while the task runs.
        run_ctx = {
            'layer_name': layer_name,
            'before_snap': _LayerSnapshot(before_layer),
            'after_snap': _LayerSnapshot(after_layer),
            'aoi_layer_id': aoi_layer.id() if aoi_layer is not None else None,
            'want_gpkg': (self.dialog.get_save_gpkg()
                          if hasattr(self.dialog, 'get_save_gpkg') else False),
            'want_mask': (self.dialog.get_save_mask()
                          if hasattr(self.dialog, 'get_save_mask') else False),
        }

        task = DamageDetectionTask(
            f"beaconGIS — {layer_name}",
            engine,
            run_ctx['before_snap'],
            run_ctx['after_snap'],
            tta_mode,
            on_done=lambda t, ok, ctx=run_ctx: self._on_detection_done(t, ok, ctx))

        self._active_task = task
        self.iface.messageBar().pushMessage(
            "Building Damage",
            "Detecting building damage in the background — QGIS stays "
            "responsive. Progress and Cancel are in the task manager "
            "(bottom status bar).",
            level=Qgis.Info, duration=8)
        QgsApplication.taskManager().addTask(task)

    def _on_detection_done(self, task, ok, ctx):
        """Main-thread completion handler: builds the layer, clips to AOI,
        runs file exports, and feeds the Assessment Reports panel."""
        self._active_task = None
        engine = task.engine
        layer_name = ctx['layer_name']

        try:
            if task.was_cancelled:
                self.iface.messageBar().pushMessage(
                    "Building Damage", "Detection cancelled.",
                    level=Qgis.Info, duration=5)
                return

            if not ok or task.features is None:
                QMessageBox.critical(
                    None, "Model Error",
                    f"Could not run AI building damage model.\n\n"
                    f"Details:\n{task.error or 'unknown error'}\n\n"
                    f"Check the 'Damage AI' log panel (View → Panels → "
                    f"Log Messages) for the full traceback.")
                return

            change_features = task.features
            if not change_features:
                QMessageBox.information(None, "Result",
                    "No buildings detected in the imagery.")
                return

            # AOI clip: drop polygons outside the user-supplied region. The
            # AOI layer is resolved by id on the main thread — it may have
            # been removed from the project while the task ran.
            if ctx['aoi_layer_id']:
                aoi_layer = QgsProject.instance().mapLayer(ctx['aoi_layer_id'])
                if aoi_layer is not None:
                    before_n = len(change_features)
                    change_features = self._clip_to_aoi(
                        change_features, aoi_layer, ctx['before_snap'])
                    print(f"[Damage AI] AOI clip: {before_n} -> "
                          f"{len(change_features)} polygons")
                    if not change_features:
                        QMessageBox.information(None, "Result",
                            "No buildings inside the AOI.")
                        return

            self.create_damage_layer(
                change_features, ctx['before_snap'], layer_name)

            # Optional file exports (Output Options checkboxes).
            try:
                self._save_outputs_if_requested(
                    engine, change_features,
                    ctx['before_snap'], ctx['after_snap'],
                    layer_name, ctx['want_gpkg'], ctx['want_mask'])
            except Exception as e:
                # Non-critical: notify but keep the detection result on screen.
                print(f"[Damage AI] output export failed: "
                      f"{type(e).__name__}: {e}")
                self.iface.messageBar().pushMessage(
                    "Output export",
                    f"Could not write all requested output files: "
                    f"{type(e).__name__}: {e}",
                    level=Qgis.Warning, duration=8)

            # Open the assistant panel and pass results (handles its own
            # stats + CSV export).
            self.show_assistant()
            if self.assistant_dock:
                self.assistant_dock.set_assessment_data(
                    change_features, layer_name)

            self.iface.messageBar().pushMessage(
                "Success",
                f"Detected {len(change_features)} buildings!",
                level=Qgis.Success, duration=5)

        except Exception as e:
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
                                    before_layer, after_layer, layer_name,
                                    want_gpkg, want_mask):
        """Dispatch GPKG / GeoTIFF / JSON sidecar exports per the dialog
        checkboxes (snapshotted when the run started). Each is wrapped so
        one failure doesn't block the others.

        Output names (when the post image is a real file):
            <post_basename>_damage.gpkg
            <post_basename>_mask.tif
            <post_basename>_meta.json"""
        if not (want_gpkg or want_mask):
            return

        out_dir = self._resolve_output_dir(after_layer)
        if not out_dir:
            self.iface.messageBar().pushMessage(
                "Output export",
                "Could not determine an output directory from the post "
                "image source; falling back to user home.",
                level=Qgis.Warning, duration=6)
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
                    level=Qgis.Critical, duration=8)
        if want_mask:
            try:
                tif_path = os.path.join(out_dir, f"{base}_mask.tif")
                engine.save_damage_mask_geotiff(tif_path)
                saved.append(os.path.basename(tif_path))
            except Exception as e:
                print(f"[Damage AI] mask export failed: "
                      f"{type(e).__name__}: {e}")
                self.iface.messageBar().pushMessage(
                    "Output export",
                    f"Mask GeoTIFF export failed: {type(e).__name__}: {e}",
                    level=Qgis.Critical, duration=8)
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
                level=Qgis.Success, duration=8)

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
            'created_at_utc': _dt.datetime.now(
                _dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
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
                'coregister': bool(
                    getattr(engine, 'coregister', True)),
                'coreg_tile_refine': bool(
                    getattr(engine, 'coreg_tile_refine', True)),
                'cls_temperature': float(
                    getattr(engine, 'cls_temperature', 1.0)),
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
