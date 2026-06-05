"""Dockable panel: download satellite/aerial tiles by lat/lon + zoom."""
import os
import tempfile
from qgis.PyQt.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                                  QLineEdit, QPushButton, QLabel, QComboBox,
                                  QGroupBox, QProgressBar, QMessageBox, QSpinBox,
                                  QDoubleSpinBox, QFileDialog, QCheckBox)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont
from qgis.core import (QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform, QgsPointXY, QgsMessageLog, Qgis)
import urllib.request
import urllib.parse
import math
import ssl


class ImageDownloadWorker(QThread):
    """Background worker: stitch web-mercator tiles into a georeferenced image."""
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def __init__(self, lat, lon, zoom, width, height, provider, output_path):
        super().__init__()
        self.lat = lat
        self.lon = lon
        self.zoom = zoom
        self.width = width
        self.height = height
        self.provider = provider
        self.output_path = output_path

        # Default trust store + full validation. Never disable — corporate-CA
        # users should install their CA system-wide, not patch this code.
        self.ssl_context = ssl.create_default_context()

    def run(self):
        try:
            self.progress.emit(10)
            self.log.emit(f"Starting download: {self.provider}")
            self.log.emit(f"Coordinates: {self.lat}, {self.lon}, Zoom: {self.zoom}")

            if self.provider == "OpenStreetMap":
                file_path = self.download_osm_tiles()
            elif self.provider == "ESRI World Imagery":
                file_path = self.download_esri_imagery()
            else:
                raise Exception(f"Unknown provider: {self.provider}")

            self.progress.emit(100)
            layer_name = f"Satellite_{self.lat:.4f}_{self.lon:.4f}"
            self.finished.emit(file_path, layer_name)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    def download_esri_imagery(self):
        """Download ESRI World Imagery tiles centered on (lat, lon)."""
        try:
            from PIL import Image
        except ImportError:
            raise Exception("Pillow library required. Install with: pip install Pillow")

        import io

        self.progress.emit(15)
        self.log.emit("Calculating tile coordinates...")

        # Lat/lon -> Web Mercator tile indices.
        n = 2 ** self.zoom
        x_tile = int((self.lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(self.lat)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

        tiles_x = max(1, (self.width + 255) // 256)
        tiles_y = max(1, (self.height + 255) // 256)

        start_x = x_tile - tiles_x // 2
        start_y = y_tile - tiles_y // 2

        self.log.emit(f"Downloading {tiles_x}x{tiles_y} tiles...")

        final_image = Image.new('RGB', (tiles_x * 256, tiles_y * 256))

        tile_count = 0
        total_tiles = tiles_x * tiles_y
        failed_tiles = 0

        for dx in range(tiles_x):
            for dy in range(tiles_y):
                tx = start_x + dx
                ty = start_y + dy

                tx = tx % n
                if ty < 0 or ty >= n:
                    continue

                url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{ty}/{tx}"

                try:
                    # Hardcoded https:// scheme above; assert the invariant
                    # for static-analyzer benefit and defense-in-depth.
                    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
                        continue
                    req = urllib.request.Request(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 QGIS-Plugin',
                        'Accept': 'image/png,image/*',
                        'Referer': 'https://www.arcgis.com/'
                    })

                    with urllib.request.urlopen(req, timeout=30, context=self.ssl_context) as response:  # nosec B310 - scheme validated above
                        content_type = response.headers.get('Content-Type', '')
                        tile_data = response.read()

                        if 'image' in content_type or len(tile_data) > 1000:
                            tile_image = Image.open(io.BytesIO(tile_data))
                            if tile_image.mode != 'RGB':
                                tile_image = tile_image.convert('RGB')
                            final_image.paste(tile_image, (dx * 256, dy * 256))
                        else:
                            self.log.emit(f"Tile {tx},{ty} returned non-image data")
                            failed_tiles += 1

                except Exception as e:
                    self.log.emit(f"Failed tile {tx},{ty}: {str(e)[:50]}")
                    failed_tiles += 1

                tile_count += 1
                progress = 15 + int(70 * tile_count / total_tiles)
                self.progress.emit(progress)

        if failed_tiles == total_tiles:
            raise Exception("All tiles failed to download. Check your internet connection.")

        if failed_tiles > 0:
            self.log.emit(f"Warning: {failed_tiles}/{total_tiles} tiles failed")

        self.progress.emit(90)
        self.log.emit("Saving image...")

        if final_image.size != (self.width, self.height):
            left = (final_image.width - self.width) // 2
            top = (final_image.height - self.height) // 2
            right = left + self.width
            bottom = top + self.height
            final_image = final_image.crop((left, top, right, bottom))

        # tempfile.NamedTemporaryFile (not mktemp) — atomic creation, no
        # TOCTOU race against symlink attacks. We close the descriptor so
        # PIL can re-open it for writing.
        if self.output_path:
            output_path = self.output_path
        else:
            tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            output_path = tf.name
            tf.close()
        final_image.save(output_path, 'PNG')

        if os.path.getsize(output_path) < 1000:
            raise Exception("Saved image is too small - download may have failed")

        self.log.emit(f"Image saved: {os.path.getsize(output_path)} bytes")
        self.create_world_file(output_path, start_x, start_y, tiles_x, tiles_y, self.zoom)
        return output_path

    def download_osm_tiles(self):
        """Download OpenStreetMap tiles centered on (lat, lon)."""
        try:
            from PIL import Image
        except ImportError:
            raise Exception("Pillow library required. Install with: pip install Pillow")

        import io

        self.progress.emit(15)
        self.log.emit("Calculating tile coordinates...")

        n = 2 ** self.zoom
        x_tile = int((self.lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(self.lat)
        y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

        tiles_x = max(1, (self.width + 255) // 256)
        tiles_y = max(1, (self.height + 255) // 256)

        start_x = x_tile - tiles_x // 2
        start_y = y_tile - tiles_y // 2

        self.log.emit(f"Downloading {tiles_x}x{tiles_y} tiles...")

        final_image = Image.new('RGB', (tiles_x * 256, tiles_y * 256))

        tile_count = 0
        total_tiles = tiles_x * tiles_y
        failed_tiles = 0

        for dx in range(tiles_x):
            for dy in range(tiles_y):
                tx = (start_x + dx) % n
                ty = start_y + dy

                if ty < 0 or ty >= n:
                    continue

                url = f"https://tile.openstreetmap.org/{self.zoom}/{tx}/{ty}.png"

                try:
                    # Hardcoded https:// scheme above; assert the invariant
                    # for static-analyzer benefit and defense-in-depth.
                    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
                        continue
                    req = urllib.request.Request(url, headers={
                        'User-Agent': 'QGIS-ChangeDetector-Plugin/1.0 (Educational)'
                    })
                    with urllib.request.urlopen(req, timeout=30, context=self.ssl_context) as response:  # nosec B310 - scheme validated above
                        tile_data = response.read()
                        tile_image = Image.open(io.BytesIO(tile_data))
                        if tile_image.mode != 'RGB':
                            tile_image = tile_image.convert('RGB')
                        final_image.paste(tile_image, (dx * 256, dy * 256))
                except Exception as e:
                    self.log.emit(f"Failed tile {tx},{ty}: {e}")
                    failed_tiles += 1

                tile_count += 1
                progress = 15 + int(70 * tile_count / total_tiles)
                self.progress.emit(progress)

        if failed_tiles == total_tiles:
            raise Exception("All tiles failed to download. Check your internet connection.")

        self.progress.emit(90)
        self.log.emit("Saving image...")

        if final_image.size != (self.width, self.height):
            left = (final_image.width - self.width) // 2
            top = (final_image.height - self.height) // 2
            final_image = final_image.crop((left, top, left + self.width, top + self.height))

        # tempfile.NamedTemporaryFile (not mktemp) — atomic creation, no
        # TOCTOU race against symlink attacks. We close the descriptor so
        # PIL can re-open it for writing.
        if self.output_path:
            output_path = self.output_path
        else:
            tf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            output_path = tf.name
            tf.close()
        final_image.save(output_path, 'PNG')

        self.log.emit(f"Image saved: {os.path.getsize(output_path)} bytes")
        self.create_world_file(output_path, start_x, start_y, tiles_x, tiles_y, self.zoom)
        return output_path

    def create_world_file(self, image_path, start_x, start_y, tiles_x, tiles_y, zoom):
        """Write .pgw + .prj sidecars so the PNG is georeferenced as EPSG:3857."""
        n = 2 ** zoom

        EARTH_CIRCUMFERENCE = 40075016.686
        ORIGIN_SHIFT = EARTH_CIRCUMFERENCE / 2.0
        tile_size_meters = EARTH_CIRCUMFERENCE / n

        x_min = start_x * tile_size_meters - ORIGIN_SHIFT
        y_max = ORIGIN_SHIFT - start_y * tile_size_meters
        x_max = (start_x + tiles_x) * tile_size_meters - ORIGIN_SHIFT
        y_min = ORIGIN_SHIFT - (start_y + tiles_y) * tile_size_meters

        total_pixel_width = tiles_x * 256
        total_pixel_height = tiles_y * 256

        crop_left = (total_pixel_width - self.width) // 2
        crop_top = (total_pixel_height - self.height) // 2

        meters_per_pixel_x = (x_max - x_min) / total_pixel_width
        meters_per_pixel_y = (y_max - y_min) / total_pixel_height

        x_min_cropped = x_min + crop_left * meters_per_pixel_x
        y_max_cropped = y_max - crop_top * meters_per_pixel_y

        pixel_width = meters_per_pixel_x
        pixel_height = -meters_per_pixel_y  # negative: image y grows downward

        x_top_left = x_min_cropped + pixel_width / 2
        y_top_left = y_max_cropped + pixel_height / 2

        world_file = image_path.rsplit('.', 1)[0] + '.pgw'
        with open(world_file, 'w') as f:
            f.write(f"{pixel_width:.10f}\n")
            f.write("0.0\n")
            f.write("0.0\n")
            f.write(f"{pixel_height:.10f}\n")
            f.write(f"{x_top_left:.6f}\n")
            f.write(f"{y_top_left:.6f}\n")

        self.log.emit(f"World file created: {world_file}")

        prj_file = image_path.rsplit('.', 1)[0] + '.prj'
        with open(prj_file, 'w') as f:
            f.write('PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
                   'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                   'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                   'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                   'AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],'
                   'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
                   'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
                   'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],'
                   'EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 '
                   '+x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"],'
                   'AUTHORITY["EPSG","3857"]]')

        self.log.emit(f"PRJ file created: {prj_file}")


class SatelliteDownloaderDock(QDockWidget):
    """Dockable satellite/aerial imagery downloader."""

    PROVIDERS = {
        "ESRI World Imagery": "Free high-resolution satellite imagery worldwide",
        "OpenStreetMap": "Free map tiles (street maps, not satellite imagery)",
    }

    def __init__(self, iface, parent=None):
        super().__init__("Satellite Imagery Downloader", parent)
        self.iface = iface
        self.worker = None

        self.setup_ui()
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setMinimumWidth(380)

    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("🛰️ Satellite Imagery Downloader")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        coord_group = QGroupBox("📍 Coordinates")
        coord_layout = QVBoxLayout(coord_group)

        lat_layout = QHBoxLayout()
        lat_layout.addWidget(QLabel("Latitude:"))
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(0.0)
        self.lat_input.setSingleStep(0.01)
        lat_layout.addWidget(self.lat_input)
        coord_layout.addLayout(lat_layout)

        lon_layout = QHBoxLayout()
        lon_layout.addWidget(QLabel("Longitude:"))
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(0.0)
        self.lon_input.setSingleStep(0.01)
        lon_layout.addWidget(self.lon_input)
        coord_layout.addLayout(lon_layout)

        get_from_map_btn = QPushButton("📌 Get from Map Center")
        get_from_map_btn.clicked.connect(self.get_coords_from_map)
        coord_layout.addWidget(get_from_map_btn)

        hint = QLabel("Enter coordinates in decimal degrees (WGS84)")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        coord_layout.addWidget(hint)

        layout.addWidget(coord_group)

        provider_group = QGroupBox("🌐 Imagery Provider")
        provider_layout = QVBoxLayout(provider_group)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(self.PROVIDERS.keys()))
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        provider_layout.addWidget(self.provider_combo)

        self.provider_description = QLabel()
        self.provider_description.setStyleSheet("color: gray; font-size: 10px;")
        self.provider_description.setWordWrap(True)
        provider_layout.addWidget(self.provider_description)

        layout.addWidget(provider_group)

        settings_group = QGroupBox("⚙️ Image Settings")
        settings_layout = QVBoxLayout(settings_group)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom Level:"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(1, 19)
        self.zoom_spin.setValue(17)
        zoom_layout.addWidget(self.zoom_spin)
        settings_layout.addLayout(zoom_layout)

        zoom_hint = QLabel("Higher = more detail (15-18 recommended for buildings)")
        zoom_hint.setStyleSheet("color: gray; font-size: 10px;")
        settings_layout.addWidget(zoom_hint)

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setValue(1024)
        self.width_spin.setSingleStep(256)
        dim_layout.addWidget(self.width_spin)

        dim_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setValue(1024)
        self.height_spin.setSingleStep(256)
        dim_layout.addWidget(self.height_spin)
        settings_layout.addLayout(dim_layout)

        self.add_to_project_cb = QCheckBox("Add to current project")
        self.add_to_project_cb.setChecked(True)
        settings_layout.addWidget(self.add_to_project_cb)

        layout.addWidget(settings_group)

        output_group = QGroupBox("📁 Output")
        output_layout = QVBoxLayout(output_group)

        output_path_layout = QHBoxLayout()
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("Leave empty for temporary file...")
        output_path_layout.addWidget(self.output_path_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_output)
        output_path_layout.addWidget(browse_btn)

        output_layout.addLayout(output_path_layout)

        layout.addWidget(output_group)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.log_group = QGroupBox("📋 Download Log")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(False)
        log_layout = QVBoxLayout(self.log_group)

        self.log_text = QLabel("")
        self.log_text.setWordWrap(True)
        self.log_text.setStyleSheet("font-size: 10px; color: #666; background: #f5f5f5; padding: 5px;")
        log_layout.addWidget(self.log_text)

        layout.addWidget(self.log_group)

        self.download_btn = QPushButton("⬇️ Download Imagery")
        self.download_btn.setMinimumHeight(40)
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.download_btn.clicked.connect(self.start_download)
        layout.addWidget(self.download_btn)

        layout.addStretch()

        self.setWidget(main_widget)

        self.on_provider_changed(self.provider_combo.currentText())

    def on_provider_changed(self, provider):
        if provider in self.PROVIDERS:
            self.provider_description.setText(self.PROVIDERS[provider])

    def get_coords_from_map(self):
        """Populate lat/lon from the current map canvas center (WGS84)."""
        canvas = self.iface.mapCanvas()
        center = canvas.center()

        canvas_crs = canvas.mapSettings().destinationCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")

        if canvas_crs != wgs84:
            transform = QgsCoordinateTransform(canvas_crs, wgs84, QgsProject.instance())
            center = transform.transform(center)

        self.lat_input.setValue(center.y())
        self.lon_input.setValue(center.x())

        self.status_label.setText(f"Got coordinates from map: {center.y():.6f}, {center.x():.6f}")

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Imagery As",
            f"satellite_{self.lat_input.value():.4f}_{self.lon_input.value():.4f}.png",
            "PNG Files (*.png);;TIFF Files (*.tif);;All Files (*.*)"
        )
        if file_path:
            self.output_path_input.setText(file_path)

    def start_download(self):
        lat = self.lat_input.value()
        lon = self.lon_input.value()

        if lat == 0 and lon == 0:
            reply = QMessageBox.question(
                self, "Confirm Coordinates",
                "Coordinates are set to 0, 0 (Gulf of Guinea). Is this correct?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "This feature requires the Pillow library.\n\n"
                "Install it with: pip install Pillow"
            )
            return

        self.download_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting download...")
        self.log_text.setText("")

        output_path = self.output_path_input.text() or None

        self.worker = ImageDownloadWorker(
            lat=lat,
            lon=lon,
            zoom=self.zoom_spin.value(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            provider=self.provider_combo.currentText(),
            output_path=output_path
        )

        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.on_log)
        self.worker.finished.connect(self.on_download_finished)
        self.worker.error.connect(self.on_download_error)
        self.worker.start()

    def on_progress(self, value):
        self.progress.setValue(value)
        if value < 30:
            self.status_label.setText("Preparing download...")
        elif value < 85:
            self.status_label.setText("Downloading tiles...")
        else:
            self.status_label.setText("Processing image...")

    def on_log(self, message):
        current = self.log_text.text()
        self.log_text.setText(current + message + "\n")

    def on_download_finished(self, file_path, layer_name):
        self.download_btn.setEnabled(True)
        self.progress.setVisible(False)

        self.status_label.setText(f"✅ Downloaded: {file_path}")

        if self.add_to_project_cb.isChecked():
            layer = QgsRasterLayer(file_path, layer_name)

            if layer.isValid():
                layer.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))

                QgsProject.instance().addMapLayer(layer)
                self.iface.mapCanvas().setExtent(layer.extent())
                self.iface.mapCanvas().refresh()

                QMessageBox.information(
                    self, "Success",
                    f"Satellite imagery downloaded and added to project!\n\n"
                    f"File: {file_path}\n"
                    f"Layer: {layer_name}"
                )
            else:
                QMessageBox.warning(
                    self, "Layer Error",
                    f"Image downloaded but couldn't be loaded as a layer.\n\n"
                    f"File saved to: {file_path}\n\n"
                    f"Try opening the file manually in QGIS."
                )
        else:
            QMessageBox.information(
                self, "Success",
                f"Satellite imagery downloaded!\n\nFile: {file_path}"
            )

    def on_download_error(self, error_msg):
        self.download_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"❌ Error occurred")

        self.log_group.setChecked(True)
        self.log_text.setText(self.log_text.text() + f"\n❌ ERROR:\n{error_msg}")

        QMessageBox.critical(
            self, "Download Failed",
            f"Failed to download imagery:\n\n{error_msg[:500]}"
        )