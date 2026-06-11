"""QGIS Processing provider for beaconGIS.

Exposes damage detection as a Processing algorithm, which buys three
things over the dialog for free:
  - batch mode (Processing toolbox right-click -> Execute as Batch),
  - headless runs via `qgis_process run beacongis:detectbuildingdamage`,
  - Model Designer integration (chain with clip / zonal stats / export).

The algorithm body delegates to BuildingDamageEngine — the same code path
the toolbar dialog uses — with Processing's feedback object wired into the
engine's progress/cancel hooks.
"""
import os

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingProvider,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsWkbTypes,
)


class BeaconGISProcessingProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(DetectBuildingDamageAlgorithm())

    def id(self):
        return 'beacongis'

    def name(self):
        return 'beaconGIS'

    def longName(self):
        return 'beaconGIS — Building Damage Assessment'

    def icon(self):
        path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(path) if os.path.exists(path) else super().icon()


class DetectBuildingDamageAlgorithm(QgsProcessingAlgorithm):
    """Pre/post RGB pair -> damage-classified building polygons
    (+ optional per-pixel damage mask GeoTIFF)."""

    PRE = 'PRE_IMAGE'
    POST = 'POST_IMAGE'
    TTA = 'USE_TTA'
    FAST = 'FAST_MODE'
    SEPARATION = 'BUILDING_SEPARATION'
    OUTPUT = 'OUTPUT'
    MASK = 'MASK_OUTPUT'

    def tr(self, string):
        return QCoreApplication.translate('DetectBuildingDamage', string)

    def createInstance(self):
        return DetectBuildingDamageAlgorithm()

    def name(self):
        return 'detectbuildingdamage'

    def displayName(self):
        return self.tr('Detect building damage')

    def group(self):
        return self.tr('Damage assessment')

    def groupId(self):
        return 'damageassessment'

    def icon(self):
        path = os.path.join(os.path.dirname(__file__), 'icon.png')
        return QIcon(path) if os.path.exists(path) else super().icon()

    def shortHelpString(self):
        return self.tr(
            'Classifies buildings into No Damage / Minor / Major / Destroyed '
            'from a pre/post-disaster RGB image pair, using the beaconGIS '
            'Siamese U-Net ensemble on ONNX Runtime.\n\n'
            'Output is one polygon per detected building with damage_class '
            '(1-4), damage_name, area_m2, and a confidence attribute. The '
            'optional mask output writes the per-pixel damage classes as a '
            'single-band uint8 GeoTIFF in the xBD label convention '
            '(0=background, 1=No Damage, 2=Minor, 3=Major, 4=Destroyed).\n\n'
            'The pre and post rasters are aligned automatically (CRS-aware '
            'warp, GSD normalization to ~0.5 m/px, and pre/post '
            'co-registration). On first run, the model weights (~234 MB) '
            'are downloaded from the plugin\'s GitHub releases page.\n\n'
            'Outputs are draft assessments for human review — not '
            'authoritative ground truth.')

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.PRE, self.tr('Pre-disaster RGB image')))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.POST, self.tr('Post-disaster RGB image')))
        self.addParameter(QgsProcessingParameterBoolean(
            self.TTA,
            self.tr('Test-time augmentation (4-way, ~4x slower, small '
                    'accuracy bump)'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FAST,
            self.tr('Fast mode (single model, no TTA — fastest, slight '
                    'accuracy cost)'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEPARATION,
            self.tr('Building separation (0-100, higher = touching '
                    'buildings split more aggressively)'),
            QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=0, maxValue=100))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr('Damage polygons'),
            QgsProcessing.TypeVectorPolygon))
        mask_param = QgsProcessingParameterRasterDestination(
            self.MASK, self.tr('Damage mask GeoTIFF (xBD labels 0-4)'),
            optional=True, createByDefault=False)
        self.addParameter(mask_param)

    def processAlgorithm(self, parameters, context, feedback):
        from .building_damage_engine import (BuildingDamageEngine,
                                             DetectionCancelledError)

        pre = self.parameterAsRasterLayer(parameters, self.PRE, context)
        post = self.parameterAsRasterLayer(parameters, self.POST, context)
        if pre is None or post is None:
            raise QgsProcessingException(
                self.tr('Both the pre and post raster are required.'))

        engine = BuildingDamageEngine()
        engine.cpu_fast_mode = self.parameterAsBool(
            parameters, self.FAST, context)
        engine._transform_context = context.transformContext()

        # Same slider->px mapping the dialog uses: 0 -> 24 px, 50 -> 14 px,
        # 100 -> 4 px minimum distance between watershed markers.
        sep = self.parameterAsInt(parameters, self.SEPARATION, context)
        engine.watershed_min_distance = int(round(4 + (100 - sep) / 100.0 * 20))

        tta_mode = ('4' if self.parameterAsBool(parameters, self.TTA, context)
                    else 'off')

        engine.progress_callback = feedback.setProgress
        engine.cancel_callback = feedback.isCanceled
        try:
            features = engine.detect(pre, post,
                                     sensitivity=50, tta_mode=tta_mode)
        except DetectionCancelledError:
            raise QgsProcessingException(self.tr('Cancelled.'))
        finally:
            engine.progress_callback = None
            engine.cancel_callback = None

        fields = QgsFields()
        fields.append(QgsField('id', QVariant.Int))
        fields.append(QgsField('damage_class', QVariant.Int))
        fields.append(QgsField('damage_name', QVariant.String))
        fields.append(QgsField('area_m2', QVariant.Double))
        fields.append(QgsField('confidence', QVariant.Double))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.Polygon, pre.crs())
        if sink is None:
            raise QgsProcessingException(
                self.invalidSinkError(parameters, self.OUTPUT))

        for idx, fd in enumerate(features, start=1):
            f = QgsFeature(fields)
            f.setGeometry(fd['geometry'])
            f.setAttributes([
                idx,
                fd.get('damage_class', 4),
                fd.get('damage_name', 'Unknown'),
                round(float(fd.get('area', 0.0)), 2),
                round(float(fd.get('confidence', 0.0)), 4),
            ])
            sink.addFeature(f, QgsFeatureSink.FastInsert)
        feedback.pushInfo(self.tr(
            '{n} buildings detected.').format(n=len(features)))

        results = {self.OUTPUT: dest_id}

        mask_path = self.parameterAsOutputLayer(parameters, self.MASK, context)
        if mask_path:
            engine.save_damage_mask_geotiff(mask_path)
            results[self.MASK] = mask_path
            feedback.pushInfo(self.tr('Damage mask written to {p}').format(
                p=mask_path))

        return results
