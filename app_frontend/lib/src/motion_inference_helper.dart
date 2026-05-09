import 'api_client.dart';
import 'package:flutter/foundation.dart';
import 'models.dart';
import 'motion_feature_extractor.dart';

/// Validates window size against backend manifest, extracts **128-D** features, calls inference.
class MotionInferenceHelper {
  MotionInferenceHelper._();

  static const bool _enableInferenceDebugLogs = true;

  /// Throws [ApiException] on HTTP errors; returns parsed JSON model on success.
  static Future<MotionInferenceResponseModel> inferFromSamples(
    BackendApiClient client,
    List<SensorReadingPayload> samples, {
    bool predictFallType = false,
    List<double>? fallTypeFeatures,
    int? expectedDim,
  }) async {
    final dim = expectedDim ?? MotionFeatureExtractor.enhancedFeatureDim;

    if (_enableInferenceDebugLogs) {
      final first = samples.isNotEmpty ? samples.first.toJson() : null;
      final last = samples.isNotEmpty ? samples.last.toJson() : null;
      debugPrint(
        '[MotionInference] Sensor values received: '
        'count=${samples.length}, first=$first, last=$last',
      );
    }

    final features = MotionFeatureExtractor.extractEnhanced(samples);
    if (_enableInferenceDebugLogs) {
      debugPrint(
        '[MotionInference] Extracted features: '
        'dim=${features.length}, preview=${_previewVector(features)}',
      );
    }
    if (features.length != dim) {
      throw ApiException(
        'Enhanced features length ${features.length} does not match expected $dim.',
      );
    }
    final sendWindows =
        predictFallType && fallTypeFeatures == null;
    if (_enableInferenceDebugLogs) {
      debugPrint(
        '[MotionInference] Model input: '
        'enhancedDim=${features.length}, '
        'predictFallType=$predictFallType, '
        'fallTypeFeaturesDim=${fallTypeFeatures?.length ?? 0}, '
        'sendAccGyroOriWindows=$sendWindows',
      );
    }
    final raw = await client.inferMotion(
      enhancedFeatures: features,
      fallTypeFeatures: fallTypeFeatures,
      predictFallType: predictFallType,
      accWindow: sendWindows ? MotionFeatureExtractor.accMatrix300(samples) : null,
      gyroWindow: sendWindows ? MotionFeatureExtractor.gyroMatrix300(samples) : null,
      oriWindow: sendWindows ? MotionFeatureExtractor.oriMatrix300(samples) : null,
    );
    final parsed = MotionInferenceResponseModel.fromJson(raw);
    if (_enableInferenceDebugLogs) {
      debugPrint(
        '[MotionInference] Model output: '
        'isFall=${parsed.isFall}, '
        'fallProbability=${parsed.fallProbability.toStringAsFixed(4)}, '
        'branch=${parsed.branch}, '
        'activity=${parsed.activityLabel}, '
        'fallType=${parsed.fallTypeLabel ?? parsed.fallTypeCode ?? '-'}, '
        'summary="${parsed.summaryLine}"',
      );
    }
    return parsed;
  }

  /// Optional server-side dimension check (cached by caller).
  static Future<int> fetchEnhancedDim(BackendApiClient client) async {
    final status = await client.getInferenceStatus();
    final d = status['enhanced_feature_dim'];
    if (d is int) return d;
    if (d is num) return d.toInt();
    return MotionFeatureExtractor.enhancedFeatureDim;
  }

  static String _previewVector(List<double> values, {int maxItems = 10}) {
    final limit = values.length < maxItems ? values.length : maxItems;
    final head = values
        .take(limit)
        .map((v) => v.toStringAsFixed(5))
        .join(', ');
    if (values.length <= maxItems) {
      return '[$head]';
    }
    return '[$head, ...]';
  }
}
