import 'api_client.dart';
import 'models.dart';
import 'motion_feature_extractor.dart';

/// Sends raw sensor windows to the server for 144-D feature extraction and XGBoost inference.
/// The server always rebuilds features from raw windows (training parity with NumPy FFT).
class MotionInferenceHelper {
  MotionInferenceHelper._();

  static const bool _enableInferenceDebugLogs = true;

  /// Throws [ApiException] on HTTP errors; returns parsed inference result on success.
  static Future<MotionInferenceResponseModel> inferFromSamples(
    BackendApiClient client,
    List<SensorReadingPayload> samples, {
    bool predictFallType = false,
    List<double>? fallTypeFeatures,
  }) async {
    if (_enableInferenceDebugLogs) {
      final first = samples.isNotEmpty ? samples.first.toJson() : null;
      final last = samples.isNotEmpty ? samples.last.toJson() : null;
      print(
        '[MotionInference] Sending ${samples.length} samples to server '
        '(server builds 144-D features): first=$first, last=$last',
      );
    }

    final sendFallTypeWindows = predictFallType && fallTypeFeatures == null;
    final raw = await client.inferMotion(
      fallTypeFeatures: fallTypeFeatures,
      predictFallType: predictFallType,
      accWindow: sendFallTypeWindows
          ? MotionFeatureExtractor.accMatrix300(samples)
          : MotionFeatureExtractor.accMatrix128(samples),
      gyroWindow: sendFallTypeWindows
          ? MotionFeatureExtractor.gyroMatrix300(samples)
          : MotionFeatureExtractor.gyroMatrix128(samples),
      oriWindow: sendFallTypeWindows
          ? MotionFeatureExtractor.oriMatrix300(samples)
          : MotionFeatureExtractor.oriMatrix128(samples),
    );
    final parsed = MotionInferenceResponseModel.fromJson(raw);
    if (_enableInferenceDebugLogs) {
      print(
        '[MotionInference] Result: '
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
}
