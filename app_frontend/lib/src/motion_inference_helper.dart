import 'api_client.dart';
import 'models.dart';
import 'motion_feature_extractor.dart';

/// Validates window size against backend manifest, extracts **128-D** features, calls inference.
class MotionInferenceHelper {
  MotionInferenceHelper._();

  /// Throws [ApiException] on HTTP errors; returns parsed JSON model on success.
  static Future<MotionInferenceResponseModel> inferFromSamples(
    BackendApiClient client,
    List<SensorReadingPayload> samples, {
    bool predictFallType = false,
    List<double>? fallTypeFeatures,
    int? expectedDim,
  }) async {
    final dim = expectedDim ?? MotionFeatureExtractor.enhancedFeatureDim;
    final features = MotionFeatureExtractor.extractEnhanced(samples);
    if (features.length != dim) {
      throw ApiException(
        'Enhanced features length ${features.length} does not match expected $dim.',
      );
    }
    final sendWindows =
        predictFallType && fallTypeFeatures == null;
    final raw = await client.inferMotion(
      enhancedFeatures: features,
      fallTypeFeatures: fallTypeFeatures,
      predictFallType: predictFallType,
      accWindow: sendWindows ? MotionFeatureExtractor.accMatrix300(samples) : null,
      gyroWindow: sendWindows ? MotionFeatureExtractor.gyroMatrix300(samples) : null,
      oriWindow: sendWindows ? MotionFeatureExtractor.oriMatrix300(samples) : null,
    );
    return MotionInferenceResponseModel.fromJson(raw);
  }

  /// Optional server-side dimension check (cached by caller).
  static Future<int> fetchEnhancedDim(BackendApiClient client) async {
    final status = await client.getInferenceStatus();
    final d = status['enhanced_feature_dim'];
    if (d is int) return d;
    if (d is num) return d.toInt();
    return MotionFeatureExtractor.enhancedFeatureDim;
  }
}
