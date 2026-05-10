import 'dart:async';
import 'dart:math' as math;

import 'package:motion_core/motion_core.dart';
import 'package:sensors_plus/sensors_plus.dart';

import 'models.dart';

typedef SensorBatchCallback =
    Future<void> Function(List<SensorReadingPayload> samples);

/// Convert fused quaternion attitude to MobiAct CSV orientation (degrees).
/// MobiAct `ori`: Azimuth(Z), Pitch(X), Roll(Y). [MotionDataEuler]: yaw(Z), pitch(Y), roll(X) in radians.
({double azimuthDeg, double pitchDeg, double rollDeg}) mobiActOrientationFromMotion(
  MotionData m,
) {
  final yaw = m.yaw;
  final pitchY = m.pitch;
  final rollX = m.roll;
  final radToDeg = 180.0 / math.pi;
  return (
    azimuthDeg: _normalizeAzimuthDeg(yaw * radToDeg),
    pitchDeg: rollX * radToDeg,
    rollDeg: pitchY * radToDeg,
  );
}

double _normalizeAzimuthDeg(double d) {
  var x = d % 360.0;
  if (x < 0) x += 360.0;
  return x;
}

class SensorStreamingService {
  SensorStreamingService({
    this.targetSamplingRateHz = 50.0,
    this.windowSize = 128,
    this.stepSize = 64,
  });

  final double targetSamplingRateHz;
  final int windowSize;
  final int stepSize;

  // ── Phase 1a: Gyroscope EMA low-pass filter ──────────────────────────────
  // α = 0.30 keeps real rotation, attenuates high-freq jitter that makes
  // the model think a standing person is walking / running.
  static const double _gyroEmaAlpha = 0.30;

  // ── Phase 1b: Accelerometer gravity removal ───────────────────────────────
  // Very-slow EMA (α = 0.05) tracks the slowly-changing gravity vector as
  // the phone orientation changes. Subtracting it yields linear acceleration.
  // A second EMA (α = 0.25) smooths residual noise on the linear signal.
  //
  // IMPORTANT: raw acc is still stored in the buffer so the ingest pipeline
  // (fall detection) keeps the full gravity spike during a real fall.
  // The filtered linear acc is stored in _filtLinAcc* and exposed via
  // the payload's filtAccX/Y/Z fields for use by the feature extractor.
  static const double _gravEmaAlpha = 0.05;
  static const double _linAccEmaAlpha = 0.25;

  // Gyro EMA state
  double _filtGyroX = 0.0;
  double _filtGyroY = 0.0;
  double _filtGyroZ = 0.0;

  // Gravity estimate state
  double _gravX = 0.0;
  double _gravY = 0.0;
  double _gravZ = 0.0;

  // Smoothed linear acceleration state
  double _filtLinAccX = 0.0;
  double _filtLinAccY = 0.0;
  double _filtLinAccZ = 0.0;

  // Whether the gravity EMA has been seeded with the first sample
  bool _gravInitialized = false;

  final List<SensorReadingPayload> _buffer = <SensorReadingPayload>[];

  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;
  StreamSubscription<MotionData>? _motionSubscription;

  double _latestGyroX = 0.0;
  double _latestGyroY = 0.0;
  double _latestGyroZ = 0.0;
  double? _latestAzimuthDeg;
  double? _latestPitchDeg;
  double? _latestRollDeg;

  int _lastSampleTimestampMs = 0;
  bool _isFlushing = false;
  bool _isRunning = false;

  bool get isRunning => _isRunning;
  int get bufferedSamples => _buffer.length;

  Future<SensorAccessStatus> probeSensors({
    Duration timeout = const Duration(seconds: 2),
  }) async {
    final accelerometerAvailable = await _checkStreamAvailable(
      accelerometerEventStream(),
      timeout,
    );
    final gyroscopeAvailable = await _checkStreamAvailable(
      gyroscopeEventStream(),
      timeout,
    );

    var fusedOrientationAvailable = false;
    try {
      fusedOrientationAvailable = await MotionCore.isAvailable();
    } catch (_) {
      fusedOrientationAvailable = false;
    }

    return SensorAccessStatus(
      accelerometerAvailable: accelerometerAvailable,
      gyroscopeAvailable: gyroscopeAvailable,
      fusedOrientationAvailable: fusedOrientationAvailable,
      checkedAt: DateTime.now(),
    );
  }

  Future<void> start(SensorBatchCallback onBatch) async {
    if (_isRunning) {
      return;
    }

    // Reset buffer and filter state for a clean session.
    _buffer.clear();
    _lastSampleTimestampMs = 0;
    _latestAzimuthDeg = null;
    _latestPitchDeg = null;
    _latestRollDeg = null;
    _filtGyroX = 0.0;
    _filtGyroY = 0.0;
    _filtGyroZ = 0.0;
    _gravInitialized = false;
    _gravX = 0.0;
    _gravY = 0.0;
    _gravZ = 0.0;
    _filtLinAccX = 0.0;
    _filtLinAccY = 0.0;
    _filtLinAccZ = 0.0;
    _isRunning = true;

    // ── Gyroscope subscription with EMA filter (Phase 1a) ──────────────────
    _gyroscopeSubscription = gyroscopeEventStream().listen(
      (event) {
        // Apply EMA low-pass filter to remove high-frequency rotation jitter.
        _filtGyroX =
            _gyroEmaAlpha * event.x + (1 - _gyroEmaAlpha) * _filtGyroX;
        _filtGyroY =
            _gyroEmaAlpha * event.y + (1 - _gyroEmaAlpha) * _filtGyroY;
        _filtGyroZ =
            _gyroEmaAlpha * event.z + (1 - _gyroEmaAlpha) * _filtGyroZ;
        _latestGyroX = _filtGyroX;
        _latestGyroY = _filtGyroY;
        _latestGyroZ = _filtGyroZ;
      },
      onError: (_) {
        _latestGyroX = 0.0;
        _latestGyroY = 0.0;
        _latestGyroZ = 0.0;
      },
      cancelOnError: false,
    );

    try {
      if (await MotionCore.isAvailable()) {
        _motionSubscription = MotionCore.motionStream.listen(
          (MotionData data) {
            if (!_isRunning) return;
            final o = mobiActOrientationFromMotion(data);
            _latestAzimuthDeg = o.azimuthDeg;
            _latestPitchDeg = o.pitchDeg;
            _latestRollDeg = o.rollDeg;
          },
          onError: (_) {},
          cancelOnError: false,
        );
      }
    } catch (_) {}

    // ── Accelerometer subscription with gravity removal (Phase 1b) ──────────
    _accelerometerSubscription = accelerometerEventStream().listen(
      (event) {
        final nowMs = DateTime.now().millisecondsSinceEpoch;
        final minGapMs = (1000 / targetSamplingRateHz).round();
        if (_lastSampleTimestampMs != 0 &&
            nowMs - _lastSampleTimestampMs < minGapMs) {
          return;
        }
        _lastSampleTimestampMs = nowMs;

        // Seed gravity on the very first sample so the filter converges fast.
        if (!_gravInitialized) {
          _gravX = event.x;
          _gravY = event.y;
          _gravZ = event.z;
          _filtLinAccX = 0.0;
          _filtLinAccY = 0.0;
          _filtLinAccZ = 0.0;
          _gravInitialized = true;
        }

        // Update very-slow gravity EMA (tracks orientation, not movement).
        _gravX = _gravEmaAlpha * event.x + (1 - _gravEmaAlpha) * _gravX;
        _gravY = _gravEmaAlpha * event.y + (1 - _gravEmaAlpha) * _gravY;
        _gravZ = _gravEmaAlpha * event.z + (1 - _gravEmaAlpha) * _gravZ;

        // Linear acceleration = total - gravity estimate.
        final linX = event.x - _gravX;
        final linY = event.y - _gravY;
        final linZ = event.z - _gravZ;

        // Smooth the linear acceleration to remove residual noise.
        _filtLinAccX =
            _linAccEmaAlpha * linX + (1 - _linAccEmaAlpha) * _filtLinAccX;
        _filtLinAccY =
            _linAccEmaAlpha * linY + (1 - _linAccEmaAlpha) * _filtLinAccY;
        _filtLinAccZ =
            _linAccEmaAlpha * linZ + (1 - _linAccEmaAlpha) * _filtLinAccZ;

        // Buffer stores raw acc (needed for fall-detection g-spike) plus the
        // gravity-removed / filtered linear acc for activity classification.
        _buffer.add(
          SensorReadingPayload(
            timestampMs: nowMs,
            // Raw total acc — sent to ingest for fall detection.
            accX: event.x,
            accY: event.y,
            accZ: event.z,
            // Filtered linear acc — used by feature extractor for activity.
            filtAccX: _filtLinAccX,
            filtAccY: _filtLinAccY,
            filtAccZ: _filtLinAccZ,
            // Filtered gyro (Phase 1a).
            gyroX: _latestGyroX,
            gyroY: _latestGyroY,
            gyroZ: _latestGyroZ,
            azimuth: _latestAzimuthDeg,
            pitch: _latestPitchDeg,
            roll: _latestRollDeg,
          ),
        );

        if (_buffer.length >= windowSize) {
          _flush(onBatch);
        }
      },
      onError: (_) {
        _isRunning = false;
      },
      cancelOnError: true,
    );
  }

  Future<void> stop() async {
    _isRunning = false;
    await _motionSubscription?.cancel();
    _motionSubscription = null;
    await _accelerometerSubscription?.cancel();
    await _gyroscopeSubscription?.cancel();
    _accelerometerSubscription = null;
    _gyroscopeSubscription = null;
    _buffer.clear();
  }

  void _flush(SensorBatchCallback onBatch) {
    if (_isFlushing || _buffer.length < windowSize) {
      return;
    }

    _isFlushing = true;
    final batch = List<SensorReadingPayload>.from(_buffer.take(windowSize));
    final overlapStep = stepSize < 1
        ? 1
        : (stepSize > windowSize ? windowSize : stepSize);
    _buffer.removeRange(0, overlapStep);

    onBatch(batch).whenComplete(() {
      _isFlushing = false;
      if (_isRunning && _buffer.length >= windowSize) {
        _flush(onBatch);
      }
    });
  }

  Future<bool> _checkStreamAvailable<T>(
    Stream<T> stream,
    Duration timeout,
  ) async {
    final completer = Completer<bool>();
    StreamSubscription<T>? subscription;
    Timer? timer;

    void finish(bool value) {
      if (completer.isCompleted) {
        return;
      }
      completer.complete(value);
      timer?.cancel();
      subscription?.cancel();
    }

    subscription = stream.listen(
      (_) => finish(true),
      onError: (_) => finish(false),
      cancelOnError: true,
    );

    timer = Timer(timeout, () => finish(false));
    return completer.future;
  }
}
