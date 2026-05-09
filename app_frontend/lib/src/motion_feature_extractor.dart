import 'dart:math' as math;

import 'models.dart';

/// Colab / Python `scripts/baseline_fall/enhanced_features.py` — **128** floats per window.
class MotionFeatureExtractor {
  MotionFeatureExtractor._();

  /// Must match training script window size (128 samples @ ~50 Hz).
  static const int windowLength = 128;
  static const int fallTypeWindowLength = 300;
  static const int enhancedFeatureDim = 128;
  static const double sampleRateHz = 50.0;

  /// Build **128** features from one window of sensor readings.
  /// If [samples] has length ≠ 128, values are linearly resampled to 128 per axis.
  static List<double> extractEnhanced(List<SensorReadingPayload> samples) {
    if (samples.length < 2) {
      throw ArgumentError('Need at least 2 samples to form a window.');
    }

    final n = samples.length;
    final accX = List<double>.generate(n, (i) => samples[i].accX);
    final accY = List<double>.generate(n, (i) => samples[i].accY);
    final accZ = List<double>.generate(n, (i) => samples[i].accZ);
    final gX = List<double>.generate(n, (i) => samples[i].gyroX);
    final gY = List<double>.generate(n, (i) => samples[i].gyroY);
    final gZ = List<double>.generate(n, (i) => samples[i].gyroZ);

    final ax = n == windowLength ? accX : _resampleSeries(accX, windowLength);
    final ay = n == windowLength ? accY : _resampleSeries(accY, windowLength);
    final az = n == windowLength ? accZ : _resampleSeries(accZ, windowLength);
    final gx = n == windowLength ? gX : _resampleSeries(gX, windowLength);
    final gy = n == windowLength ? gY : _resampleSeries(gY, windowLength);
    final gz = n == windowLength ? gZ : _resampleSeries(gZ, windowLength);

    final oa = List<double>.generate(n, (i) => samples[i].azimuth ?? 0.0);
    final ob = List<double>.generate(n, (i) => samples[i].pitch ?? 0.0);
    final oc = List<double>.generate(n, (i) => samples[i].roll ?? 0.0);
    final ox = n == windowLength ? oa : _resampleSeries(oa, windowLength);
    final oy = n == windowLength ? ob : _resampleSeries(ob, windowLength);
    final oz = n == windowLength ? oc : _resampleSeries(oc, windowLength);
    final ori = <List<double>>[ox, oy, oz];

    final acc = <List<double>>[ax, ay, az];
    final gyro = <List<double>>[gx, gy, gz];

    final feat = <double>[];

    feat.addAll(_timeDomainFeatures(acc));
    feat.addAll(_crossAxisCorrelations(acc));
    feat.addAll(_magnitudeFeatures(acc));
    feat.addAll(_frequencyDomainFeatures(acc, sampleRateHz));

    feat.addAll(_timeDomainFeatures(gyro));
    feat.addAll(_crossAxisCorrelations(gyro));
    feat.addAll(_magnitudeFeatures(gyro));
    feat.addAll(_frequencyDomainFeatures(gyro, sampleRateHz));

    feat.addAll(_orientationStats(ori));

    if (feat.length != enhancedFeatureDim) {
      throw StateError('Feature length ${feat.length} != $enhancedFeatureDim');
    }
    return feat;
  }

  /// **300×3** rows for `POST /api/v1/inference/motion` `acc_window` / `gyro_window` (server fall-type path).
  static List<List<double>> accMatrix300(List<SensorReadingPayload> samples) =>
      _sensorMatrix(samples, _accTriplets, fallTypeWindowLength);

  /// Gyro columns for the same API (rad/s).
  static List<List<double>> gyroMatrix300(List<SensorReadingPayload> samples) =>
      _sensorMatrix(samples, _gyroTriplets, fallTypeWindowLength);

  /// Orientation columns (degrees, MobiAct azimuth / pitch / roll) for fall-type server features.
  static List<List<double>> oriMatrix300(List<SensorReadingPayload> samples) =>
      _sensorMatrix(samples, _oriTriplets, fallTypeWindowLength);

  static List<double> _accTriplets(SensorReadingPayload s) =>
      <double>[s.accX, s.accY, s.accZ];

  static List<double> _gyroTriplets(SensorReadingPayload s) =>
      <double>[s.gyroX, s.gyroY, s.gyroZ];

  static List<double> _oriTriplets(SensorReadingPayload s) =>
      <double>[s.azimuth ?? 0.0, s.pitch ?? 0.0, s.roll ?? 0.0];

  static List<List<double>> _sensorMatrix(
    List<SensorReadingPayload> samples,
    List<double> Function(SensorReadingPayload) trip,
    int targetLength,
  ) {
    if (samples.length < 2) {
      throw ArgumentError('Need at least 2 samples to form a window.');
    }
    final n = samples.length;
    final c0 = List<double>.generate(n, (i) => trip(samples[i])[0]);
    final c1 = List<double>.generate(n, (i) => trip(samples[i])[1]);
    final c2 = List<double>.generate(n, (i) => trip(samples[i])[2]);
    final r0 = n == targetLength ? c0 : _resampleSeries(c0, targetLength);
    final r1 = n == targetLength ? c1 : _resampleSeries(c1, targetLength);
    final r2 = n == targetLength ? c2 : _resampleSeries(c2, targetLength);
    return List<List<double>>.generate(
      targetLength,
      (i) => <double>[r0[i], r1[i], r2[i]],
    );
  }

  static List<double> _resampleSeries(List<double> src, int targetLen) {
    if (src.length == targetLen) return List<double>.from(src);
    final out = <double>[];
    final last = src.length - 1;
    for (var t = 0; t < targetLen; t++) {
      final u = last * t / (targetLen - 1);
      final i = u.floor();
      final j = (i + 1).clamp(0, last);
      final f = u - i;
      out.add(src[i] * (1 - f) + src[j] * f);
    }
    return out;
  }

  static List<double> _timeDomainFeatures(List<List<double>> data) {
    final out = <double>[];
    for (var axis = 0; axis < data.length; axis++) {
      final x = data[axis];
      out.addAll([
        _mean(x),
        _std(x),
        _median(x),
        x.reduce(math.min),
        x.reduce(math.max),
        _ptp(x),
        _percentile(x, 5),
        _percentile(x, 25),
        _percentile(x, 75),
        _percentile(x, 95),
        math.sqrt(x.map((e) => e * e).reduce((a, b) => a + b) / x.length),
        _meanAbsDiff(x),
        _sumAbsDiff(x),
        _variance(x),
        x.map((e) => e * e).reduce((a, b) => a + b) / x.length,
      ]);
    }
    return out;
  }

  static List<double> _crossAxisCorrelations(List<List<double>> data) {
    return <double>[
      _corr(data[0], data[1]),
      _corr(data[0], data[2]),
      _corr(data[1], data[2]),
    ];
  }

  static List<double> _magnitudeFeatures(List<List<double>> data) {
    final mag = List<double>.generate(
      data[0].length,
      (i) => math.sqrt(
        data[0][i] * data[0][i] + data[1][i] * data[1][i] + data[2][i] * data[2][i],
      ),
    );
    return <double>[
      _mean(mag),
      _std(mag),
      mag.reduce(math.max),
      _percentile(mag, 95),
      mag.fold<double>(0.0, (a, b) => a + b),
    ];
  }

  static List<double> _frequencyDomainFeatures(
    List<List<double>> data,
    double fs,
  ) {
    final out = <double>[];
    for (var axis = 0; axis < data.length; axis++) {
      final fftMag = _rfftMagnitudes(data[axis]);
      if (fftMag.length <= 1) {
        out.addAll([0.0, 0.0]);
        continue;
      }
      var maxIdx = 1;
      for (var i = 2; i < fftMag.length; i++) {
        if (fftMag[i] > fftMag[maxIdx]) {
          maxIdx = i;
        }
      }
      final domFreq = maxIdx * fs / data[axis].length;
      var spectralEnergy = 0.0;
      for (final v in fftMag) {
        spectralEnergy += v * v;
      }
      spectralEnergy /= fftMag.length;
      out.addAll([domFreq, spectralEnergy]);
    }
    return out;
  }

  static List<double> _orientationStats(List<List<double>> data) {
    final out = <double>[];
    for (var axis = 0; axis < data.length; axis++) {
      final x = data[axis];
      out.addAll([_mean(x), _std(x), _ptp(x)]);
    }
    final azimuthRad = data[0].map((v) => v * math.pi / 180.0).toList();
    final cMean = _mean(azimuthRad.map(math.cos).toList());
    final sMean = _mean(azimuthRad.map(math.sin).toList());
    final mrl = math.sqrt(cMean * cMean + sMean * sMean);
    out.add(mrl);
    return out;
  }

  /// Naive real FFT magnitudes for indices 0..n/2 (inclusive).
  static List<double> _rfftMagnitudes(List<double> x) {
    final n = x.length;
    final bins = n ~/ 2 + 1;
    if (bins <= 0) return [];
    final out = List<double>.filled(bins, 0.0);
    const tau = 2 * math.pi;
    for (var k = 0; k < bins; k++) {
      double re = 0, im = 0;
      for (var t = 0; t < n; t++) {
        final angle = -tau * k * t / n;
        re += x[t] * math.cos(angle);
        im += x[t] * math.sin(angle);
      }
      out[k] = math.sqrt(re * re + im * im);
    }
    return out;
  }

  static double _mean(List<double> d) =>
      d.isEmpty ? 0.0 : d.fold<double>(0.0, (a, b) => a + b) / d.length;

  static double _variance(List<double> d) {
    if (d.length < 2) return 0.0;
    final m = _mean(d);
    return d.map((v) => (v - m) * (v - m)).reduce((a, b) => a + b) / d.length;
  }

  static double _std(List<double> d) => math.sqrt(_variance(d));

  static double _median(List<double> d) {
    final s = List<double>.from(d)..sort();
    final n = s.length;
    if (n == 0) return 0.0;
    if (n.isOdd) return s[n ~/ 2];
    return (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
  }

  static double _ptp(List<double> d) =>
      d.isEmpty ? 0.0 : d.reduce(math.max) - d.reduce(math.min);

  static double _percentile(List<double> d, int p) {
    if (d.isEmpty) return 0.0;
    final s = List<double>.from(d)..sort();
    final idx = (p / 100.0) * (s.length - 1);
    final lo = idx.floor();
    final hi = idx.ceil().clamp(0, s.length - 1);
    final f = idx - lo;
    return s[lo] * (1 - f) + s[hi] * f;
  }

  static double _meanAbsDiff(List<double> d) {
    if (d.length < 2) return 0.0;
    var sum = 0.0;
    for (var i = 0; i < d.length - 1; i++) {
      sum += (d[i + 1] - d[i]).abs();
    }
    return sum / (d.length - 1);
  }

  static double _sumAbsDiff(List<double> d) {
    if (d.length < 2) return 0.0;
    var sum = 0.0;
    for (var i = 0; i < d.length - 1; i++) {
      sum += (d[i + 1] - d[i]).abs();
    }
    return sum;
  }

  static double _corr(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;
    final ma = _mean(a);
    final mb = _mean(b);
    var num = 0.0, da = 0.0, db = 0.0;
    for (var i = 0; i < a.length; i++) {
      final xa = a[i] - ma;
      final xb = b[i] - mb;
      num += xa * xb;
      da += xa * xa;
      db += xb * xb;
    }
    final den = math.sqrt(da * db);
    if (den < 1e-12) return 0.0;
    final c = num / den;
    return c.isNaN ? 0.0 : c;
  }
}
