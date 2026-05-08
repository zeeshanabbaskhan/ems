class ApiException implements Exception {
  ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// FastAPI / SQLite may encode numbers as int, double, or string; some maps mix in strings (e.g. `branch`).
int? _parseIntLoose(dynamic v) {
  if (v == null) return null;
  if (v is int) return v;
  if (v is num) return v.toInt();
  if (v is String) return int.tryParse(v.trim());
  return null;
}

int _parseIntLooseWithDefault(dynamic v, [int fallback = 0]) =>
    _parseIntLoose(v) ?? fallback;

double? _parseDoubleLoose(dynamic v) {
  if (v == null) return null;
  if (v is num) return v.toDouble();
  if (v is String) return double.tryParse(v.trim());
  if (v is bool) return v ? 1.0 : 0.0;
  return null;
}

double _parseDoubleLooseWithDefault(dynamic v, [double fallback = 0.0]) =>
    _parseDoubleLoose(v) ?? fallback;

bool? _parseBoolLoose(dynamic v) {
  if (v == null) return null;
  if (v is bool) return v;
  if (v is num) return v != 0;
  if (v is String) {
    final s = v.trim().toLowerCase();
    if (s == 'true' || s == '1' || s == 'yes') return true;
    if (s == 'false' || s == '0' || s == 'no') return false;
  }
  return null;
}

const Map<String, String> _adlCodeToName = <String, String>{
  'STD': 'Standing',
  'WAL': 'Walking',
  'JOG': 'Jogging',
  'JUM': 'Jumping',
  'STU': 'Stairs Up',
  'STN': 'Stairs Down',
  'SCH': 'Sit to Stand',
  'SIT': 'Sitting',
  'CHU': 'Stand to Sit',
  'CSI': 'Car Step In',
  'CSO': 'Car Step Out',
  'LYI': 'Lying',
};

const Map<String, String> _fallCodeToName = <String, String>{
  'BSC': 'Back Fall',
  'FOL': 'Forward Fall',
  'FKL': 'Knees Fall',
  'SDL': 'Side Fall',
};

const Map<int, String> _adlIndexToCode = <int, String>{
  0: 'CHU',
  1: 'CSI',
  2: 'CSO',
  4: 'JOG',
  5: 'JUM',
  6: 'LYI',
  7: 'SCH',
  8: 'SIT',
  9: 'STD',
  10: 'STN',
  11: 'STU',
  12: 'WAL',
};

String? _humanizeActivityLabel(dynamic raw) {
  if (raw == null) return null;
  final s = raw.toString().trim();
  if (s.isEmpty) return null;

  final upper = s.toUpperCase();
  final direct = _adlCodeToName[upper];
  if (direct != null) return direct;
  final fallDirect = _fallCodeToName[upper];
  if (fallDirect != null) return fallDirect;

  final idx = int.tryParse(s);
  if (idx == null) return s;
  final code = _adlIndexToCode[idx];
  if (code == null) return s;
  return _adlCodeToName[code] ?? code;
}

String? _simplifyActivityLabel(String? label) {
  if (label == null) return null;
  final t = label.trim();
  if (t.isEmpty) return null;
  final lower = t.toLowerCase();

  // Keep core ADL classes stable and avoid over-merging transitions.
  if (lower.contains('jog') || lower.contains('run')) return 'Running';
  if (lower.contains('walking')) return 'Walking';
  if (lower.contains('sitting')) return 'Sitting';
  if (lower.contains('standing')) return 'Standing';
  if (lower.contains('lying')) return 'Lying';

  // Group uncommon navigation variants into walking.
  if (lower.contains('stairs') || lower.contains('car step')) return 'Walking';

  // Transitional postures are valid ADLs; don't force them into standing.
  if (lower.contains('sit to stand') || lower.contains('stand to sit')) {
    return 'Transition';
  }

  if (lower.contains('jump')) return 'Active movement';

  return t;
}

Map<String, double> _parseMetricsMap(dynamic raw) {
  if (raw is! Map) return const {};
  final out = <String, double>{};
  for (final e in raw.entries) {
    final k = e.key;
    if (k is! String) continue;
    final d = _parseDoubleLoose(e.value);
    if (d != null) {
      out[k] = d;
    }
  }
  return out;
}

enum UserRole {
  patient,
  caregiver;

  String get value => name;

  String get label {
    switch (this) {
      case UserRole.patient:
        return 'Patient';
      case UserRole.caregiver:
        return 'Caregiver';
    }
  }

  static UserRole fromWire(String raw) {
    final normalized = raw.trim().toLowerCase();
    switch (normalized) {
      case 'patient':
        return UserRole.patient;
      case 'caregiver':
        return UserRole.caregiver;
      default:
        throw ApiException('Unsupported user role: $raw');
    }
  }
}

class AuthUserProfileModel {
  AuthUserProfileModel({
    required this.userId,
    required this.email,
    required this.displayName,
    required this.availableRoles,
    this.patientId,
  });

  final String userId;
  final String email;
  final String displayName;
  final List<UserRole> availableRoles;
  final String? patientId;

  factory AuthUserProfileModel.fromJson(Map<String, dynamic> json) {
    final availableRolesJson =
        json['available_roles'] as List<dynamic>? ?? const [];
    return AuthUserProfileModel(
      userId: json['user_id'] as String? ?? '',
      email: json['email'] as String? ?? '',
      displayName: json['display_name'] as String? ?? 'User',
      availableRoles: availableRolesJson
          .map((role) => UserRole.fromWire(role.toString()))
          .toList(),
      patientId: json['patient_id'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'user_id': userId,
      'email': email,
      'display_name': displayName,
      'available_roles': availableRoles.map((role) => role.value).toList(),
      'patient_id': patientId,
    };
  }

  AuthUserProfileModel copyWith({
    String? userId,
    String? email,
    String? displayName,
    List<UserRole>? availableRoles,
    String? patientId,
    bool clearPatientId = false,
  }) {
    return AuthUserProfileModel(
      userId: userId ?? this.userId,
      email: email ?? this.email,
      displayName: displayName ?? this.displayName,
      availableRoles: availableRoles ?? this.availableRoles,
      patientId: clearPatientId ? null : (patientId ?? this.patientId),
    );
  }
}

class AuthSessionModel {
  AuthSessionModel({
    required this.accessToken,
    required this.tokenType,
    required this.selectedRole,
    required this.user,
  });

  final String accessToken;
  final String tokenType;
  final UserRole selectedRole;
  final AuthUserProfileModel user;

  factory AuthSessionModel.fromJson(Map<String, dynamic> json) {
    return AuthSessionModel(
      accessToken: json['access_token'] as String? ?? '',
      tokenType: json['token_type'] as String? ?? 'bearer',
      selectedRole: UserRole.fromWire(
        json['selected_role'] as String? ?? 'patient',
      ),
      user: AuthUserProfileModel.fromJson(
        json['user'] as Map<String, dynamic>? ?? const {},
      ),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'access_token': accessToken,
      'token_type': tokenType,
      'selected_role': selectedRole.value,
      'user': user.toJson(),
    };
  }

  AuthSessionModel copyWith({
    String? accessToken,
    String? tokenType,
    UserRole? selectedRole,
    AuthUserProfileModel? user,
  }) {
    return AuthSessionModel(
      accessToken: accessToken ?? this.accessToken,
      tokenType: tokenType ?? this.tokenType,
      selectedRole: selectedRole ?? this.selectedRole,
      user: user ?? this.user,
    );
  }
}

class PatientRecord {
  PatientRecord({
    required this.id,
    required this.fullName,
    this.age,
  });

  final String id;
  final String fullName;
  final int? age;

  factory PatientRecord.fromJson(Map<String, dynamic> json) {
    return PatientRecord(
      id: json['id'] as String,
      fullName: json['full_name'] as String? ?? 'Unknown Patient',
      age: _parseIntLoose(json['age']),
    );
  }
}

class DeviceRecord {
  DeviceRecord({required this.id, required this.label, required this.platform});

  final String id;
  final String label;
  final String platform;

  factory DeviceRecord.fromJson(Map<String, dynamic> json) {
    return DeviceRecord(
      id: json['id'] as String,
      label: json['label'] as String? ?? 'Unknown Device',
      platform: json['platform'] as String? ?? 'mobile_web',
    );
  }
}

class SessionRecord {
  SessionRecord({
    required this.id,
    required this.patientId,
    required this.deviceId,
    required this.status,
    required this.sampleRateHz,
  });

  final String id;
  final String patientId;
  final String deviceId;
  final String status;
  final double sampleRateHz;

  factory SessionRecord.fromJson(Map<String, dynamic> json) {
    return SessionRecord(
      id: json['id'] as String,
      patientId: json['patient_id'] as String,
      deviceId: json['device_id'] as String,
      status: json['status'] as String? ?? 'active',
      sampleRateHz: _parseDoubleLooseWithDefault(json['sample_rate_hz'], 50.0),
    );
  }
}

class DetectionResultModel {
  DetectionResultModel({
    required this.severity,
    required this.score,
    required this.fallProbability,
    this.predictedActivityClass,
    this.frailtyProxyScore,
    this.gaitStabilityScore,
    this.movementDisorderScore,
    required this.peakAccG,
    required this.peakGyroDps,
    required this.peakJerkGps,
    required this.stillnessRatio,
    required this.samplesAnalyzed,
    required this.message,
    required this.reasons,
  });

  final String severity;
  final double score;
  final double fallProbability;
  final String? predictedActivityClass;
  final double? frailtyProxyScore;
  final double? gaitStabilityScore;
  final double? movementDisorderScore;
  final double peakAccG;
  final double peakGyroDps;
  final double peakJerkGps;
  final double stillnessRatio;
  final int samplesAnalyzed;
  final String message;
  final List<String> reasons;

  factory DetectionResultModel.fromJson(Map<String, dynamic> json) {
    final reasonsJson = json['reasons'] as List<dynamic>? ?? const [];
    return DetectionResultModel(
      severity: json['severity'] as String? ?? 'low',
      score: _parseDoubleLooseWithDefault(json['score']),
      fallProbability: _parseDoubleLooseWithDefault(json['fall_probability']),
      predictedActivityClass: _simplifyActivityLabel(
        _humanizeActivityLabel(json['predicted_activity_class']),
      ),
      frailtyProxyScore: _parseDoubleLoose(json['frailty_proxy_score']),
      gaitStabilityScore: _parseDoubleLoose(json['gait_stability_score']),
      movementDisorderScore: _parseDoubleLoose(json['movement_disorder_score']),
      peakAccG: _parseDoubleLooseWithDefault(json['peak_acc_g']),
      peakGyroDps: _parseDoubleLooseWithDefault(json['peak_gyro_dps']),
      peakJerkGps: _parseDoubleLooseWithDefault(json['peak_jerk_g_per_s']),
      stillnessRatio: _parseDoubleLooseWithDefault(json['stillness_ratio']),
      samplesAnalyzed: _parseIntLooseWithDefault(json['samples_analyzed']),
      message: json['message'] as String? ?? 'No message',
      reasons: reasonsJson.map((item) => item.toString()).toList(),
    );
  }
}

/// Response from `POST /api/v1/inference/motion` (XGBoost pipeline).
class MotionInferenceResponseModel {
  MotionInferenceResponseModel({
    required this.isFall,
    required this.fallProbability,
    required this.fallThreshold,
    required this.branch,
    this.activityLabel,
    this.activityClassIndex,
    this.fallTypeCode,
    this.fallTypeLabel,
    this.fallTypeClassIndex,
    this.fallTypeSkippedReason,
    required this.schemaVersion,
  });

  final bool isFall;
  final double fallProbability;
  final double fallThreshold;
  final String branch;
  final String? activityLabel;
  final int? activityClassIndex;
  final String? fallTypeCode;
  final String? fallTypeLabel;
  final int? fallTypeClassIndex;
  final String? fallTypeSkippedReason;
  final String schemaVersion;

  factory MotionInferenceResponseModel.fromJson(Map<String, dynamic> json) {
    return MotionInferenceResponseModel(
      isFall: json['is_fall'] as bool? ?? false,
      fallProbability: _parseDoubleLooseWithDefault(json['fall_probability']),
      fallThreshold: _parseDoubleLooseWithDefault(json['fall_threshold'], 0.5),
      branch: json['branch'] as String? ?? 'unknown',
      activityLabel: _simplifyActivityLabel(
        _humanizeActivityLabel(json['activity_label']),
      ),
      activityClassIndex: _parseIntLoose(json['activity_class_index']),
      fallTypeCode: json['fall_type_code'] as String?,
      fallTypeLabel: json['fall_type_label'] as String?,
      fallTypeClassIndex: _parseIntLoose(json['fall_type_class_index']),
      fallTypeSkippedReason: json['fall_type_skipped_reason'] as String?,
      schemaVersion: json['schema_version'] as String? ?? '1.0',
    );
  }

  String get summaryLine {
    if (isFall) {
      final ft = fallTypeLabel ?? fallTypeCode;
      if (ft != null && ft.isNotEmpty) {
        return 'Fall detected (${(fallProbability * 100).toStringAsFixed(1)}%): type $ft';
      }
      return 'Fall detected (${(fallProbability * 100).toStringAsFixed(1)}%)';
    }
    final act = activityLabel ?? 'ADL';
    return 'No fall (${(fallProbability * 100).toStringAsFixed(1)}%): $act';
  }
}

class LiveStatusModel {
  LiveStatusModel({
    required this.patientId,
    required this.patientName,
    required this.severity,
    required this.score,
    required this.fallProbability,
    this.predictedActivityClass,
    required this.lastMessage,
    this.sessionId,
    this.deviceId,
    this.sampleRateHz,
    this.latestMetrics = const <String, double>{},
    this.activeAlertIds = const <String>[],
    this.latitude,
    this.longitude,
    this.locationAccuracyM,
    this.locationUpdatedAt,
    this.headingDegrees,
  });

  final String patientId;
  final String patientName;
  final String? sessionId;
  final String? deviceId;
  final String severity;
  final double score;
  final double fallProbability;
  final String? predictedActivityClass;
  final String lastMessage;
  final double? sampleRateHz;
  final Map<String, double> latestMetrics;
  final List<String> activeAlertIds;
  /// Last GPS point shared by the elder device (see POST `/patients/me/location`).
  final double? latitude;
  final double? longitude;
  final double? locationAccuracyM;
  final DateTime? locationUpdatedAt;
  /// Compass / course over ground (degrees), when provided by the device.
  final double? headingDegrees;

  bool get hasLiveLocation =>
      latitude != null && longitude != null;

  factory LiveStatusModel.fromJson(Map<String, dynamic> json) {
    final alertsJson = json['active_alert_ids'] as List<dynamic>? ?? const [];
    return LiveStatusModel(
      patientId: json['patient_id'] as String? ?? '',
      patientName: json['patient_name'] as String? ?? 'Unknown Patient',
      sessionId: json['session_id'] as String?,
      deviceId: json['device_id'] as String?,
      severity: json['severity'] as String? ?? 'low',
      score: _parseDoubleLooseWithDefault(json['score']),
      fallProbability: _parseDoubleLooseWithDefault(json['fall_probability']),
      predictedActivityClass: _simplifyActivityLabel(
        _humanizeActivityLabel(json['predicted_activity_class']),
      ),
      lastMessage: json['last_message'] as String? ?? 'No live status yet.',
      sampleRateHz: _parseDoubleLoose(json['sample_rate_hz']),
      latestMetrics: _parseMetricsMap(json['latest_metrics']),
      activeAlertIds: alertsJson.map((item) => item.toString()).toList(),
      latitude: _parseDoubleLoose(json['latitude']),
      longitude: _parseDoubleLoose(json['longitude']),
      locationAccuracyM: _parseDoubleLoose(json['location_accuracy_m']),
      locationUpdatedAt: DateTime.tryParse(json['location_updated_at'] as String? ?? ''),
      headingDegrees: _parseDoubleLoose(json['heading_degrees']),
    );
  }
}

class AlertRecordModel {
  AlertRecordModel({
    required this.id,
    required this.patientId,
    required this.severity,
    required this.status,
    required this.message,
    required this.score,
    this.createdAt,
    this.acknowledgedAt,
    this.resolvedAt,
    this.manuallyTriggered = false,
    this.alarmEligible,
  });

  final String id;
  final String patientId;
  final String severity;
  final String status;
  final String message;
  final double score;
  final DateTime? createdAt;
  final DateTime? acknowledgedAt;
  final DateTime? resolvedAt;
  final bool manuallyTriggered;
  final bool? alarmEligible;

  factory AlertRecordModel.fromJson(Map<String, dynamic> json) {
    return AlertRecordModel(
      id: json['id'] as String,
      patientId: json['patient_id'] as String? ?? '',
      severity: json['severity'] as String? ?? 'low',
      status: json['status'] as String? ?? 'open',
      message: json['message'] as String? ?? 'Alert',
      score: _parseDoubleLooseWithDefault(json['score']),
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? ''),
      acknowledgedAt: DateTime.tryParse(json['acknowledged_at'] as String? ?? ''),
      resolvedAt: DateTime.tryParse(json['resolved_at'] as String? ?? ''),
      manuallyTriggered: json['manually_triggered'] as bool? ?? false,
      alarmEligible: _parseBoolLoose(json['alarm_eligible']),
    );
  }
}

class SystemSummaryModel {
  SystemSummaryModel({
    required this.totalPatients,
    required this.activeSessions,
    required this.openAlerts,
    this.lastEventAt,
  });

  final int totalPatients;
  final int activeSessions;
  final int openAlerts;
  final DateTime? lastEventAt;

  factory SystemSummaryModel.fromJson(Map<String, dynamic> json) {
    return SystemSummaryModel(
      totalPatients: _parseIntLooseWithDefault(json['total_patients']),
      activeSessions: _parseIntLooseWithDefault(json['active_sessions']),
      openAlerts: _parseIntLooseWithDefault(json['open_alerts']),
      lastEventAt: DateTime.tryParse(json['last_event_at'] as String? ?? ''),
    );
  }
}

class SensorAccessStatus {
  SensorAccessStatus({
    required this.accelerometerAvailable,
    required this.gyroscopeAvailable,
    required this.checkedAt,
    this.fusedOrientationAvailable = false,
  });

  final bool accelerometerAvailable;
  final bool gyroscopeAvailable;
  /// True when [motion_core] rotation-vector / Core Motion fusion is available (optional).
  final bool fusedOrientationAvailable;
  final DateTime checkedAt;

  bool get allAvailable => accelerometerAvailable && gyroscopeAvailable;
}

class SensorReadingPayload {
  SensorReadingPayload({
    required this.timestampMs,
    required this.accX,
    required this.accY,
    required this.accZ,
    required this.gyroX,
    required this.gyroY,
    required this.gyroZ,
    this.azimuth,
    this.pitch,
    this.roll,
  });

  final int timestampMs;
  final double accX;
  final double accY;
  final double accZ;
  final double gyroX;
  final double gyroY;
  final double gyroZ;
  /// MobiAct `*_ori_*.txt` convention: degrees. Optional; omit in JSON if unknown.
  final double? azimuth;
  final double? pitch;
  final double? roll;

  factory SensorReadingPayload.fromJson(Map<String, dynamic> json) {
    return SensorReadingPayload(
      timestampMs: _parseIntLooseWithDefault(json['timestamp_ms']),
      accX: _parseDoubleLooseWithDefault(json['acc_x']),
      accY: _parseDoubleLooseWithDefault(json['acc_y']),
      accZ: _parseDoubleLooseWithDefault(json['acc_z']),
      gyroX: _parseDoubleLooseWithDefault(json['gyro_x']),
      gyroY: _parseDoubleLooseWithDefault(json['gyro_y']),
      gyroZ: _parseDoubleLooseWithDefault(json['gyro_z']),
      azimuth: _parseDoubleLoose(json['azimuth']),
      pitch: _parseDoubleLoose(json['pitch']),
      roll: _parseDoubleLoose(json['roll']),
    );
  }

  Map<String, dynamic> toJson() {
    final m = <String, dynamic>{
      'timestamp_ms': timestampMs,
      'acc_x': accX,
      'acc_y': accY,
      'acc_z': accZ,
      'gyro_x': gyroX,
      'gyro_y': gyroY,
      'gyro_z': gyroZ,
    };
    if (azimuth != null) m['azimuth'] = azimuth!;
    if (pitch != null) m['pitch'] = pitch!;
    if (roll != null) m['roll'] = roll!;
    return m;
  }
}

class TelemetrySnapshotModel {
  TelemetrySnapshotModel({
    required this.patientId,
    required this.patientName,
    required this.sessionId,
    required this.deviceId,
    required this.source,
    required this.samplingRateHz,
    required this.accelerationUnit,
    required this.gyroscopeUnit,
    required this.receivedAt,
    required this.samplesInLastBatch,
    required this.latestSamples,
    this.batteryLevel,
  });

  final String patientId;
  final String patientName;
  final String sessionId;
  final String deviceId;
  final String source;
  final double samplingRateHz;
  final String accelerationUnit;
  final String gyroscopeUnit;
  final double? batteryLevel;
  final DateTime receivedAt;
  final int samplesInLastBatch;
  final List<SensorReadingPayload> latestSamples;

  factory TelemetrySnapshotModel.fromJson(Map<String, dynamic> json) {
    final samples = json['latest_samples'] as List<dynamic>? ?? const [];
    return TelemetrySnapshotModel(
      patientId: json['patient_id'] as String? ?? '',
      patientName: json['patient_name'] as String? ?? 'Unknown Patient',
      sessionId: json['session_id'] as String? ?? '',
      deviceId: json['device_id'] as String? ?? '',
      source: json['source'] as String? ?? 'mobile',
      samplingRateHz: _parseDoubleLooseWithDefault(json['sampling_rate_hz']),
      accelerationUnit: json['acceleration_unit'] as String? ?? 'm_s2',
      gyroscopeUnit: json['gyroscope_unit'] as String? ?? 'rad_s',
      batteryLevel: _parseDoubleLoose(json['battery_level']),
      receivedAt:
          DateTime.tryParse(json['received_at'] as String? ?? '') ??
          DateTime.now(),
      samplesInLastBatch: _parseIntLooseWithDefault(json['samples_in_last_batch']),
      latestSamples: samples
          .map(
            (item) =>
                SensorReadingPayload.fromJson(item as Map<String, dynamic>),
          )
          .toList(),
    );
  }
}

class IngestResponseModel {
  IngestResponseModel({
    required this.ingestedSamples,
    required this.detection,
    required this.liveStatus,
    this.activeAlert,
    this.telemetry,
  });

  final int ingestedSamples;
  final DetectionResultModel detection;
  final LiveStatusModel liveStatus;
  final AlertRecordModel? activeAlert;
  final TelemetrySnapshotModel? telemetry;

  factory IngestResponseModel.fromJson(Map<String, dynamic> json) {
    return IngestResponseModel(
      ingestedSamples: _parseIntLooseWithDefault(json['ingested_samples']),
      detection: DetectionResultModel.fromJson(
        json['detection'] as Map<String, dynamic>? ?? const {},
      ),
      liveStatus: LiveStatusModel.fromJson(
        json['live_status'] as Map<String, dynamic>? ?? const {},
      ),
      activeAlert: json['active_alert'] == null
          ? null
          : AlertRecordModel.fromJson(
              json['active_alert'] as Map<String, dynamic>,
            ),
      telemetry: json['telemetry'] == null
          ? null
          : TelemetrySnapshotModel.fromJson(
              json['telemetry'] as Map<String, dynamic>,
            ),
    );
  }
}

class CaregiverAuthModel {
  CaregiverAuthModel({
    required this.accessToken,
    required this.caregiverId,
    required this.caregiverName,
    required this.caregiverEmail,
  });

  final String accessToken;
  final String caregiverId;
  final String caregiverName;
  final String caregiverEmail;

  factory CaregiverAuthModel.fromJson(Map<String, dynamic> json) {
    final caregiver = json['caregiver'] as Map<String, dynamic>? ?? const {};
    return CaregiverAuthModel(
      accessToken: json['access_token'] as String? ?? '',
      caregiverId: caregiver['id'] as String? ?? '',
      caregiverName: caregiver['full_name'] as String? ?? 'Caregiver',
      caregiverEmail: caregiver['email'] as String? ?? '',
    );
  }
}

class CaregiverAssignedPatientModel {
  CaregiverAssignedPatientModel({
    required this.id,
    required this.fullName,
    this.age,
  });

  final String id;
  final String fullName;
  final int? age;

  factory CaregiverAssignedPatientModel.fromJson(Map<String, dynamic> json) {
    return CaregiverAssignedPatientModel(
      id: json['id'] as String? ?? '',
      fullName: json['full_name'] as String? ?? 'Patient',
      age: _parseIntLoose(json['age']),
    );
  }
}

class GeneratedPatientCredentialModel {
  GeneratedPatientCredentialModel({
    required this.patientId,
    required this.patientName,
    required this.homeAddress,
    required this.username,
    required this.temporaryPassword,
  });

  final String patientId;
  final String patientName;
  final String homeAddress;
  final String username;
  final String temporaryPassword;

  factory GeneratedPatientCredentialModel.fromJson(Map<String, dynamic> json) {
    return GeneratedPatientCredentialModel(
      patientId: json['patient_id'] as String? ?? '',
      patientName: json['patient_name'] as String? ?? '',
      homeAddress: json['home_address'] as String? ?? '',
      username: json['username'] as String? ?? '',
      temporaryPassword: json['temporary_password'] as String? ?? '',
    );
  }
}
