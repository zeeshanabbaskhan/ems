import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'http_client_factory.dart';
import 'models.dart';

class BackendApiClient {
  BackendApiClient({required String baseUrl})
      : _baseUrl = _normalizeBaseUrl(baseUrl),
        _httpClient = createBackendHttpClient();

  /// Mobile/cellular paths can exceed a few seconds; keep below typical OS limits.
  static const Duration _requestTimeout = Duration(seconds: 25);
  static const int _sendAttempts = 2;

  final http.Client _httpClient;
  String _baseUrl;
  String? _bearerToken;

  void setBearerToken(String? token) {
    _bearerToken = (token == null || token.isEmpty) ? null : token;
  }

  Map<String, String> _headers({bool jsonBody = false}) {
    final h = <String, String>{};
    if (jsonBody) {
      h['Content-Type'] = 'application/json';
    }
    final t = _bearerToken;
    if (t != null && t.isNotEmpty) {
      h['Authorization'] = 'Bearer $t';
    }
    return h;
  }

  static String _normalizeBaseUrl(String input) {
    final trimmed = input.trim();
    final withScheme = trimmed.startsWith('http://') || trimmed.startsWith('https://')
        ? trimmed
        : 'http://$trimmed';
    if (withScheme.endsWith('/')) {
      return withScheme.substring(0, withScheme.length - 1);
    }
    return withScheme;
  }

  void updateBaseUrl(String baseUrl) {
    _baseUrl = _normalizeBaseUrl(baseUrl);
  }

  void close() {
    _httpClient.close();
  }

  Uri _uri(String path) => Uri.parse('$_baseUrl$path');

  Future<Object> _send(Future<http.Response> Function() sendRequest) async {
    for (var attempt = 0; attempt < _sendAttempts; attempt++) {
      try {
        final response = await sendRequest().timeout(_requestTimeout);
        return await _decodeResponse(response);
      } on TimeoutException {
        if (attempt >= _sendAttempts - 1) {
          throw ApiException(
            'The backend request timed out (${_requestTimeout.inSeconds}s, '
            '$_sendAttempts attempts). Base URL: $_baseUrl. Try Wi‑Fi instead of mobile data, '
            'or open $_baseUrl/api/v1/health in the phone browser.',
          );
        }
      } on http.ClientException catch (e) {
        if (attempt >= _sendAttempts - 1) {
          throw ApiException('Could not reach $_baseUrl: $e');
        }
      } catch (error) {
        if (error is ApiException) {
          rethrow;
        }
        throw ApiException('Could not reach the backend: $error');
      }
    }
    throw ApiException('Could not reach the backend (unexpected).');
  }

  Future<Object> _decodeResponse(http.Response response) async {
    final body = response.body.trim();
    final jsonBody = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      if (jsonBody is Map<String, dynamic>) {
        throw ApiException(jsonBody['detail']?.toString() ?? 'Request failed (${response.statusCode})');
      }
      throw ApiException('Request failed (${response.statusCode})');
    }

    if (jsonBody is Map<String, dynamic> || jsonBody is List<dynamic>) {
      return jsonBody as Object;
    }
    throw ApiException('Unexpected response format from backend.');
  }

  Map<String, dynamic> _asMap(Object payload) {
    if (payload is Map<String, dynamic>) {
      return payload;
    }
    throw ApiException('Unexpected response object format from backend.');
  }

  List<dynamic> _asList(Object payload) {
    if (payload is List<dynamic>) {
      return payload;
    }
    throw ApiException('Unexpected response list format from backend.');
  }

  /// Returns manifest-backed dimensions when the inference stack is loaded (503 if not).
  Future<Map<String, dynamic>> getInferenceStatus() async {
    return _asMap(
      await _send(() => _httpClient.get(_uri('/api/v1/inference/status'), headers: _headers())),
    );
  }

  Future<void> ping() async {
    await _send(() => _httpClient.get(_uri('/api/v1/health'), headers: _headers()));
  }

  Future<PatientRecord> createPatient({
    required String fullName,
    int? age,
  }) async {
    return PatientRecord.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/patients'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'full_name': fullName,
            'age': age,
          }),
        ),
      )),
    );
  }

  Future<PatientRecord> getPatient(String patientId) async {
    return PatientRecord.fromJson(
      _asMap(await _send(() => _httpClient.get(_uri('/api/v1/patients/$patientId'), headers: _headers()))),
    );
  }

  Future<DeviceRecord> createDevice({
    required String patientId,
    required String label,
    String platform = 'flutter_mobile',
    String? ownerName,
  }) async {
    return DeviceRecord.fromJson(
      _asMap(await _send(
        () => _httpClient.post(
          _uri('/api/v1/devices'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'patient_id': patientId,
            'label': label,
            'platform': platform,
            'owner_name': ownerName,
          }),
        ),
      )),
    );
  }

  Future<DeviceRecord> getDevice(String deviceId) async {
    return DeviceRecord.fromJson(
      _asMap(await _send(() => _httpClient.get(_uri('/api/v1/devices/$deviceId'), headers: _headers()))),
    );
  }

  Future<SessionRecord> startSession({
    required String patientId,
    required String deviceId,
    required double sampleRateHz,
    String startedBy = 'flutter_app',
  }) async {
    return SessionRecord.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/sessions'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'sample_rate_hz': sampleRateHz,
            'started_by': startedBy,
          }),
        ),
      )),
    );
  }

  Future<void> stopSession(String sessionId) async {
    await _send(
      () => _httpClient.post(
        _uri('/api/v1/sessions/$sessionId/stop'),
        headers: _headers(jsonBody: true),
        body: jsonEncode({
          'stopped_by': 'flutter_app',
          'note': 'Stopped from mobile app.',
        }),
      ),
    );
  }

  Future<IngestResponseModel> ingestLiveBatch({
    required String patientId,
    required String deviceId,
    required String sessionId,
    required double samplingRateHz,
    required double? batteryLevel,
    required List<SensorReadingPayload> samples,
  }) async {
    return IngestResponseModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/ingest/live'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'session_id': sessionId,
            'source': 'flutter_mobile',
            'sampling_rate_hz': samplingRateHz,
            'acceleration_unit': 'm_s2',
            'gyroscope_unit': 'rad_s',
            'battery_level': batteryLevel,
            'samples': samples.map((sample) => sample.toJson()).toList(),
          }),
        ),
      )),
    );
  }

  Future<AlertRecordModel> triggerManualAlert({
    required String patientId,
    required String? deviceId,
    required String? sessionId,
    String severity = 'fall_detected',
    String message = 'Emergency alert triggered from mobile app.',
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/alerts/manual'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'session_id': sessionId,
            'severity': severity,
            'message': message,
            'actor': 'flutter_app',
          }),
        ),
      )),
    );
  }

  /// Elder role JWT — stores last GPS on the server for the caregiver map.
  Future<void> postPatientLocation({
    required double latitude,
    required double longitude,
    double? accuracyM,
    double? headingDegrees,
    required String bearerToken,
  }) async {
    await _send(
      () => _httpClient.post(
        _uri('/api/v1/patients/me/location'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $bearerToken',
        },
        body: jsonEncode({
          'latitude': latitude,
          'longitude': longitude,
          if (accuracyM != null) 'accuracy_m': accuracyM,
          if (headingDegrees != null) 'heading_degrees': headingDegrees,
        }),
      ),
    );
  }

  Future<SystemSummaryModel> getSummary() async {
    return SystemSummaryModel.fromJson(
      _asMap(await _send(() => _httpClient.get(_uri('/api/v1/summary'), headers: _headers()))),
    );
  }

  Future<List<LiveStatusModel>> getLivePatients() async {
    final rows = _asList(await _send(() => _httpClient.get(_uri('/api/v1/monitor/patients/live'), headers: _headers())));
    return rows
        .whereType<Map<String, dynamic>>()
        .map(LiveStatusModel.fromJson)
        .toList();
  }

  /// Caretaker’s enrolled patients (one per account on current backend).
  Future<List<CaregiverAssignedPatientModel>> getCaregiverMyPatients() async {
    final m = _asMap(
      await _send(() => _httpClient.get(_uri('/api/v1/caregiver/my-patients'), headers: _headers())),
    );
    final list = m['patients'] as List<dynamic>? ?? const <dynamic>[];
    return list
        .whereType<Map<String, dynamic>>()
        .map(CaregiverAssignedPatientModel.fromJson)
        .toList();
  }

  Future<void> deleteCaregiverPatient(String patientId) async {
    await _send(
      () => _httpClient.delete(_uri('/api/v1/caregiver/my-patients/$patientId'), headers: _headers()),
    );
  }

  Future<List<AlertRecordModel>> getAlerts({
    String? status,
    String? patientId,
  }) async {
    final query = <String, String>{};
    if (status != null && status.isNotEmpty) {
      query['status'] = status;
    }
    if (patientId != null && patientId.isNotEmpty) {
      query['patient_id'] = patientId;
    }

    final uri = _uri('/api/v1/alerts').replace(queryParameters: query.isEmpty ? null : query);
    final rows = _asList(await _send(() => _httpClient.get(uri, headers: _headers())));
    return rows
        .whereType<Map<String, dynamic>>()
        .map(AlertRecordModel.fromJson)
        .toList();
  }

  Future<AlertRecordModel> acknowledgeAlert({
    required String alertId,
    String actor = 'caregiver_app',
    String? note,
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/alerts/$alertId/acknowledge'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'actor': actor,
            'note': note,
          }),
        ),
      )),
    );
  }

  Future<AlertRecordModel> resolveAlert({
    required String alertId,
    String actor = 'caregiver_app',
    String? note,
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/alerts/$alertId/resolve'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'actor': actor,
            'note': note,
          }),
        ),
      )),
    );
  }

  Future<void> updateDetectorSensitivity(String level) async {
    Map<String, dynamic> payload;
    switch (level) {
      case 'low':
        payload = {
          'medium_risk_score': 0.45,
          'high_risk_score': 0.68,
          'fall_score': 0.88,
        };
      case 'high':
        payload = {
          'medium_risk_score': 0.28,
          'high_risk_score': 0.50,
          'fall_score': 0.72,
        };
      default:
        payload = {
          'medium_risk_score': 0.35,
          'high_risk_score': 0.58,
          'fall_score': 0.80,
        };
    }

    await _send(
      () => _httpClient.put(
        _uri('/api/v1/detector/config'),
        headers: _headers(jsonBody: true),
        body: jsonEncode(payload),
      ),
    );
  }

  Future<CaregiverAuthModel> caregiverSignup({
    required String fullName,
    required String email,
    required String password,
  }) async {
    return CaregiverAuthModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/auth/caregiver/signup'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'full_name': fullName,
            'email': email,
            'password': password,
          }),
        ),
      )),
    );
  }

  Future<CaregiverAuthModel> caregiverLogin({
    required String email,
    required String password,
  }) async {
    return CaregiverAuthModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/auth/caregiver/login'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'email': email,
            'password': password,
          }),
        ),
      )),
    );
  }

  Future<Map<String, dynamic>> adminLogin({
    required String email,
    required String password,
  }) async {
    return _asMap(
      await _send(
        () => _httpClient.post(
          _uri('/api/v1/auth/admin/login'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({'email': email, 'password': password}),
        ),
      ),
    );
  }

  Future<Map<String, dynamic>> getAdminDashboard() async {
    return _asMap(
      await _send(() => _httpClient.get(_uri('/api/v1/admin/dashboard'), headers: _headers())),
    );
  }

  Future<List<Map<String, dynamic>>> adminListCaregivers() async {
    final rows = _asList(
      await _send(() => _httpClient.get(_uri('/api/v1/admin/caregivers'), headers: _headers())),
    );
    return rows.whereType<Map<String, dynamic>>().toList();
  }

  Future<Map<String, dynamic>> adminCreateCaregiver({
    required String fullName,
    required String email,
    required String password,
  }) async {
    return _asMap(
      await _send(
        () => _httpClient.post(
          _uri('/api/v1/admin/caregivers'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'full_name': fullName,
            'email': email,
            'password': password,
          }),
        ),
      ),
    );
  }

  Future<void> adminDeleteCaregiver(String userId) async {
    await _send(
      () => _httpClient.delete(_uri('/api/v1/admin/caregivers/$userId'), headers: _headers()),
    );
  }

  Future<List<Map<String, dynamic>>> adminListPatients() async {
    final rows = _asList(
      await _send(() => _httpClient.get(_uri('/api/v1/admin/patients'), headers: _headers())),
    );
    return rows.whereType<Map<String, dynamic>>().toList();
  }

  Future<Map<String, dynamic>> adminCreatePatient({
    required String fullName,
    int? age,
    String? caregiverId,
  }) async {
    return _asMap(
      await _send(
        () => _httpClient.post(
          _uri('/api/v1/admin/patients'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'full_name': fullName,
            if (age != null) 'age': age,
            if (caregiverId != null && caregiverId.trim().isNotEmpty) 'caregiver_id': caregiverId.trim(),
          }),
        ),
      ),
    );
  }

  Future<void> adminDeletePatient(String patientId) async {
    await _send(
      () => _httpClient.delete(_uri('/api/v1/admin/patients/$patientId'), headers: _headers()),
    );
  }

  Future<Map<String, dynamic>> elderLogin({
    required String username,
    required String password,
  }) async {
    return _asMap(
      await _send(
        () => _httpClient.post(
          _uri('/api/v1/auth/elder/login'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({'username': username, 'password': password}),
        ),
      ),
    );
  }

  Future<void> submitFallFeedback(Map<String, dynamic> payload) async {
    await _send(
      () => _httpClient.post(
        _uri('/api/v1/events/fall-feedback'),
        headers: _headers(jsonBody: true),
        body: jsonEncode(payload),
      ),
    );
  }

  Future<GeneratedPatientCredentialModel> generatePatientCredentials({
    required String caregiverToken,
    required String fullName,
    required int? age,
    required String homeAddress,
    String? emergencyContact,
    String? notes,
  }) async {
    return GeneratedPatientCredentialModel.fromJson(
      _asMap(      await _send(
        () => _httpClient.post(
          _uri('/api/v1/auth/caregiver/patient-credentials'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'caregiver_token': caregiverToken,
            'full_name': fullName,
            'age': age,
            'home_address': homeAddress,
            'emergency_contact': emergencyContact,
            'notes': notes,
          }),
        ),
      )),
    );
  }

  /// XGBoost pipeline: `enhancedFeatures` length = `enhanced_feature_dim` in `models/inference_manifest.json`.
  /// When fall is predicted, send either `fallTypeFeatures` (263-D) or raw `accWindow`/`gyroWindow`/`oriWindow`
  /// (300×3 each) so the server can build Colab fall-type features.
  Future<Map<String, dynamic>> inferMotion({
    required List<double> enhancedFeatures,
    List<double>? fallTypeFeatures,
    bool predictFallType = true,
    List<List<double>>? accWindow,
    List<List<double>>? gyroWindow,
    List<List<double>>? oriWindow,
  }) async {
    return _asMap(
      await _send(
        () => _httpClient.post(
          _uri('/api/v1/inference/motion'),
          headers: _headers(jsonBody: true),
          body: jsonEncode({
            'enhanced_features': enhancedFeatures,
            if (fallTypeFeatures != null) 'fall_type_features': fallTypeFeatures,
            'predict_fall_type': predictFallType,
            if (accWindow != null) 'acc_window': accWindow,
            if (gyroWindow != null) 'gyro_window': gyroWindow,
            if (oriWindow != null) 'ori_window': oriWindow,
          }),
        ),
      ),
    );
  }
}
