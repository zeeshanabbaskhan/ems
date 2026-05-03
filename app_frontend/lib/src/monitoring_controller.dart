import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_ringtone_player/flutter_ringtone_player.dart';
import 'package:geolocator/geolocator.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import 'api_client.dart';
import 'api_config.dart';
import 'models.dart';
import 'motion_inference_helper.dart';
import 'sensor_streaming_service.dart';

class MonitoringController extends ChangeNotifier {
  MonitoringController({
    BackendApiClient? apiClient,
    SensorStreamingService? sensorService,
  })  : _apiClient = apiClient ?? BackendApiClient(baseUrl: defaultBackendUrl),
        _sensorService = sensorService ??
            SensorStreamingService(
              targetSamplingRateHz: defaultSampleRateHz,
              windowSize: offlineWindowSizeSamples,
              stepSize: offlineWindowStepSamples,
            );

  /// Single source of truth: [AppApiConfig.backendBaseUrl] (edit there for deploy / local).
  static const String defaultBackendUrl = AppApiConfig.backendBaseUrl;

  /// Former shipped default; if still stored, migrate so installs pick the configured host.
  static const String _legacyDefaultBackendUrl = 'http://10.0.2.2:8000';

  static bool _shouldMigrateStoredBackendUrl(String url) {
    final t = url.trim();
    if (t == _legacyDefaultBackendUrl) return true;
    final lower = t.toLowerCase();
    return lower.startsWith('http://127.0.0.1') || lower.startsWith('http://localhost');
  }
  static const String defaultDeviceLabel = 'Caregiver Phone';
  static const String elderDeviceLabel = 'Patient phone';
  static const double defaultSampleRateHz = 50.0;
  static const int offlineWindowSizeSamples = 128;
  static const int offlineWindowStepSamples = 64;

  static const String _backendUrlKey = 'backend_url';
  static const String _patientNameKey = 'patient_name';
  static const String _patientAgeKey = 'patient_age';
  static const String _deviceLabelKey = 'device_label';
  static const String _patientIdKey = 'patient_id';
  static const String _deviceIdKey = 'device_id';
  static const String _medicalNotesKey = 'medical_notes';
  static const String _emergencyContactKey = 'emergency_contact';
  static const String _photoPathKey = 'photo_path';
  static const String _notificationsEnabledKey = 'notifications_enabled';
  static const String _alertSensitivityKey = 'alert_sensitivity';
  static const String _alertViaEmailKey = 'alert_via_email';
  static const String _alertViaAlarmKey = 'alert_via_alarm';
  static const String _caregiverEmailKey = 'caregiver_email';
  static const String _homeLatitudeKey = 'home_latitude';
  static const String _homeLongitudeKey = 'home_longitude';
  static const String _caregiverTokenKey = 'caregiver_token';
  static const String _caregiverNameKey = 'caregiver_name';
  static const String _caregiverEmailAuthKey = 'caregiver_email_auth';
  static const String _elderAccessTokenKey = 'elder_access_token';

  final BackendApiClient _apiClient;
  final SensorStreamingService _sensorService;

  WebSocketChannel? _caregiverAlertSocket;

  /// For elder login / admin tools that need direct HTTP access.
  BackendApiClient get apiClient => _apiClient;

  SharedPreferences? _preferences;

  bool _initialized = false;
  bool _isBusy = false;
  bool _backendReachable = false;
  bool _isStreaming = false;

  String _backendUrl = defaultBackendUrl;
  String _patientName = '';
  int? _patientAge;
  String _deviceLabel = defaultDeviceLabel;

  String? _patientId;
  String? _deviceId;
  String? _sessionId;

  String _statusMessage = 'Enter patient and backend details to begin.';
  String? _lastError;

  int _batchesSent = 0;
  int _lastBatchSize = 0;
  DateTime? _lastTransmissionAt;

  DetectionResultModel? _lastDetection;
  MotionInferenceResponseModel? _lastMotionInference;
  LiveStatusModel? _liveStatus;
  AlertRecordModel? _activeAlert;
  TelemetrySnapshotModel? _latestTelemetry;
  SensorAccessStatus? _sensorAccessStatus;
  String _medicalNotes = '';
  String _emergencyContact = '';
  String? _photoPath;
  SystemSummaryModel? _summary;
  List<LiveStatusModel> _livePatients = <LiveStatusModel>[];
  List<AlertRecordModel> _caregiverAlerts = <AlertRecordModel>[];
  Timer? _autoRefreshTimer;
  bool _notificationsEnabled = true;
  String _alertSensitivity = 'medium';
  bool _alertViaEmail = false;
  bool _alertViaAlarm = true;
  String _caregiverEmail = '';
  bool _alarmPlaying = false;
  bool _alarmSilencedByUser = false;
  StreamSubscription<Position>? _locationSubscription;
  Position? _currentPosition;
  double? _homeLatitude;
  double? _homeLongitude;
  bool _locationTrackingEnabled = false;
  String? _locationError;
  String? _caregiverToken;
  String _caregiverName = '';
  String _caregiverAuthEmail = '';
  String? _elderAccessToken;
  DateTime? _lastLocationUploadAt;
  final List<GeneratedPatientCredentialModel> _credentialHistory = <GeneratedPatientCredentialModel>[];
  List<CaregiverAssignedPatientModel> _assignedPatients = <CaregiverAssignedPatientModel>[];

  bool get initialized => _initialized;
  bool get isBusy => _isBusy;
  bool get backendReachable => _backendReachable;
  bool get isStreaming => _isStreaming;
  bool get hasSetup {
    if (_elderAccessToken != null && _elderAccessToken!.trim().isNotEmpty) {
      return _backendUrl.trim().isNotEmpty &&
          _patientId != null &&
          _patientId!.trim().isNotEmpty &&
          _patientName.trim().isNotEmpty;
    }
    return _backendUrl.trim().isNotEmpty &&
        _patientName.trim().isNotEmpty &&
        _deviceLabel.trim().isNotEmpty;
  }
  bool get isReady => hasSetup && _backendReachable;

  String get backendUrl => _backendUrl;
  String get patientName => _patientName;
  int? get patientAge => _patientAge;
  String get deviceLabel => _deviceLabel;
  String? get patientId => _patientId;
  String? get deviceId => _deviceId;
  String? get sessionId => _sessionId;
  String get statusMessage => _statusMessage;
  String? get lastError => _lastError;
  int get batchesSent => _batchesSent;
  int get lastBatchSize => _lastBatchSize;
  DateTime? get lastTransmissionAt => _lastTransmissionAt;
  DetectionResultModel? get lastDetection => _lastDetection;
  MotionInferenceResponseModel? get lastMotionInference => _lastMotionInference;
  LiveStatusModel? get liveStatus => _liveStatus;
  AlertRecordModel? get activeAlert => _activeAlert;
  TelemetrySnapshotModel? get latestTelemetry => _latestTelemetry;
  SensorAccessStatus? get sensorAccessStatus => _sensorAccessStatus;
  String get medicalNotes => _medicalNotes;
  String get emergencyContact => _emergencyContact;
  String? get photoPath => _photoPath;
  SystemSummaryModel? get summary => _summary;
  List<LiveStatusModel> get livePatients => List.unmodifiable(_livePatients);
  List<AlertRecordModel> get caregiverAlerts => List.unmodifiable(_caregiverAlerts);
  bool get notificationsEnabled => _notificationsEnabled;
  String get alertSensitivity => _alertSensitivity;
  bool get alertViaEmail => _alertViaEmail;
  bool get alertViaAlarm => _alertViaAlarm;
  String get caregiverEmail => _caregiverEmail;
  bool get isAlarmPlaying => _alarmPlaying;
  Position? get currentPosition => _currentPosition;
  bool get locationTrackingEnabled => _locationTrackingEnabled;
  String? get locationError => _locationError;
  double? get homeLatitude => _homeLatitude;
  double? get homeLongitude => _homeLongitude;
  bool get hasHomeLocation => _homeLatitude != null && _homeLongitude != null;

  bool get hasElderSession =>
      _elderAccessToken != null && _elderAccessToken!.trim().isNotEmpty;
  bool get isCaregiverAuthenticated => _caregiverToken != null && _caregiverToken!.isNotEmpty;
  String get caregiverName => _caregiverName;
  String get caregiverAuthEmail => _caregiverAuthEmail;
  List<GeneratedPatientCredentialModel> get credentialHistory => List.unmodifiable(_credentialHistory);
  List<CaregiverAssignedPatientModel> get assignedPatients => List.unmodifiable(_assignedPatients);

  /// Patients to show on the caregiver dashboard (server list, or session credential fallbacks).
  List<CaregiverAssignedPatientModel> get dashboardPatients {
    if (_assignedPatients.isNotEmpty) {
      return List.unmodifiable(_assignedPatients);
    }
    return _credentialHistory
        .map(
          (g) => CaregiverAssignedPatientModel(
            id: g.patientId,
            fullName: g.patientName,
            age: null,
          ),
        )
        .toList();
  }

  /// Caregivers may enroll multiple patients; backend does not cap assignments.
  bool get canEnrollAnotherPatient => true;

  GeneratedPatientCredentialModel? get lastGeneratedCredential =>
      _credentialHistory.isEmpty ? null : _credentialHistory.last;

  LiveStatusModel? liveStatusForPatient(String patientId) {
    for (final row in _livePatients) {
      if (row.patientId == patientId) {
        return row;
      }
    }
    return null;
  }

  Future<void> initialize() async {
    if (_initialized) {
      return;
    }

    _preferences = await SharedPreferences.getInstance();
    final storedUrl = _preferences?.getString(_backendUrlKey);
    _backendUrl = storedUrl == null || storedUrl.trim().isEmpty
        ? defaultBackendUrl
        : storedUrl.trim();
    if (_shouldMigrateStoredBackendUrl(_backendUrl)) {
      _backendUrl = defaultBackendUrl;
      await _preferences?.setString(_backendUrlKey, _backendUrl);
    }
    _patientName = _preferences?.getString(_patientNameKey) ?? '';
    _patientAge = _preferences?.getInt(_patientAgeKey);
    _deviceLabel = _preferences?.getString(_deviceLabelKey) ?? defaultDeviceLabel;
    _patientId = _preferences?.getString(_patientIdKey);
    _deviceId = _preferences?.getString(_deviceIdKey);
    _medicalNotes = _preferences?.getString(_medicalNotesKey) ?? '';
    _emergencyContact = _preferences?.getString(_emergencyContactKey) ?? '';
    _photoPath = _preferences?.getString(_photoPathKey);
    _notificationsEnabled = _preferences?.getBool(_notificationsEnabledKey) ?? true;
    _alertSensitivity = _preferences?.getString(_alertSensitivityKey) ?? 'medium';
    _alertViaEmail = _preferences?.getBool(_alertViaEmailKey) ?? false;
    _alertViaAlarm = _preferences?.getBool(_alertViaAlarmKey) ?? true;
    _caregiverEmail = _preferences?.getString(_caregiverEmailKey) ?? '';
    _homeLatitude = _preferences?.getDouble(_homeLatitudeKey);
    _homeLongitude = _preferences?.getDouble(_homeLongitudeKey);
    _caregiverToken = _preferences?.getString(_caregiverTokenKey);
    _caregiverName = _preferences?.getString(_caregiverNameKey) ?? '';
    _caregiverAuthEmail = _preferences?.getString(_caregiverEmailAuthKey) ?? '';
    _elderAccessToken = _preferences?.getString(_elderAccessTokenKey);
    _sessionId = null;

    _apiClient.updateBaseUrl(_backendUrl);
    if (_caregiverToken != null && _caregiverToken!.isNotEmpty) {
      _apiClient.setBearerToken(_caregiverToken);
    } else if (_elderAccessToken != null && _elderAccessToken!.isNotEmpty) {
      _apiClient.setBearerToken(_elderAccessToken);
    } else {
      _apiClient.setBearerToken(null);
    }

    _statusMessage = hasSetup
        ? 'Saved setup loaded. Check the backend and start monitoring.'
        : 'Enter patient and backend details to begin.';

    _initialized = true;
    _ensureAutoRefresh();
    unawaited(startLocationTracking());
    unawaited(refreshCaregiverData(silent: true));
    _connectCaregiverAlertSocketIfNeeded();
    notifyListeners();
  }

  Future<void> caregiverSignup({
    required String fullName,
    required String email,
    required String password,
  }) async {
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      final auth = await _apiClient.caregiverSignup(
        fullName: fullName.trim(),
        email: email.trim(),
        password: password,
      );
      _caregiverToken = auth.accessToken;
      _caregiverName = auth.caregiverName;
      _caregiverAuthEmail = auth.caregiverEmail;
      await _clearElderAuth();
      await _persistCaregiverAuth();
      _apiClient.setBearerToken(_caregiverToken);
      _statusMessage = 'Caregiver account ready.';
      unawaited(refreshCaregiverData(silent: true));
      _connectCaregiverAlertSocketIfNeeded();
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to create caregiver account.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> caregiverLogin({
    required String email,
    required String password,
  }) async {
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      final auth = await _apiClient.caregiverLogin(
        email: email.trim(),
        password: password,
      );
      _caregiverToken = auth.accessToken;
      _caregiverName = auth.caregiverName;
      _caregiverAuthEmail = auth.caregiverEmail;
      await _clearElderAuth();
      await _persistCaregiverAuth();
      _apiClient.setBearerToken(_caregiverToken);
      _statusMessage = 'Welcome back $_caregiverName.';
      unawaited(refreshCaregiverData(silent: true));
      _connectCaregiverAlertSocketIfNeeded();
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to sign in.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  /// After elder signs in with generated username/password from caregiver enrollment.
  Future<void> applyElderSession({
    required String accessToken,
    required String patientId,
    String? displayName,
  }) async {
    _caregiverToken = null;
    _disconnectCaregiverAlertSocket();
    _elderAccessToken = accessToken;
    _apiClient.setBearerToken(accessToken);
    final trimmedPid = patientId.trim();
    _patientId = trimmedPid.isEmpty ? null : trimmedPid;
    _sessionId = null;
    _deviceId = null;
    _lastMotionInference = null;
    _lastDetection = null;
    _activeAlert = null;
    if (displayName != null && displayName.isNotEmpty) {
      _patientName = displayName;
    }
    if (_deviceLabel.trim().isEmpty) {
      _deviceLabel = elderDeviceLabel;
    }
    await _persistIdentifiers();
    await _persistSetup();
    await _persistElderAuth();
    notifyListeners();
  }

  Future<void> deleteEnrolledPatient(String patientId) async {
    if (patientId.trim().isEmpty) {
      return;
    }
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      await _apiClient.deleteCaregiverPatient(patientId.trim());
      _credentialHistory.removeWhere((c) => c.patientId == patientId.trim());
      await refreshCaregiverData(silent: true);
      _statusMessage = 'Patient removed. You can enroll someone new.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to remove patient.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> caregiverLogout() async {
    _disconnectCaregiverAlertSocket();
    _caregiverToken = null;
    _caregiverName = '';
    _caregiverAuthEmail = '';
    _apiClient.setBearerToken(null);
    _credentialHistory.clear();
    _assignedPatients = <CaregiverAssignedPatientModel>[];
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.remove(_caregiverTokenKey);
    await preferences.remove(_caregiverNameKey);
    await preferences.remove(_caregiverEmailAuthKey);
    notifyListeners();
  }

  Future<void> generatePatientCredentials({
    required String fullName,
    required String ageText,
    required String homeAddress,
    required String emergencyContact,
    required String notes,
  }) async {
    final token = _caregiverToken;
    if (token == null || token.isEmpty) {
      _lastError = 'Caregiver sign-in required first.';
      notifyListeners();
      return;
    }

    int? age;
    final parsed = ageText.trim();
    if (parsed.isNotEmpty) {
      final maybeAge = int.tryParse(parsed);
      if (maybeAge == null || maybeAge < 0 || maybeAge > 130) {
        _lastError = 'Patient age must be between 0 and 130.';
        notifyListeners();
        return;
      }
      age = maybeAge;
    }

    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      final created = await _apiClient.generatePatientCredentials(
        caregiverToken: token,
        fullName: fullName.trim(),
        age: age,
        homeAddress: homeAddress.trim(),
        emergencyContact: emergencyContact.trim().isEmpty ? null : emergencyContact.trim(),
        notes: notes.trim().isEmpty ? null : notes.trim(),
      );
      _credentialHistory.add(created);
      while (_credentialHistory.length > 1) {
        _credentialHistory.removeAt(0);
      }

      if (_patientName.trim().isEmpty) {
        _patientName = created.patientName;
      }
      if (_emergencyContact.trim().isEmpty) {
        _emergencyContact = emergencyContact.trim();
      }
      _statusMessage = 'Patient credentials generated successfully.';
      await refreshCaregiverData(silent: true);
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to generate patient credentials.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> startLocationTracking() async {
    await _ensureInitialized();
    _locationError = null;

    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      _locationTrackingEnabled = false;
      _locationError = 'Location services are turned off.';
      notifyListeners();
      return;
    }

    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied || permission == LocationPermission.deniedForever) {
      _locationTrackingEnabled = false;
      _locationError = 'Location permission denied.';
      notifyListeners();
      return;
    }

    _locationTrackingEnabled = true;
    await _locationSubscription?.cancel();
    _locationSubscription = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 5,
      ),
    ).listen(
      (position) {
        _currentPosition = position;
        _locationError = null;
        unawaited(_throttledUploadPatientLocation(position));
        notifyListeners();
      },
      onError: (error) {
        _locationError = error.toString();
        notifyListeners();
      },
    );

    try {
      _currentPosition ??= await Geolocator.getCurrentPosition();
      if (_currentPosition != null) {
        unawaited(_throttledUploadPatientLocation(_currentPosition!));
      }
    } catch (error) {
      _locationError = error.toString();
    }
    notifyListeners();
  }

  Future<void> _throttledUploadPatientLocation(Position p) async {
    final token = _elderAccessToken;
    if (token == null || token.isEmpty) return;
    final now = DateTime.now();
    final last = _lastLocationUploadAt;
    if (last != null && now.difference(last) < const Duration(seconds: 25)) {
      return;
    }
    _lastLocationUploadAt = now;
    double? heading;
    final h = p.heading;
    if (h >= 0 && h <= 360) {
      heading = h;
    }
    try {
      await _apiClient.postPatientLocation(
        latitude: p.latitude,
        longitude: p.longitude,
        accuracyM: p.accuracy,
        headingDegrees: heading,
        bearerToken: token,
      );
    } catch (_) {}
  }

  /// Opens Google Maps with walking directions from current GPS to saved home (external app/browser).
  Future<void> openWalkingDirectionsHome() async {
    final cur = _currentPosition;
    final hLat = _homeLatitude;
    final hLon = _homeLongitude;
    if (cur == null || hLat == null || hLon == null) return;
    final uri = Uri.parse(
      'https://www.google.com/maps/dir/?api=1&origin=${cur.latitude},${cur.longitude}'
      '&destination=$hLat,$hLon&travelmode=walking',
    );
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> stopLocationTracking() async {
    await _locationSubscription?.cancel();
    _locationSubscription = null;
    _locationTrackingEnabled = false;
    notifyListeners();
  }

  Future<void> setHomeLocationFromCurrent() async {
    if (_currentPosition == null) {
      _locationError = 'Current location is not available yet.';
      notifyListeners();
      return;
    }
    _homeLatitude = _currentPosition!.latitude;
    _homeLongitude = _currentPosition!.longitude;
    await _persistHomeLocation();
    _statusMessage = 'Home location updated.';
    notifyListeners();
  }

  Future<void> clearHomeLocation() async {
    _homeLatitude = null;
    _homeLongitude = null;
    await _persistHomeLocation();
    notifyListeners();
  }

  Future<void> setNotificationsEnabled(bool enabled) async {
    _notificationsEnabled = enabled;
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.setBool(_notificationsEnabledKey, enabled);
    if (!enabled) {
      _stopAlarmIfPlaying();
    } else {
      _syncAlarmWithAlerts();
    }
    notifyListeners();
  }

  Future<void> setAlertSensitivity(String level) async {
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      await _apiClient.updateDetectorSensitivity(level);
      _alertSensitivity = level;
      final preferences = _preferences ?? await SharedPreferences.getInstance();
      _preferences = preferences;
      await preferences.setString(_alertSensitivityKey, level);
      _statusMessage = 'Alert sensitivity updated.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to update alert sensitivity.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> setAlertViaEmail(bool enabled) async {
    _alertViaEmail = enabled;
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.setBool(_alertViaEmailKey, enabled);
    notifyListeners();
  }

  Future<void> setAlertViaAlarm(bool enabled) async {
    _alertViaAlarm = enabled;
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.setBool(_alertViaAlarmKey, enabled);
    if (!enabled) {
      _stopAlarmIfPlaying();
    } else {
      _syncAlarmWithAlerts();
    }
    notifyListeners();
  }

  Future<void> setCaregiverEmail(String email) async {
    _caregiverEmail = email.trim();
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.setString(_caregiverEmailKey, _caregiverEmail);
    notifyListeners();
  }

  Future<void> updateProfile({
    required String patientName,
    required String patientAgeText,
    required String medicalNotes,
    required String emergencyContact,
  }) async {
    await saveSetup(
      backendUrl: _backendUrl,
      patientName: patientName,
      patientAgeText: patientAgeText,
      deviceLabel: _deviceLabel,
    );

    _medicalNotes = medicalNotes.trim();
    _emergencyContact = emergencyContact.trim();
    await _persistCareProfile();
    notifyListeners();
  }

  Future<SensorAccessStatus> refreshSensorStatus({bool silent = false}) async {
    await _ensureInitialized();

    if (!silent) {
      _isBusy = true;
      _lastError = null;
      _statusMessage = 'Checking phone sensors...';
      notifyListeners();
    }

    try {
      final status = await _sensorService.probeSensors();
      _sensorAccessStatus = status;
      if (!silent) {
        _statusMessage = status.allAvailable
            ? 'Phone sensors are available and ready.'
            : 'Some required sensors are unavailable on this device.';
      }
      return status;
    } catch (error) {
      final fallback = SensorAccessStatus(
        accelerometerAvailable: false,
        gyroscopeAvailable: false,
        fusedOrientationAvailable: false,
        checkedAt: DateTime.now(),
      );
      _sensorAccessStatus = fallback;
      _lastError = _formatError(error);
      if (!silent) {
        _statusMessage = 'Unable to verify phone sensors.';
      }
      return fallback;
    } finally {
      if (!silent) {
        _isBusy = false;
        notifyListeners();
      }
    }
  }

  Future<void> saveSetup({
    required String backendUrl,
    required String patientName,
    required String patientAgeText,
    required String deviceLabel,
  }) async {
    await _ensureInitialized();

    final normalizedBackendUrl =
        backendUrl.trim().isEmpty ? defaultBackendUrl : backendUrl.trim();
    final normalizedPatientName = patientName.trim();
    final normalizedDeviceLabel =
        deviceLabel.trim().isEmpty ? defaultDeviceLabel : deviceLabel.trim();

    int? parsedAge;
    final trimmedAge = patientAgeText.trim();
    if (trimmedAge.isNotEmpty) {
      parsedAge = int.tryParse(trimmedAge);
      if (parsedAge == null || parsedAge < 0 || parsedAge > 130) {
        _lastError = 'Patient age must be a whole number between 0 and 130.';
        notifyListeners();
        return;
      }
    }

    final patientChanged = _patientName != normalizedPatientName || _patientAge != parsedAge;
    final deviceChanged = _deviceLabel != normalizedDeviceLabel;

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Saving setup...';
    notifyListeners();

    try {
      _backendUrl = normalizedBackendUrl;
      _patientName = normalizedPatientName;
      _patientAge = parsedAge;
      _deviceLabel = normalizedDeviceLabel;

      if (patientChanged) {
        _patientId = null;
        _sessionId = null;
        _lastDetection = null;
        _liveStatus = null;
        _activeAlert = null;
        _latestTelemetry = null;
      }

      if (deviceChanged) {
        _deviceId = null;
        _sessionId = null;
      }

      _apiClient.updateBaseUrl(_backendUrl);

      await _persistSetup();
      await _persistIdentifiers();

      final reachable = await refreshBackendReachability(silent: true);
      _statusMessage = reachable
          ? 'Setup saved. Backend connection looks healthy.'
          : 'Setup saved. Backend could not be reached yet.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Setup was updated locally, but the connection check failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<bool> refreshBackendReachability({bool silent = false}) async {
    await _ensureInitialized();

    if (!silent) {
      _isBusy = true;
      _lastError = null;
      _statusMessage = 'Checking backend connection...';
      notifyListeners();
    }

    try {
      await _apiClient.ping();
      _backendReachable = true;
      _lastError = null;
      if (!silent) {
        _statusMessage = 'Backend connection is healthy.';
      }
      return true;
    } catch (error) {
      _backendReachable = false;
      _lastError = _formatError(error);
      if (!silent) {
        _statusMessage = 'Backend is not reachable right now.';
      }
      return false;
    } finally {
      if (!silent) {
        _isBusy = false;
        notifyListeners();
      }
    }
  }

  Future<void> startMonitoring() async {
    await _ensureInitialized();

    if (_isStreaming) {
      return;
    }

    if (!hasSetup) {
      _lastError = hasElderSession
          ? 'Patient sign-in is incomplete. Go back and sign in again.'
          : 'Complete the backend URL, patient name, and device label first.';
      _statusMessage = 'Setup is incomplete.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Starting live monitoring...';
    notifyListeners();

    try {
      _statusMessage = 'Checking backend…';
      notifyListeners();
      final reachable = await refreshBackendReachability(silent: true);
      if (!reachable) {
        final err = _lastError;
        final hint =
            err != null && err.trim().isNotEmpty ? err.trim() : 'no response';
        throw ApiException(
          'Backend ($hint at $_backendUrl). Open $_backendUrl/api/v1/health in the phone browser, '
          'or fix Wi‑Fi / VPN. Caregiver: confirm Backend URL in setup matches the deployed server.',
        );
      }

      _statusMessage = 'Checking motion sensors…';
      notifyListeners();
      final sensorStatus = await refreshSensorStatus(silent: true);
      if (!sensorStatus.allAvailable) {
        throw ApiException(
          'Phone sensors: accelerometer or gyroscope is not available. Grant motion permissions in '
          'Android Settings → Apps → this app → Permissions (Physical activity / Sensors).',
        );
      }

      _statusMessage = 'Verifying patient on server…';
      notifyListeners();
      try {
        await _ensurePatient();
      } catch (e) {
        throw ApiException('Patient record: ${_formatError(e)}');
      }

      _statusMessage = 'Registering this device…';
      notifyListeners();
      try {
        await _ensureDevice();
      } catch (e) {
        throw ApiException('Device registration: ${_formatError(e)}');
      }

      _statusMessage = 'Starting monitoring session…';
      notifyListeners();
      late final SessionRecord session;
      try {
        session = await _apiClient.startSession(
          patientId: _patientId!,
          deviceId: _deviceId!,
          sampleRateHz: defaultSampleRateHz,
        );
      } catch (e) {
        throw ApiException('Session start: ${_formatError(e)}');
      }

      _sessionId = session.id;
      _isStreaming = true;
      _batchesSent = 0;
      _lastBatchSize = 0;
      _lastTransmissionAt = null;
      _lastDetection = null;
      _lastMotionInference = null;
      _activeAlert = null;
      _latestTelemetry = null;
      _liveStatus = LiveStatusModel(
        patientId: _patientId!,
        patientName: _patientName,
        sessionId: _sessionId,
        deviceId: _deviceId,
        severity: 'low',
        score: 0.0,
        fallProbability: 0.0,
        lastMessage: 'Session started. Waiting for live motion data...',
      );

      await _persistIdentifiers();
      await _sensorService.start(_handleSensorBatch);
      await WakelockPlus.enable();
      _statusMessage = 'Monitoring is live. The phone is now streaming sensor batches.';
    } catch (error) {
      _isStreaming = false;
      _sessionId = null;
      await WakelockPlus.disable();
      _lastError = _formatError(error);
      _statusMessage = 'Unable to start monitoring.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> stopMonitoring() async {
    await _ensureInitialized();

    if (!_isStreaming && _sessionId == null) {
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Stopping monitoring...';
    notifyListeners();

    final currentSessionId = _sessionId;

    try {
      await _sensorService.stop();
      _isStreaming = false;

      if (currentSessionId != null) {
        await _apiClient.stopSession(currentSessionId);
      }

      _sessionId = null;
      await _persistIdentifiers();
      await WakelockPlus.disable();
      _statusMessage = 'Monitoring stopped.';
    } catch (error) {
      _isStreaming = false;
      _sessionId = null;
      await _persistIdentifiers();
      await WakelockPlus.disable();
      _lastError = _formatError(error);
      _statusMessage =
          'Streaming stopped on the phone, but the backend session may still be open.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  void _disconnectCaregiverAlertSocket() {
    try {
      _caregiverAlertSocket?.sink.close();
    } catch (_) {}
    _caregiverAlertSocket = null;
  }

  void _connectCaregiverAlertSocketIfNeeded() {
    if (!isCaregiverAuthenticated) {
      _disconnectCaregiverAlertSocket();
      return;
    }
    final tok = _caregiverToken;
    if (tok == null || tok.isEmpty) {
      return;
    }
    _disconnectCaregiverAlertSocket();
    try {
      final base = Uri.parse(_backendUrl);
      final scheme = base.scheme == 'https' ? 'wss' : 'ws';
      var p = base.path;
      if (p.endsWith('/')) {
        p = p.substring(0, p.length - 1);
      }
      final wsPath = '${p.isEmpty ? '' : p}/api/v1/ws/caregiver'.replaceAll('//', '/');
      final uri = Uri(
        scheme: scheme,
        host: base.host,
        port: base.hasPort ? base.port : null,
        path: wsPath.startsWith('/') ? wsPath : '/$wsPath',
        queryParameters: {'token': tok},
      );
      final ch = WebSocketChannel.connect(uri);
      _caregiverAlertSocket = ch;
      ch.stream.listen(
        (dynamic raw) {
          try {
            final m = jsonDecode(raw as String) as Map<String, dynamic>;
            if (m['type'] == 'alert') {
              unawaited(refreshCaregiverData(silent: true));
            }
          } catch (_) {}
        },
        onError: (_) {},
        onDone: () {},
      );
    } catch (_) {
      _caregiverAlertSocket = null;
    }
  }

  Future<void> triggerEmergencyAlert() async {
    await _ensureInitialized();

    if (_patientId == null || _patientId!.trim().isEmpty) {
      _lastError = 'Patient session is not ready. Sign in again.';
      _statusMessage = 'Emergency alert was not sent.';
      notifyListeners();
      return;
    }
    if (!hasElderSession && _patientName.trim().isEmpty) {
      _lastError = 'Add a patient name before sending an emergency alert.';
      _statusMessage = 'Emergency alert was not sent.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Triggering emergency alert...';
    notifyListeners();

    try {
      final reachable = await refreshBackendReachability(silent: true);
      if (!reachable) {
        throw ApiException(
          'The backend is not reachable. The manual alert could not be delivered.',
        );
      }

      await _ensurePatient();
      await _ensureDevice();

      final alert = await _apiClient.triggerManualAlert(
        patientId: _patientId!,
        deviceId: _deviceId,
        sessionId: _sessionId,
      );

      _activeAlert = alert;
      await refreshCaregiverData(silent: true);
      _statusMessage = 'Emergency alert sent successfully.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Emergency alert failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> refreshCaregiverData({bool silent = false}) async {
    await _ensureInitialized();
    if (!silent) {
      _isBusy = true;
      _lastError = null;
      notifyListeners();
    }

    try {
      if (isCaregiverAuthenticated) {
        final summaryFuture = _apiClient.getSummary();
        final liveFuture = _apiClient.getLivePatients();
        final alertsFuture = _apiClient.getAlerts();
        final responses = await Future.wait<Object>([
          summaryFuture,
          liveFuture,
          alertsFuture,
        ]);
        _summary = responses[0] as SystemSummaryModel;
        _livePatients = responses[1] as List<LiveStatusModel>;
        _caregiverAlerts = responses[2] as List<AlertRecordModel>;
        try {
          _assignedPatients = await _apiClient.getCaregiverMyPatients();
        } catch (_) {
          // Keep previous assignment list if this endpoint is unavailable.
        }
      } else {
        _summary = null;
        _caregiverAlerts = <AlertRecordModel>[];
        _assignedPatients = <CaregiverAssignedPatientModel>[];
        _livePatients = await _apiClient.getLivePatients();
      }

      if (_patientId != null) {
        final matched = _livePatients.where((item) => item.patientId == _patientId).toList();
        if (matched.isNotEmpty) {
          _liveStatus = matched.first;
        }
      }
      _syncAlarmWithAlerts();
      _backendReachable = true;
      _lastError = null;
    } catch (error) {
      _backendReachable = false;
      _lastError = _formatError(error);
      if (!silent) {
        _statusMessage = 'Unable to refresh monitoring information right now.';
      }
    } finally {
      if (!silent) {
        _isBusy = false;
      }
      notifyListeners();
    }
  }

  Future<void> acknowledgeAlert(String alertId) async {
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      await _apiClient.acknowledgeAlert(alertId: alertId);
      await refreshCaregiverData(silent: true);
      _statusMessage = 'Alert acknowledged.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to acknowledge alert.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> resolveAlert(String alertId) async {
    _isBusy = true;
    _lastError = null;
    notifyListeners();
    try {
      await _apiClient.resolveAlert(alertId: alertId);
      await refreshCaregiverData(silent: true);
      _statusMessage = 'Alert resolved.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Unable to resolve alert.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> _handleSensorBatch(List<SensorReadingPayload> samples) async {
    if (!_isStreaming ||
        _patientId == null ||
        _deviceId == null ||
        _sessionId == null ||
        samples.isEmpty) {
      return;
    }

    final currentSessionId = _sessionId;

    try {
      final response = await _apiClient.ingestLiveBatch(
        patientId: _patientId!,
        deviceId: _deviceId!,
        sessionId: currentSessionId!,
        samplingRateHz: defaultSampleRateHz,
        batteryLevel: null,
        samples: samples,
      );

      if (!_isStreaming || _sessionId != currentSessionId) {
        return;
      }

      _backendReachable = true;
      _lastError = null;
      _batchesSent += 1;
      _lastBatchSize = samples.length;
      _lastTransmissionAt = DateTime.now();
      _lastDetection = response.detection;
      _liveStatus = response.liveStatus;
      _latestTelemetry = response.telemetry;

      try {
        final motion = await MotionInferenceHelper.inferFromSamples(
          _apiClient,
          samples,
          predictFallType: true,
        );
        _lastMotionInference = motion;
        _statusMessage =
            '${response.detection.message} · ${motion.summaryLine}';
      } catch (_) {
        // Optional stack: `/api/v1/inference/motion` returns 503 when models are missing.
      }

      if (response.activeAlert != null) {
        _activeAlert = response.activeAlert;
      } else if (response.liveStatus.activeAlertIds.isEmpty) {
        _activeAlert = null;
      }

      _statusMessage = response.detection.message;
      unawaited(refreshCaregiverData(silent: true));
    } catch (error) {
      if (!_isStreaming || _sessionId != currentSessionId) {
        return;
      }

      _backendReachable = false;
      _lastError = 'Live upload (${samples.length} samples): ${_formatError(error)}';
      _lastBatchSize = samples.length;
      _statusMessage =
          'Streaming is still active on the phone, but the latest batch upload failed.';
    } finally {
      notifyListeners();
    }
  }

  Future<void> _ensurePatient() async {
    final existing = _patientId?.trim();
    if (existing != null && existing.isNotEmpty) {
      try {
        await _apiClient.getPatient(existing);
        return;
      } catch (_) {
        _patientId = null;
      }
    }

    if (hasElderSession) {
      throw ApiException(
        'Your elder account is not linked to a patient on the server. Sign out, then sign in again with the username and password from your caregiver.',
      );
    }

    final patient = await _apiClient.createPatient(
      fullName: _patientName,
      age: _patientAge,
    );

    _patientId = patient.id;
    await _persistIdentifiers();
  }

  Future<void> _ensureDevice() async {
    if (_deviceId != null) {
      try {
        await _apiClient.getDevice(_deviceId!);
        return;
      } catch (_) {
        _deviceId = null;
      }
    }

    final device = await _apiClient.createDevice(
      patientId: _patientId!,
      label: _deviceLabel,
      ownerName: _patientName,
    );

    _deviceId = device.id;
    await _persistIdentifiers();
  }

  Future<void> _persistSetup() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

    await preferences.setString(_backendUrlKey, _backendUrl);
    await preferences.setString(_patientNameKey, _patientName);
    await preferences.setString(_deviceLabelKey, _deviceLabel);

    if (_patientAge == null) {
      await preferences.remove(_patientAgeKey);
    } else {
      await preferences.setInt(_patientAgeKey, _patientAge!);
    }
  }

  Future<void> _persistIdentifiers() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

    if (_patientId == null) {
      await preferences.remove(_patientIdKey);
    } else {
      await preferences.setString(_patientIdKey, _patientId!);
    }

    if (_deviceId == null) {
      await preferences.remove(_deviceIdKey);
    } else {
      await preferences.setString(_deviceIdKey, _deviceId!);
    }
  }

  Future<void> _persistCareProfile() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.setString(_medicalNotesKey, _medicalNotes);
    await preferences.setString(_emergencyContactKey, _emergencyContact);
    if (_photoPath == null || _photoPath!.trim().isEmpty) {
      await preferences.remove(_photoPathKey);
    } else {
      await preferences.setString(_photoPathKey, _photoPath!);
    }
  }

  Future<void> _persistHomeLocation() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    if (_homeLatitude == null || _homeLongitude == null) {
      await preferences.remove(_homeLatitudeKey);
      await preferences.remove(_homeLongitudeKey);
    } else {
      await preferences.setDouble(_homeLatitudeKey, _homeLatitude!);
      await preferences.setDouble(_homeLongitudeKey, _homeLongitude!);
    }
  }

  Future<void> _persistCaregiverAuth() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    if (_caregiverToken == null || _caregiverToken!.isEmpty) {
      await preferences.remove(_caregiverTokenKey);
      await preferences.remove(_caregiverNameKey);
      await preferences.remove(_caregiverEmailAuthKey);
      return;
    }
    await preferences.setString(_caregiverTokenKey, _caregiverToken!);
    await preferences.setString(_caregiverNameKey, _caregiverName);
    await preferences.setString(_caregiverEmailAuthKey, _caregiverAuthEmail);
  }

  Future<void> _persistElderAuth() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    if (_elderAccessToken == null || _elderAccessToken!.isEmpty) {
      await preferences.remove(_elderAccessTokenKey);
      return;
    }
    await preferences.setString(_elderAccessTokenKey, _elderAccessToken!);
  }

  Future<void> _clearElderAuth() async {
    _elderAccessToken = null;
    _lastLocationUploadAt = null;
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;
    await preferences.remove(_elderAccessTokenKey);
  }

  /// Patient mode: stop streaming, clear elder JWT and local patient/device ids, then return to role picker via [runApp] from UI.
  Future<void> elderSignOut() async {
    await _ensureInitialized();
    if (_isStreaming) {
      await stopMonitoring();
    } else {
      await _sensorService.stop();
      await WakelockPlus.disable();
    }
    _locationSubscription?.cancel();
    _locationSubscription = null;
    _locationTrackingEnabled = false;
    await _clearElderAuth();
    _apiClient.setBearerToken(null);
    _patientId = null;
    _deviceId = null;
    _sessionId = null;
    _patientName = '';
    _patientAge = null;
    _lastMotionInference = null;
    _lastDetection = null;
    _liveStatus = null;
    _latestTelemetry = null;
    _lastError = null;
    await _persistIdentifiers();
    await _persistSetup();
    notifyListeners();
  }

  void _ensureAutoRefresh() {
    _autoRefreshTimer?.cancel();
    _autoRefreshTimer = Timer.periodic(const Duration(seconds: 15), (_) {
      if (_initialized) {
        unawaited(refreshCaregiverData(silent: true));
      }
    });
  }

  void _syncAlarmWithAlerts() {
    final hasSevereOpenAlert = _caregiverAlerts.any(
          (alert) =>
              alert.status != 'resolved' &&
              (alert.severity == 'high_risk' || alert.severity == 'fall_detected'));

    if (!hasSevereOpenAlert && _alarmSilencedByUser) {
      _alarmSilencedByUser = false;
    }

    final shouldPlay = _notificationsEnabled && _alertViaAlarm && hasSevereOpenAlert && !_alarmSilencedByUser;

    if (shouldPlay && !_alarmPlaying) {
      FlutterRingtonePlayer().playAlarm(
        looping: true,
        volume: 1.0,
        asAlarm: true,
      );
      _alarmPlaying = true;
      return;
    }

    if (!shouldPlay) {
      _stopAlarmIfPlaying();
    }
  }

  void _stopAlarmIfPlaying() {
    if (!_alarmPlaying) {
      return;
    }
    FlutterRingtonePlayer().stop();
    _alarmPlaying = false;
  }

  /// Loud alarm if elder does not dismiss fall dialog (escalation ladder).
  Future<void> playEscalationFallAlarm() async {
    if (!_alarmPlaying) {
      FlutterRingtonePlayer().playAlarm(looping: true, volume: 1.0, asAlarm: true);
      _alarmPlaying = true;
      notifyListeners();
    }
  }

  Future<void> triggerTestAlarm() async {
    if (!_notificationsEnabled || !_alertViaAlarm) {
      _lastError = 'Enable notifications and alarm sound first.';
      notifyListeners();
      return;
    }

    _alarmSilencedByUser = false;
    if (!_alarmPlaying) {
      FlutterRingtonePlayer().playAlarm(
        looping: true,
        volume: 1.0,
        asAlarm: true,
      );
      _alarmPlaying = true;
      _statusMessage = 'Test alarm is playing.';
      notifyListeners();
    }
  }

  Future<void> clearActiveAlarm() async {
    _alarmSilencedByUser = true;
    _stopAlarmIfPlaying();
    _statusMessage = 'Alarm cleared.';
    notifyListeners();
  }

  Future<void> _ensureInitialized() async {
    if (!_initialized) {
      await initialize();
    }
  }

  String _formatError(Object error) {
    if (error is ApiException) {
      return error.message;
    }

    final message = error.toString();
    if (message.startsWith('Exception: ')) {
      return message.substring('Exception: '.length);
    }
    return message;
  }

  @override
  void dispose() {
    _disconnectCaregiverAlertSocket();
    _autoRefreshTimer?.cancel();
    _stopAlarmIfPlaying();
    _locationSubscription?.cancel();
    _apiClient.close();
    unawaited(_sensorService.stop());
    unawaited(WakelockPlus.disable());
    super.dispose();
  }
}
