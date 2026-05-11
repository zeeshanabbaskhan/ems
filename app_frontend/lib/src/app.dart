import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import 'models.dart';
import 'monitoring_controller.dart';
import 'patient_shell_pages.dart';
import 'role_launcher.dart';

class ElderlyMonitorApp extends StatefulWidget {
  const ElderlyMonitorApp({super.key, this.controller});

  final MonitoringController? controller;

  @override
  State<ElderlyMonitorApp> createState() => _ElderlyMonitorAppState();
}

class _ElderlyMonitorAppState extends State<ElderlyMonitorApp> {
  late final MonitoringController _controller;
  late final Future<void> _initializationFuture;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ?? MonitoringController();
    _initializationFuture = _controller.initialize();
  }

  @override
  void dispose() {
    if (_ownsController) {
      _controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'SisFall Care Monitor',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0A7FA6),
          brightness: Brightness.light,
        ),
      ),
      builder: (context, child) {
        final media = MediaQuery.of(context);
        return MediaQuery(
          data: media.copyWith(textScaler: const TextScaler.linear(0.92)),
          child: child ?? const SizedBox.shrink(),
        );
      },
      home: FutureBuilder<void>(
        future: _initializationFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Scaffold(
              body: Center(child: CircularProgressIndicator()),
            );
          }
          return RoleLauncher(
            controller: _controller,
            caregiverShell: MonitoringShell(controller: _controller),
            patientHomeBuilder: (c) => FallAwarePatientHome(controller: c),
          );
        },
      ),
    );
  }
}

/// Wraps [PatientModeHome] with a fall confirmation dialog.
class FallAwarePatientHome extends StatefulWidget {
  const FallAwarePatientHome({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<FallAwarePatientHome> createState() => _FallAwarePatientHomeState();
}

class _FallAwarePatientHomeState extends State<FallAwarePatientHome> {
  String? _lastShownKey;
  bool _fallDialogOpen = false;
  int _patientTab = 0;

  Future<void> _exitToRolePicker(BuildContext context) async {
    final c = widget.controller;
    await c.elderSignOut();
    if (!context.mounted) {
      return;
    }
    // Pop all routes back to root — RoleLauncher's listener will switch to
    // the 'choose' screen automatically via _onControllerChange.
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.controller,
      builder: (context, _) {
        final alert = widget.controller.activeAlert;
        if (alert != null) {
          final key = alert.id;
          if (key != _lastShownKey && !_fallDialogOpen) {
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (!mounted || _fallDialogOpen) {
                return;
              }
              setState(() {
                _lastShownKey = key;
                _fallDialogOpen = true;
              });
              _showFallDialog(context, widget.controller, alert, key);
            });
          }
        }

        return Scaffold(
          appBar: AppBar(
            title: Text(
              _patientTab == 0
                  ? 'Patient'
                  : _patientTab == 1
                  ? 'Live'
                  : 'Settings',
            ),
            actions: [
              TextButton(
                onPressed: widget.controller.isBusy
                    ? null
                    : () => _exitToRolePicker(context),
                child: const Text('Sign out'),
              ),
            ],
          ),
          body: SafeArea(
            child: IndexedStack(
              index: _patientTab,
              sizing: StackFit.expand,
              children: [
                PatientModeHome(controller: widget.controller),
                PatientLiveSensorPage(controller: widget.controller),
                PatientSettingsMapPage(controller: widget.controller),
              ],
            ),
          ),
          bottomNavigationBar: NavigationBar(
            selectedIndex: _patientTab,
            onDestinationSelected: (i) => setState(() => _patientTab = i),
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.shield_outlined),
                label: 'Home',
              ),
              NavigationDestination(icon: Icon(Icons.sensors), label: 'Live'),
              NavigationDestination(
                icon: Icon(Icons.map_outlined),
                label: 'Settings',
              ),
            ],
          ),
        );
      },
    );
  }

  void _showFallDialog(
    BuildContext context,
    MonitoringController controller,
    AlertRecordModel alert,
    String key,
  ) {
    String selectedResponse = 'need_help';
    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('Fall detected'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(alert.message),
                    const SizedBox(height: 10),
                    const Text(
                      'Your caregiver has already been alerted automatically.',
                    ),
                    const SizedBox(height: 12),
                    DropdownButtonFormField<String>(
                      initialValue: selectedResponse,
                      decoration: const InputDecoration(
                        labelText: 'Your feedback',
                        border: OutlineInputBorder(),
                      ),
                      items: const [
                        DropdownMenuItem(
                          value: 'need_help',
                          child: Text('Need help'),
                        ),
                        DropdownMenuItem(
                          value: 'okay',
                          child: Text('I am okay'),
                        ),
                      ],
                      onChanged: (value) {
                        if (value == null) {
                          return;
                        }
                        setDialogState(() {
                          selectedResponse = value;
                        });
                      },
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(),
                  child: const Text('Close'),
                ),
                FilledButton(
                  onPressed: () => _submit(
                    ctx,
                    controller,
                    alert,
                    selectedResponse,
                  ),
                  child: const Text('Submit'),
                ),
              ],
            );
          },
        );
      },
    ).then((_) {
      if (mounted) {
        setState(() => _fallDialogOpen = false);
      }
    });
  }

  Future<void> _submit(
    BuildContext ctx,
    MonitoringController controller,
    AlertRecordModel alert,
    String response,
  ) async {
    Navigator.of(ctx).pop();
    final pid = controller.patientId ?? '';
    try {
      await controller.apiClient.submitFallFeedback({
        'patient_id': pid,
        'response': response,
        'fall_detected': true,
        'predicted_fall_type_code': alert.severity,
        'fall_probability': alert.score,
      });
    } catch (_) {}
  }
}

class MonitoringShell extends StatefulWidget {
  const MonitoringShell({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<MonitoringShell> createState() => _MonitoringShellState();
}

class _MonitoringShellState extends State<MonitoringShell> {
  int _tabIndex = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.controller.refreshCaregiverData(silent: true);
    });
  }

  void _openAlertsSheet(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (ctx) {
        return DraggableScrollableSheet(
          expand: false,
          initialChildSize: 0.72,
          minChildSize: 0.38,
          maxChildSize: 0.96,
          builder: (_, scrollController) {
            return AlertsScreen(
              controller: widget.controller,
              scrollController: scrollController,
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final controller = widget.controller;
        if (!controller.isCaregiverAuthenticated) {
          return CaregiverAuthScreen(controller: controller);
        }

        return Scaffold(
          appBar: AppBar(
            title: const Text('Caregiver'),
            actions: [
              IconButton(
                tooltip: 'Alerts',
                onPressed: () => _openAlertsSheet(context),
                icon: Badge(
                  isLabelVisible: controller.caregiverAlerts.isNotEmpty,
                  label: Text('${controller.caregiverAlerts.length}'),
                  child: const Icon(Icons.notifications_outlined),
                ),
              ),
            ],
          ),
          body: Stack(
            children: [
              Column(
                children: [
                  if (controller.isAlarmPlaying)
                    Container(
                      width: double.infinity,
                      color: const Color(0xFFB53B34),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 10,
                      ),
                      child: SafeArea(
                        bottom: false,
                        child: Row(
                          children: [
                            const Icon(
                              Icons.notification_important,
                              color: Colors.white,
                            ),
                            const SizedBox(width: 8),
                            const Expanded(
                              child: Text(
                                'Emergency alarm is active',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                            TextButton(
                              onPressed: controller.clearActiveAlarm,
                              style: TextButton.styleFrom(
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Clear'),
                            ),
                          ],
                        ),
                      ),
                    ),
                  Expanded(child: _buildCaregiverTabBody(controller, context)),
                ],
              ),
              if (controller.isBusy)
                const Align(
                  alignment: Alignment.topCenter,
                  child: LinearProgressIndicator(minHeight: 3),
                ),
            ],
          ),
          bottomNavigationBar: NavigationBar(
            selectedIndex: _tabIndex,
            onDestinationSelected: (value) => setState(() => _tabIndex = value),
            destinations: const [
              NavigationDestination(
                icon: Icon(Icons.dashboard_outlined),
                selectedIcon: Icon(Icons.dashboard),
                label: 'Dashboard',
              ),
              NavigationDestination(
                icon: Icon(Icons.person_add_alt_1_outlined),
                label: 'Enrollment',
              ),
              NavigationDestination(
                icon: Icon(Icons.map_outlined),
                label: 'Map',
              ),
              NavigationDestination(
                icon: Icon(Icons.monitor_heart_outlined),
                label: 'Live',
              ),
              NavigationDestination(
                icon: Icon(Icons.insights_outlined),
                label: 'Insights',
              ),
              NavigationDestination(
                icon: Icon(Icons.settings_outlined),
                label: 'Settings',
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildCaregiverTabBody(
    MonitoringController controller,
    BuildContext context,
  ) {
    switch (_tabIndex) {
      case 0:
        return CaregiverDashboard(
          controller: controller,
          onShowAllAlerts: () => _openAlertsSheet(context),
        );
      case 1:
        return CaregiverEnrollmentScreen(controller: controller);
      case 2:
        return CaregiverMapTabScreen(controller: controller);
      case 3:
        return LiveMonitoringScreen(controller: controller);
      case 4:
        return InsightsScreen(controller: controller);
      case 5:
      default:
        return SettingsScreen(controller: controller);
    }
  }
}

class CaregiverAuthScreen extends StatefulWidget {
  const CaregiverAuthScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<CaregiverAuthScreen> createState() => _CaregiverAuthScreenState();
}

class _CaregiverAuthScreenState extends State<CaregiverAuthScreen> {
  bool _isSignup = true;
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    final email = _emailCtrl.text.trim();
    final password = _passwordCtrl.text;
    final fullName = _nameCtrl.text.trim();

    if (_isSignup && fullName.isEmpty) {
      _showValidation('Full name is required.');
      return;
    }
    if (email.isEmpty || !email.contains('@') || !email.contains('.')) {
      _showValidation('Please enter a valid email address.');
      return;
    }
    if (password.trim().length < 6) {
      _showValidation('Password must be at least 6 characters.');
      return;
    }

    if (_isSignup) {
      await widget.controller.caregiverSignup(
        fullName: fullName,
        email: email,
        password: password,
      );
    } else {
      await widget.controller.caregiverLogin(
        email: email,
        password: password,
      );
    }
  }

  void _showValidation(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    return Scaffold(
      appBar: AppBar(title: const Text('Caregiver Access')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _isSignup ? 'Create caregiver account' : 'Caregiver sign in',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 12),
                if (_isSignup) ...[
                  TextField(
                    controller: _nameCtrl,
                    decoration: const InputDecoration(labelText: 'Full Name'),
                  ),
                  const SizedBox(height: 10),
                ],
                TextField(
                  controller: _emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: 'Email'),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _passwordCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password'),
                ),
                const SizedBox(height: 14),
                FilledButton(
                  onPressed: controller.isBusy ? null : _submit,
                  child: Text(_isSignup ? 'Sign Up as Caregiver' : 'Sign In'),
                ),
                const SizedBox(height: 8),
                TextButton(
                  onPressed: () => setState(() => _isSignup = !_isSignup),
                  child: Text(
                    _isSignup
                        ? 'Already have an account? Sign in'
                        : 'Need an account? Sign up',
                  ),
                ),
              ],
            ),
          ),
          if (controller.lastError != null) ...[
            const SizedBox(height: 12),
            _StatusBanner(
              color: const Color(0xFFB53B34),
              title: _isSignup ? 'Sign-up Error' : 'Sign-in Error',
              message: controller.lastError!,
            ),
          ],
        ],
      ),
    );
  }
}

class CaregiverEnrollmentScreen extends StatefulWidget {
  const CaregiverEnrollmentScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<CaregiverEnrollmentScreen> createState() =>
      _CaregiverEnrollmentScreenState();
}

class _CaregiverEnrollmentScreenState extends State<CaregiverEnrollmentScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _homeCtrl = TextEditingController();
  final _contactCtrl = TextEditingController();
  final _notesCtrl = TextEditingController();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _ageCtrl.dispose();
    _homeCtrl.dispose();
    _contactCtrl.dispose();
    _notesCtrl.dispose();
    super.dispose();
  }

  Future<void> _generate() async {
    await widget.controller.generatePatientCredentials(
      fullName: _nameCtrl.text,
      ageText: _ageCtrl.text,
      homeAddress: _homeCtrl.text,
      emergencyContact: _contactCtrl.text,
      notes: _notesCtrl.text,
    );
    if (!mounted || widget.controller.lastError != null) {
      return;
    }
    _nameCtrl.clear();
    _ageCtrl.clear();
    _homeCtrl.clear();
    _contactCtrl.clear();
    _notesCtrl.clear();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final history = controller.credentialHistory;
    final assigned = controller.assignedPatients;
    final enrolled = assigned.isNotEmpty
        ? assigned.first
        : (history.isNotEmpty
              ? CaregiverAssignedPatientModel(
                  id: history.last.patientId,
                  fullName: history.last.patientName,
                  age: null,
                )
              : null);

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const _StatusBanner(
          color: Color(0xFF2A7DA8),
          title: 'Enrollment',
          message:
              'You can have one patient on this account. Share generated credentials only with their phone. '
              'To replace them, remove the current patient first.',
        ),
        if (enrolled != null) ...[
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  enrolled.fullName,
                  style: const TextStyle(
                    fontWeight: FontWeight.w800,
                    fontSize: 20,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  'Patient ID: ${enrolled.id}',
                  style: const TextStyle(
                    color: Color(0xFF5D7385),
                    fontSize: 13,
                  ),
                ),
                if (history.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  _StatCard(
                    label: 'Elder username',
                    value: history.last.username,
                  ),
                  const SizedBox(height: 6),
                  _StatCard(
                    label: 'Temporary password',
                    value: history.last.temporaryPassword,
                  ),
                ],
                const SizedBox(height: 16),
                OutlinedButton.icon(
                  onPressed: controller.isBusy
                      ? null
                      : () async {
                          final ok = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                              title: const Text('Remove patient?'),
                              content: const Text(
                                'This deletes their login, devices, and alerts on the server. You can enroll someone new afterward.',
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, false),
                                  child: const Text('Cancel'),
                                ),
                                FilledButton(
                                  onPressed: () => Navigator.pop(ctx, true),
                                  child: const Text('Remove'),
                                ),
                              ],
                            ),
                          );
                          if (ok == true && context.mounted) {
                            await controller.deleteEnrolledPatient(enrolled.id);
                          }
                        },
                  icon: const Icon(Icons.delete_outline),
                  label: const Text('Remove patient'),
                ),
              ],
            ),
          ),
        ],
        if (enrolled == null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _nameCtrl,
                  decoration: const InputDecoration(labelText: 'Patient name'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _ageCtrl,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(labelText: 'Age'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _homeCtrl,
                  decoration: const InputDecoration(labelText: 'Home address'),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _contactCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Emergency contact',
                  ),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _notesCtrl,
                  decoration: const InputDecoration(labelText: 'Notes'),
                ),
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed:
                      (controller.isBusy || !controller.canEnrollAnotherPatient)
                      ? null
                      : _generate,
                  icon: const Icon(Icons.vpn_key_outlined),
                  label: const Text('Generate credentials'),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }
}

class _GeneratedCredentialCard extends StatelessWidget {
  const _GeneratedCredentialCard({
    required this.credential,
    required this.index,
  });

  final GeneratedPatientCredentialModel credential;
  final int index;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Generated access · patient $index',
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
          ),
          const SizedBox(height: 8),
          _StatCard(label: 'Patient', value: credential.patientName),
          const SizedBox(height: 8),
          _StatCard(label: 'Username', value: credential.username),
          const SizedBox(height: 8),
          _StatCard(
            label: 'Temporary Password',
            value: credential.temporaryPassword,
          ),
          const SizedBox(height: 8),
          const Text(
            'Share these credentials securely with the patient device.',
            style: TextStyle(color: Color(0xFF5D7385)),
          ),
        ],
      ),
    );
  }
}

int _dashboardSeverityRank(String severity) {
  switch (severity) {
    case 'fall_detected':
      return 4;
    case 'high_risk':
      return 3;
    case 'medium':
      return 2;
    case 'low':
      return 1;
    default:
      return 0;
  }
}

LiveStatusModel? _worstLiveAmong(
  MonitoringController controller,
  List<CaregiverAssignedPatientModel> patients,
) {
  LiveStatusModel? worst;
  var bestRank = -1;
  for (final p in patients) {
    final live = controller.liveStatusForPatient(p.id);
    if (live == null) {
      continue;
    }
    final r = _dashboardSeverityRank(live.severity);
    if (r > bestRank) {
      bestRank = r;
      worst = live;
    }
  }
  return worst;
}

class CaregiverDashboard extends StatelessWidget {
  const CaregiverDashboard({
    super.key,
    required this.controller,
    required this.onShowAllAlerts,
  });

  final MonitoringController controller;
  final VoidCallback onShowAllAlerts;

  @override
  Widget build(BuildContext context) {
    final patients = controller.dashboardPatients;
    final worst = _worstLiveAmong(controller, patients);
    final fallback = controller.liveStatus;
    final overviewLive = worst ?? (patients.length <= 1 ? fallback : null);
    final latchedAny = controller.hasLatchedFall;
    final overviewSeverity = latchedAny
        ? 'fall_detected'
        : (overviewLive?.severity ?? 'low');
    final overviewText = latchedAny
        ? 'Fall detected. Alarm state is latched until caregiver clears alarm.'
        :
        _cleanDetectionMessage(overviewLive?.lastMessage) ??
        (patients.isEmpty
            ? 'Enroll one patient from the Enrollment tab. They sign in on their own device.'
            : 'When your patient signs in on their device, live status appears here.');

    final children = <Widget>[
      if (patients.isNotEmpty)
        const Padding(
          padding: EdgeInsets.only(bottom: 8),
          child: Text(
            'Your patient',
            style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
          ),
        ),
    ];

    if (patients.isEmpty) {
      final live = controller.liveStatus;
      final isLatched =
          controller.patientId != null &&
          controller.isPatientFallLatched(controller.patientId!);
      final severity = isLatched ? 'fall_detected' : controller.displaySeverity;
      final risk = (controller.displayRiskScore * 100).round();
      final statusText = isLatched
          ? 'Fall detected. Alarm state is latched until caregiver clears alarm.'
          : (_cleanDetectionMessage(live?.lastMessage) ?? 'No live data yet.');
      final movement = controller.displayFallProbability * 100;
      children.addAll([
        _PatientHeroCard(
          name: controller.patientName.isEmpty
              ? 'No patient linked'
              : controller.patientName,
          age: controller.patientAge == null
              ? 'Age not set'
              : '${controller.patientAge} years',
          severity: severity,
          lastUpdated: _formatDateTime(controller.lastTransmissionAt),
        ),
        const SizedBox(height: 12),
        _StatusBanner(
          color: _severityColor(severity),
          title: _severityLabel(severity),
          message: statusText,
        ),
        if (live != null) ...[
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Latest detection',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 6),
                Text(statusText, style: const TextStyle(fontSize: 14)),
                Text(
                  'Fall probability ${(controller.displayFallProbability * 100).toStringAsFixed(1)}% · score ${controller.displayRiskScore.toStringAsFixed(2)}',
                  style: const TextStyle(fontSize: 13, color: Color(0xFF5D7385)),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: _cardDecoration(),
            child: Text(
              _liveSummaryLine(live),
              style: const TextStyle(fontSize: 14),
            ),
          ),
        ],
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: _StatCard(label: 'Risk Score', value: '$risk%'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _StatCard(
                label: 'Movement Activity',
                value: '${movement.toStringAsFixed(0)}%',
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: _StatCard(
                label: 'Last Movement',
                value: _formatDateTime(controller.lastTransmissionAt),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(child: _MiniTrendCard(value: movement / 100)),
          ],
        ),
      ]);
    } else {
      for (final p in patients) {
        final live = controller.liveStatusForPatient(p.id);
        final isLatched = controller.isPatientFallLatched(p.id);
        final severity = isLatched ? 'fall_detected' : (live?.severity ?? 'low');
        final risk = ((live?.score ?? 0) * 100).round();
        final statusText = isLatched
            ? 'Fall detected. Alarm state is latched until caregiver clears alarm.'
            :
            _cleanDetectionMessage(live?.lastMessage) ??
            'No live data yet. Patient can sign in with generated credentials.';
        final movement = (live?.fallProbability ?? 0) * 100;
        final ageLabel = p.age == null ? 'Age not set' : '${p.age} years';
        final activityLabel =
            (live?.predictedActivityClass ?? '').trim().isEmpty
            ? 'Unavailable'
            : live!.predictedActivityClass!;
        final updatedLabel = live != null
            ? (live.locationUpdatedAt != null
                  ? _formatDateTime(live.locationUpdatedAt)
                  : 'live')
            : 'offline';

        children.addAll([
          _PatientHeroCard(
            name: p.fullName,
            age: ageLabel,
            severity: severity,
            lastUpdated: updatedLabel,
          ),
          const SizedBox(height: 8),
          _StatusBanner(
            color: _severityColor(severity),
            title: '${p.fullName} · ${_severityLabel(severity)}',
            message: statusText,
          ),
          if (live != null) ...[
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: _cardDecoration(),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Latest detection',
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 6),
                  Text(statusText, style: const TextStyle(fontSize: 14)),
                  Text(
                    'Fall probability ${(live.fallProbability * 100).toStringAsFixed(1)}% · score ${live.score.toStringAsFixed(2)}',
                    style: const TextStyle(fontSize: 13, color: Color(0xFF5D7385)),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: _cardDecoration(),
              child: Text(
                _liveSummaryLine(live),
                style: const TextStyle(fontSize: 14),
              ),
            ),
          ],
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _StatCard(label: 'Risk Score', value: '$risk%'),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _StatCard(
                  label: 'Movement',
                  value: '${movement.toStringAsFixed(0)}%',
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _StatCard(
                  label: 'Stability',
                  value: _stabilityText((live?.score ?? 0).clamp(0.0, 1.0)),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(child: _MiniTrendCard(value: movement / 100)),
            ],
          ),
          const SizedBox(height: 10),
          _StatCard(label: 'Current Activity', value: activityLabel),
          const SizedBox(height: 16),
        ]);
      }
    }

    final alerts = controller.caregiverAlerts;
    if (alerts.isNotEmpty) {
      children.addAll([
        const SizedBox(height: 8),
        Row(
          children: [
            const Text(
              'Open alerts',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
            ),
            const Spacer(),
            TextButton(
              onPressed: onShowAllAlerts,
              child: Text('View all (${alerts.length})'),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ...alerts
            .take(3)
            .map(
              (a) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _AlertCard(
                  alert: a,
                  onAcknowledge: () => controller.acknowledgeAlert(a.id),
                  onResolve: () => controller.resolveAlert(a.id),
                ),
              ),
            ),
      ]);
    }

    children.addAll([
      const SizedBox(height: 8),
      Row(
        children: [
          Expanded(
            child: FilledButton.icon(
              onPressed: () => controller.refreshCaregiverData(),
              icon: const Icon(Icons.refresh),
              label: const Text('Refresh'),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFFC2453F),
              ),
              onPressed: onShowAllAlerts,
              icon: const Icon(Icons.notifications_active_outlined),
              label: const Text('Alerts'),
            ),
          ),
        ],
      ),
    ]);

    return RefreshIndicator(
      onRefresh: () => controller.refreshCaregiverData(),
      child: ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        children: children,
      ),
    );
  }
}

class LiveMonitoringScreen extends StatelessWidget {
  const LiveMonitoringScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final live = controller.liveStatus;
    final severity = controller.displaySeverity;
    final riskValue = controller.displayRiskScore.clamp(0.0, 1.0);
    final fallValue = controller.displayFallProbability.clamp(0.0, 1.0);
    final activityLabel =
        (controller.smoothedActivityLabel ?? '').trim().isEmpty
        ? 'Unavailable'
        : controller.smoothedActivityLabel!;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _StatusBanner(
          color: _severityColor(severity),
          title: 'Live Status: ${_severityLabel(severity)}',
          message: _cleanDetectionMessage(live?.lastMessage) ?? 'Monitoring active.',
        ),
        if (live != null || controller.lastDetection != null) ...[
          const SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Latest detection',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 6),
                Text(
                  _cleanDetectionMessage(
                        controller.lastDetection?.message ?? live?.lastMessage,
                      ) ??
                      live?.lastMessage ??
                      'Monitoring active.',
                  style: const TextStyle(fontSize: 14),
                ),
                Text(
                  'Fall probability ${(controller.displayFallProbability * 100).toStringAsFixed(1)}% · score ${controller.displayRiskScore.toStringAsFixed(2)}',
                  style: const TextStyle(fontSize: 13, color: Color(0xFF5D7385)),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          if (live != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: _cardDecoration(),
              child: Text(
                _liveSummaryLine(live),
                style: const TextStyle(fontSize: 14),
              ),
            ),
        ],
        const SizedBox(height: 16),
        const _WavePulseCard(),
        const SizedBox(height: 16),
        _CircularRiskMeter(label: 'Risk Level', value: riskValue),
        const SizedBox(height: 16),
        _StatCard(
          label: 'Movement Intensity',
          value: '${(fallValue * 100).toStringAsFixed(0)}%',
        ),
        const SizedBox(height: 10),
        _StatCard(label: 'Stability', value: _stabilityText(riskValue)),
        const SizedBox(height: 10),
        _StatCard(label: 'Alert Level', value: _severityLabel(severity)),
        const SizedBox(height: 10),
        _StatCard(label: 'Current Activity', value: activityLabel),
        const SizedBox(height: 16),
        const Text(
          'Recent timeline',
          style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
        ),
        const SizedBox(height: 8),
        _TimelineItem(
          label: _formatDateTime(
            DateTime.now().subtract(const Duration(minutes: 1)),
          ),
          text: 'Monitoring active',
        ),
        _TimelineItem(
          label: _formatDateTime(controller.lastTransmissionAt),
          text: 'Latest movement analyzed',
        ),
        _TimelineItem(
          label: _formatDateTime(DateTime.now()),
          text: _cleanDetectionMessage(live?.lastMessage) ?? 'No abnormal movement',
        ),
      ],
    );
  }
}

/// Full-screen map tab (Google Maps). Patient location comes from backend live feed.
class CaregiverMapTabScreen extends StatelessWidget {
  const CaregiverMapTabScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final liveRows = controller.livePatients;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text(
          'Live map',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 8),
        const Text(
          'Shows the last GPS point the patient device shared with the server. The elder must sign in on their phone and enable location.',
          style: TextStyle(color: Color(0xFF5D7385), fontSize: 14),
        ),
        const SizedBox(height: 14),
        CaregiverPatientsLocationMap(
          livePatients: controller.livePatients,
          mapHeight: 420,
        ),
        const SizedBox(height: 12),
        _CaregiverLiveRiskPanel(livePatients: liveRows),
      ],
    );
  }
}

class _CaregiverLiveRiskPanel extends StatelessWidget {
  const _CaregiverLiveRiskPanel({required this.livePatients});

  final List<LiveStatusModel> livePatients;

  @override
  Widget build(BuildContext context) {
    if (livePatients.isEmpty) {
      return const _StatusBanner(
        color: Color(0xFF4A708F),
        title: 'Live risk analysis',
        message:
            'No patient telemetry yet. Ask the patient to sign in and start Active protection.',
      );
    }

    final sorted = [...livePatients]
      ..sort((a, b) => b.score.compareTo(a.score));

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Live risk analysis',
            style: TextStyle(fontWeight: FontWeight.w800, fontSize: 16),
          ),
          const SizedBox(height: 8),
          ...sorted.map((row) {
            final riskPercent = (row.score * 100)
                .clamp(0, 100)
                .toStringAsFixed(0);
            final fallPercent = (row.fallProbability * 100)
                .clamp(0, 100)
                .toStringAsFixed(0);
            return Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      row.patientName,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  Text(
                    'Risk $riskPercent% · Fall $fallPercent% · ${_severityLabel(row.severity)}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: Color(0xFF4A5E70),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({
    super.key,
    required this.controller,
    this.scrollController,
  });

  final MonitoringController controller;
  final ScrollController? scrollController;

  @override
  Widget build(BuildContext context) {
    final alerts = controller.caregiverAlerts;
    return RefreshIndicator(
      onRefresh: () => controller.refreshCaregiverData(),
      child: ListView(
        controller: scrollController,
        padding: const EdgeInsets.all(16),
        children: [
          if (alerts.isEmpty)
            const _StatusBanner(
              color: Color(0xFF1B9B8B),
              title: 'No Active Alerts',
              message: 'Everything looks stable right now.',
            ),
          ...alerts.map(
            (alert) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: _AlertCard(
                alert: alert,
                onAcknowledge: () => controller.acknowledgeAlert(alert.id),
                onResolve: () => controller.resolveAlert(alert.id),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class InsightsScreen extends StatelessWidget {
  const InsightsScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final risk = (controller.displayRiskScore * 100).round();
    final movement = (controller.displayFallProbability * 100).round();
    final summary = controller.summary;
    final insight = movement < 25
        ? 'Patient has been less active than usual today.'
        : movement > 70
        ? 'Higher instability trends detected in recent hours.'
        : 'Movement pattern appears balanced today.';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text(
          'Daily & Weekly Insights',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _StatCard(label: 'Activity Level', value: '$movement%'),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _StatCard(label: 'Risk Pattern', value: '$risk%'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _SimpleBarGraph(
          values: [
            (movement / 100).clamp(0.1, 1.0),
            ((movement + 10) / 100).clamp(0.1, 1.0),
            ((movement - 5) / 100).clamp(0.1, 1.0),
            ((movement + 8) / 100).clamp(0.1, 1.0),
            (risk / 100).clamp(0.1, 1.0),
            ((risk - 7) / 100).clamp(0.1, 1.0),
            ((risk + 4) / 100).clamp(0.1, 1.0),
          ],
        ),
        const SizedBox(height: 12),
        _StatusBanner(
          color: const Color(0xFF2A7DA8),
          title: 'AI Insight',
          message: insight,
        ),
        if (summary != null) ...[
          const SizedBox(height: 12),
          _StatusBanner(
            color: const Color(0xFF4A708F),
            title: 'Monitoring Summary',
            message:
                '${summary.activeSessions} active monitoring sessions and ${summary.openAlerts} open alerts across ${summary.totalPatients} patients.',
          ),
        ] else ...[
          const SizedBox(height: 12),
          const _StatusBanner(
            color: Color(0xFF6B7C8D),
            title: 'Backend summary',
            message:
                'Sign in as caregiver and ensure the server is reachable to load /api/v1/summary.',
          ),
        ],
      ],
    );
  }
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late double _sensitivityValue;

  @override
  void initState() {
    super.initState();
    _sensitivityValue = _levelToSlider(widget.controller.alertSensitivity);
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final backendValue = _levelToSlider(controller.alertSensitivity);
    if (!controller.isBusy && backendValue != _sensitivityValue) {
      _sensitivityValue = backendValue;
    }

    String sliderHint(double value) {
      switch (value.round()) {
        case 0:
          return 'Lower sensitivity: fewer alerts, but subtle falls can be missed.';
        case 2:
          return 'Higher sensitivity: catches subtle falls, may raise extra alerts.';
        case 1:
        default:
          return 'Balanced sensitivity for everyday use.';
      }
    }

    String sliderLabel(double value) {
      switch (value.round()) {
        case 0:
          return 'Low';
        case 2:
          return 'High';
        case 1:
        default:
          return 'Medium';
      }
    }

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Text(
          'Settings',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 16),
        Container(
          decoration: _cardDecoration(),
          child: SwitchListTile(
            title: const Text('Alarm for severe alerts'),
            subtitle: const Text(
              'Play an in-app sound when a serious alert needs attention',
            ),
            value: controller.alertViaAlarm,
            onChanged: (v) => controller.setAlertViaAlarm(v),
          ),
        ),
        const SizedBox(height: 14),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: _cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Fall detection sensitivity',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
              ),
              const SizedBox(height: 6),
              Text(
                sliderHint(_sensitivityValue),
                style: const TextStyle(color: Color(0xFF5D7385)),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  const Text('Low'),
                  Expanded(
                    child: Slider(
                      value: _sensitivityValue,
                      min: 0,
                      max: 2,
                      divisions: 2,
                      label: sliderLabel(_sensitivityValue),
                      onChanged: controller.isBusy
                          ? null
                          : (value) {
                              setState(() => _sensitivityValue = value);
                            },
                      onChangeEnd: controller.isBusy
                          ? null
                          : (value) {
                              controller.setAlertSensitivity(
                                _sliderToLevel(value),
                              );
                            },
                    ),
                  ),
                  const Text('High'),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                'Current: ${sliderLabel(_sensitivityValue)}',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        SizedBox(
          width: double.infinity,
          child: FilledButton.icon(
            onPressed: controller.caregiverLogout,
            icon: const Icon(Icons.logout),
            label: const Text('Log out'),
          ),
        ),
      ],
    );
  }
}

double _levelToSlider(String level) {
  switch (level.trim().toLowerCase()) {
    case 'low':
      return 0;
    case 'high':
      return 2;
    case 'medium':
    default:
      return 1;
  }
}

String _sliderToLevel(double value) {
  switch (value.round()) {
    case 0:
      return 'low';
    case 2:
      return 'high';
    case 1:
    default:
      return 'medium';
  }
}

class PatientModeHome extends StatefulWidget {
  const PatientModeHome({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<PatientModeHome> createState() => _PatientModeHomeState();
}

class _PatientModeHomeState extends State<PatientModeHome> {
  Future<void> _confirmEmergency(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Send emergency alert?'),
          content: const Text(
            'Your caregiver is notified in real time through the app when this is sent.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Send Alert'),
            ),
          ],
        );
      },
    );
    if (confirmed == true && mounted) {
      await widget.controller.triggerEmergencyAlert();
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final c = widget.controller;
        final severity = c.liveStatus?.severity ?? 'low';

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: _severityColor(severity).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Your status',
                    style: TextStyle(fontWeight: FontWeight.w800, fontSize: 26),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    c.isStreaming
                        ? 'Protection is on — motion data is sent to your care team.'
                        : 'Turn on protection to stream motion and detect falls.',
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            SwitchListTile(
              title: const Text('Active protection'),
              subtitle: const Text(
                'Uses accelerometer & gyroscope. The phone may stay awake while this is on. '
                'For best results keep the app open; true background streaming depends on your device.',
              ),
              value: c.isStreaming,
              onChanged: c.isBusy
                  ? null
                  : (on) async {
                      if (on) {
                        await c.startMonitoring();
                      } else {
                        await c.stopMonitoring();
                      }
                    },
            ),
            if (c.hasElderSession) ...[
              SwitchListTile(
                title: const Text('Share live location'),
                subtitle: const Text(
                  'Shows you on the caregiver map and improves directions home.',
                ),
                value: c.locationTrackingEnabled,
                onChanged: c.isBusy
                    ? null
                    : (on) async {
                        if (on) {
                          await c.startLocationTracking();
                        } else {
                          await c.stopLocationTracking();
                        }
                      },
              ),
            ],
            if (c.locationError != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _StatusBanner(
                  color: const Color(0xFFB53B34),
                  title: 'Location',
                  message: c.locationError!,
                ),
              ),
            if (c.hasElderSession) ...[
              const SizedBox(height: 8),
              const _StatusBanner(
                color: Color(0xFF2A7DA8),
                title: 'Map & live distance',
                message:
                    'Open the Settings tab to set your home on the map and see live distance from your GPS. '
                    'Open Live to view accelerometer and gyroscope samples.',
              ),
            ],
            const SizedBox(height: 18),
            FilledButton.icon(
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFFC2453F),
                padding: const EdgeInsets.symmetric(vertical: 18),
              ),
              onPressed: c.isBusy ? null : () => _confirmEmergency(context),
              icon: const Icon(Icons.sos_outlined),
              label: const Text('Alert caregiver now'),
            ),
            if (c.lastError != null && c.lastError!.trim().isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                c.lastError!,
                style: const TextStyle(color: Color(0xFFB53B34), fontSize: 13),
              ),
            ],
          ],
        );
      },
    );
  }
}

class _PatientHeroCard extends StatelessWidget {
  const _PatientHeroCard({
    required this.name,
    required this.age,
    required this.severity,
    required this.lastUpdated,
  });

  final String name;
  final String age;
  final String severity;
  final String lastUpdated;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0A7FA6), Color(0xFF155A9B)],
        ),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            name,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(age, style: const TextStyle(color: Color(0xFFE3F4FB))),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatusPill(
                label: _severityEmoji(severity),
                color: _severityColor(severity),
              ),
              const SizedBox(width: 10),
              Text(
                'Updated $lastUpdated',
                style: const TextStyle(color: Colors.white70),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({
    required this.color,
    required this.title,
    required this.message,
  });

  final Color color;
  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(color: color, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(message),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: TextStyle(color: color, fontWeight: FontWeight.w700),
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(color: Color(0xFF5D7385))),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

class _MiniTrendCard extends StatelessWidget {
  const _MiniTrendCard({required this.value});
  final double value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('24h Activity'),
          const SizedBox(height: 8),
          LinearProgressIndicator(value: value, minHeight: 10),
        ],
      ),
    );
  }
}

class _WavePulseCard extends StatelessWidget {
  const _WavePulseCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Motion Activity',
            style: TextStyle(fontWeight: FontWeight.w700),
          ),
          SizedBox(height: 12),
          Row(
            children: [
              Expanded(child: _PulseBar(height: 14)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 20)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 30)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 18)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 12)),
            ],
          ),
        ],
      ),
    );
  }
}

class _PulseBar extends StatelessWidget {
  const _PulseBar({required this.height});
  final double height;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: const Color(0xFF1788B3).withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(8),
      ),
    );
  }
}

class _CircularRiskMeter extends StatelessWidget {
  const _CircularRiskMeter({required this.label, required this.value});

  final String label;
  final double value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Row(
        children: [
          SizedBox(
            width: 76,
            height: 76,
            child: CircularProgressIndicator(
              value: value,
              strokeWidth: 9,
              backgroundColor: Colors.grey.shade200,
            ),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 4),
              Text(
                '${(value * 100).toStringAsFixed(0)}%',
                style: const TextStyle(fontSize: 22),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _TimelineItem extends StatelessWidget {
  const _TimelineItem({required this.label, required this.text});
  final String label;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: _cardDecoration(),
      child: Row(
        children: [
          Text(label, style: const TextStyle(color: Color(0xFF5D7385))),
          const SizedBox(width: 10),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}

class _AlertCard extends StatelessWidget {
  const _AlertCard({
    required this.alert,
    required this.onAcknowledge,
    required this.onResolve,
  });

  final AlertRecordModel alert;
  final VoidCallback onAcknowledge;
  final VoidCallback onResolve;

  @override
  Widget build(BuildContext context) {
    final color = _severityColor(alert.severity);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _StatusPill(label: _severityLabel(alert.severity), color: color),
              const Spacer(),
              Text(
                _formatDateTime(alert.createdAt),
                style: const TextStyle(color: Color(0xFF6A7B8A)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            alert.message,
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 6),
          Text(
            alert.manuallyTriggered
                ? 'Manual alert'
                : 'Automatic detection alert',
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              OutlinedButton(
                onPressed: onAcknowledge,
                child: const Text('Acknowledge'),
              ),
              OutlinedButton(
                onPressed: onResolve,
                child: const Text('Resolve'),
              ),
              OutlinedButton(
                onPressed: () {},
                child: const Text('Call Patient'),
              ),
              FilledButton(
                onPressed: () {},
                child: const Text('Notify Contact'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class CaregiverPatientsLocationMap extends StatelessWidget {
  const CaregiverPatientsLocationMap({
    super.key,
    required this.livePatients,
    this.mapHeight = 280,
  });

  final List<LiveStatusModel> livePatients;
  final double mapHeight;

  @override
  Widget build(BuildContext context) {
    final withLoc = livePatients.where((p) => p.hasLiveLocation).toList();
    if (withLoc.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(14),
        decoration: _cardDecoration(),
        child: const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'No positions yet',
              style: TextStyle(fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 8),
            Text(
              'When an elder signs in on their phone, enables GPS, and walks outside, their last location appears here.',
              style: TextStyle(color: Color(0xFF5D7385)),
            ),
          ],
        ),
      );
    }

    final points = withLoc.map((p) => LatLng(p.latitude!, p.longitude!)).toList();
    double sumLat = 0, sumLon = 0;
    for (final q in points) {
      sumLat += q.latitude;
      sumLon += q.longitude;
    }
    final n = points.length;
    final center = LatLng(sumLat / n, sumLon / n);

    return Container(
      height: mapHeight,
      clipBehavior: Clip.antiAlias,
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: FlutterMap(
              options: MapOptions(
                initialCenter: center,
                initialZoom: n == 1 ? 15 : 11,
                interactionOptions: const InteractionOptions(
                  flags: InteractiveFlag.all & ~InteractiveFlag.rotate,
                ),
              ),
              children: [
                TileLayer(
                  urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'com.example.newapp',
                ),
                MarkerLayer(
                  markers: [
                    for (final p in withLoc)
                      Marker(
                        point: LatLng(p.latitude!, p.longitude!),
                        width: 40,
                        height: 40,
                        child: Icon(
                          Icons.location_on,
                          color: p.headingDegrees != null
                              ? const Color(0xFF2A7DA8)
                              : const Color(0xFFB53B34),
                          size: 34,
                        ),
                      ),
                  ],
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 10),
            child: Wrap(
              spacing: 12,
              runSpacing: 6,
              children: [
                for (final p in withLoc)
                  Text(
                    '${p.patientName}: ${p.latitude!.toStringAsFixed(4)}, ${p.longitude!.toStringAsFixed(4)}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: Color(0xFF4A5E70),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LocationMapCard extends StatelessWidget {
  const _LocationMapCard({
    super.key,
    required this.current,
    required this.home,
    required this.compact,
    this.headingDegrees,
  });

  final LatLng current;
  final LatLng? home;
  final bool compact;
  final double? headingDegrees;

  @override
  Widget build(BuildContext context) {
    final points = <LatLng>[if (home != null) home!, current];
    return Container(
      height: compact ? 200 : 280,
      clipBehavior: Clip.antiAlias,
      decoration: _cardDecoration(),
      child: FlutterMap(
        options: MapOptions(
          initialCenter: current,
          initialZoom: 15,
          interactionOptions: const InteractionOptions(
            flags: InteractiveFlag.all & ~InteractiveFlag.rotate,
          ),
        ),
        children: [
          TileLayer(
            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            userAgentPackageName: 'com.example.newapp',
          ),
          if (points.length == 2)
            PolylineLayer(
              polylines: [
                Polyline(
                  points: points,
                  strokeWidth: 4,
                  color: const Color(0xFF2A7DA8),
                ),
              ],
            ),
          MarkerLayer(
            markers: [
              Marker(
                point: current,
                width: 40,
                height: 40,
                child: Icon(
                  Icons.my_location,
                  color: headingDegrees != null
                      ? const Color(0xFF2A7DA8)
                      : const Color(0xFFB53B34),
                ),
              ),
              if (home != null)
                Marker(
                  point: home!,
                  width: 40,
                  height: 40,
                  child: const Icon(
                    Icons.home,
                    color: Color(0xFF1B9B8B),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SimpleBarGraph extends StatelessWidget {
  const _SimpleBarGraph({required this.values});
  final List<double> values;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: values
            .map(
              (value) => Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: Container(
                    height: 90 * value.clamp(0.1, 1.0),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2B88B3),
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                ),
              ),
            )
            .toList(),
      ),
    );
  }
}

BoxDecoration _cardDecoration() {
  return BoxDecoration(
    color: Colors.white,
    borderRadius: BorderRadius.circular(16),
    boxShadow: const [
      BoxShadow(color: Color(0x140F2E4D), blurRadius: 14, offset: Offset(0, 6)),
    ],
  );
}

Color _severityColor(String severity) {
  switch (severity) {
    case 'medium':
      return const Color(0xFFF0A542);
    case 'high_risk':
      return const Color(0xFFDE6B48);
    case 'fall_detected':
      return const Color(0xFFB53B34);
    default:
      return const Color(0xFF1B9B8B);
  }
}

String _severityLabel(String severity) {
  switch (severity) {
    case 'high_risk':
      return 'High Risk';
    case 'fall_detected':
      return 'Fall Detected';
    case 'medium':
      return 'Medium Risk';
    default:
      return 'Safe';
  }
}

String _severityEmoji(String severity) {
  switch (severity) {
    case 'high_risk':
      return '🔴 High Risk';
    case 'fall_detected':
      return '⚫ Fall Detected';
    case 'medium':
      return '🟡 Medium Risk';
    default:
      return '🟢 Safe';
  }
}

String _formatDateTime(DateTime? value) {
  if (value == null) return 'just now';
  final local = value.toLocal();
  final h = local.hour.toString().padLeft(2, '0');
  final m = local.minute.toString().padLeft(2, '0');
  return '$h:$m';
}

String _stabilityText(double riskValue) {
  if (riskValue > 0.75) return 'Unstable';
  if (riskValue > 0.45) return 'Observe closely';
  return 'Stable';
}

String? _cleanDetectionMessage(String? message) {
  if (message == null) return null;
  final raw = message.trim();
  if (raw.isEmpty) return null;
  final idx = raw.toLowerCase().indexOf('activity hint:');
  if (idx <= 0) {
    return raw;
  }
  return raw.substring(0, idx).trim();
}

String _liveSummaryLine(LiveStatusModel live) {
  final p = (live.fallProbability * 100).toStringAsFixed(1);
  if (live.severity == 'fall_detected') {
    return 'Fall detected ($p%)';
  }
  final act = (live.predictedActivityClass ?? '').trim();
  final label = act.isEmpty ? 'Activity' : act;
  return 'No fall ($p%): $label';
}

double _distanceKm(double lat1, double lon1, double lat2, double lon2) {
  const distance = Distance();
  return distance.as(
    LengthUnit.Kilometer,
    LatLng(lat1, lon1),
    LatLng(lat2, lon2),
  );
}

String _headingToCompassRose(double headingDegrees) {
  final n = (headingDegrees % 360 + 360) % 360;
  if (n >= 337.5 || n < 22.5) return 'North';
  if (n < 67.5) return 'North-East';
  if (n < 112.5) return 'East';
  if (n < 157.5) return 'South-East';
  if (n < 202.5) return 'South';
  if (n < 247.5) return 'South-West';
  if (n < 292.5) return 'West';
  return 'North-West';
}

String _bearingDirection(double lat1, double lon1, double lat2, double lon2) {
  const distance = Distance();
  final bearing = distance.bearing(LatLng(lat1, lon1), LatLng(lat2, lon2));
  final normalized = (bearing + 360) % 360;
  if (normalized >= 337.5 || normalized < 22.5) return 'North';
  if (normalized < 67.5) return 'North-East';
  if (normalized < 112.5) return 'East';
  if (normalized < 157.5) return 'South-East';
  if (normalized < 202.5) return 'South';
  if (normalized < 247.5) return 'South-West';
  if (normalized < 292.5) return 'West';
  return 'North-West';
}
