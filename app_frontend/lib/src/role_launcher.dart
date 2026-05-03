import 'package:flutter/material.dart';

import 'admin_dashboard_screen.dart';
import 'monitoring_controller.dart';

typedef PatientHomeBuilder = Widget Function(MonitoringController controller);

/// Caretaker vs Elder vs Admin entry — avoids importing `app.dart` (no circular refs).
class RoleLauncher extends StatefulWidget {
  const RoleLauncher({
    super.key,
    required this.controller,
    required this.caregiverShell,
    required this.patientHomeBuilder,
  });

  final MonitoringController controller;
  final Widget caregiverShell;
  final PatientHomeBuilder patientHomeBuilder;

  @override
  State<RoleLauncher> createState() => _RoleLauncherState();
}

enum _LaunchPick { choose, caregiver, elder, admin }

class _RoleLauncherState extends State<RoleLauncher> {
  _LaunchPick _pick = _LaunchPick.choose;

  @override
  Widget build(BuildContext context) {
    switch (_pick) {
      case _LaunchPick.caregiver:
        return widget.caregiverShell;
      case _LaunchPick.elder:
        return ElderGateScreen(
          controller: widget.controller,
          patientHomeBuilder: widget.patientHomeBuilder,
          onBack: () => setState(() => _pick = _LaunchPick.choose),
        );
      case _LaunchPick.admin:
        return AdminDashboardScreen(onBack: () => setState(() => _pick = _LaunchPick.choose));
      case _LaunchPick.choose:
        return Scaffold(
          appBar: AppBar(title: const Text('SisFall Monitor')),
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Who is using this phone?',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Caregiver: one patient at a time, dashboard, map, and alerts.\n'
                    'Patient: sign in with the username and password your caregiver gave you.',
                    style: TextStyle(color: Color(0xFF5D7385), height: 1.45),
                  ),
                  const SizedBox(height: 28),
                  FilledButton(
                    onPressed: () => setState(() => _pick = _LaunchPick.caregiver),
                    child: const Text('I am the caregiver'),
                  ),
                  const SizedBox(height: 12),
                  FilledButton(
                    style: FilledButton.styleFrom(backgroundColor: const Color(0xFF1B9B8B)),
                    onPressed: () => setState(() => _pick = _LaunchPick.elder),
                    child: const Text('I am the patient'),
                  ),
                  const SizedBox(height: 48),
                  TextButton(
                    onPressed: () => setState(() => _pick = _LaunchPick.admin),
                    child: const Text('System administrator'),
                  ),
                ],
              ),
            ),
          ),
        );
    }
  }
}

class ElderGateScreen extends StatefulWidget {
  const ElderGateScreen({
    super.key,
    required this.controller,
    required this.patientHomeBuilder,
    required this.onBack,
  });

  final MonitoringController controller;
  final PatientHomeBuilder patientHomeBuilder;
  final VoidCallback onBack;

  @override
  State<ElderGateScreen> createState() => _ElderGateScreenState();
}

class _ElderGateScreenState extends State<ElderGateScreen> {
  final _userCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _userCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final map = await widget.controller.apiClient.elderLogin(
        username: _userCtrl.text.trim(),
        password: _passCtrl.text,
      );
      final token = map['access_token'] as String? ?? '';
      final rawPid = map['patient_id'];
      final pid = rawPid is String
          ? rawPid.trim()
          : rawPid is num
              ? rawPid.toString()
              : '';
      final name = map['display_name'] as String? ?? '';
      if (token.isEmpty || pid.isEmpty) {
        throw Exception('Invalid elder login response (missing patient link).');
      }
      await widget.controller.applyElderSession(accessToken: token, patientId: pid, displayName: name);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute<void>(
          builder: (_) => widget.patientHomeBuilder(widget.controller),
        ),
      );
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: widget.onBack),
        title: const Text('Elder sign in'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          TextField(
            controller: _userCtrl,
            decoration: const InputDecoration(labelText: 'Username (from caretaker)'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _passCtrl,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'Temporary password'),
          ),
          const SizedBox(height: 16),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Text(_error!, style: const TextStyle(color: Color(0xFFB53B34))),
            ),
          FilledButton(
            onPressed: _busy ? null : _submit,
            child: Text(_busy ? 'Signing in…' : 'Continue'),
          ),
        ],
      ),
    );
  }
}
