import 'package:flutter/material.dart';

import 'api_client.dart';
import 'monitoring_controller.dart' show MonitoringController;

/// Admin metrics + caretaker / patient CRUD (`/api/v1/admin/*`).
class AdminDashboardScreen extends StatefulWidget {
  const AdminDashboardScreen({super.key, required this.onBack});

  final VoidCallback onBack;

  @override
  State<AdminDashboardScreen> createState() => _AdminDashboardScreenState();
}

class _AdminDashboardScreenState extends State<AdminDashboardScreen> {
  final _emailCtrl = TextEditingController(text: 'admin@local');
  final _passCtrl = TextEditingController(text: 'admin123');
  final _urlCtrl = TextEditingController(text: MonitoringController.defaultBackendUrl);

  BackendApiClient? _client;

  Map<String, dynamic>? _dash;
  List<Map<String, dynamic>> _caregivers = <Map<String, dynamic>>[];
  List<Map<String, dynamic>> _patients = <Map<String, dynamic>>[];

  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _client?.close();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _loginAndLoad() async {
    setState(() {
      _busy = true;
      _error = null;
      _dash = null;
      _caregivers = <Map<String, dynamic>>[];
      _patients = <Map<String, dynamic>>[];
    });
    _client?.close();
    if (mounted) setState(() => _client = null);
    BackendApiClient? pending;
    try {
      pending = BackendApiClient(baseUrl: _urlCtrl.text.trim());
      final login =
          await pending.adminLogin(email: _emailCtrl.text.trim(), password: _passCtrl.text);
      final tok = login['access_token'] as String? ?? '';
      if (tok.isEmpty) throw Exception('No access_token from admin login');
      pending.setBearerToken(tok);
      if (!mounted) return;
      setState(() => _client = pending);
      pending = null;
      await _refreshAll();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      pending?.close();
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _refreshAll() async {
    final c = _client;
    if (c == null) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final dash = await c.getAdminDashboard();
      final cg = await c.adminListCaregivers();
      final pt = await c.adminListPatients();
      if (!mounted) return;
      setState(() {
        _dash = dash;
        _caregivers = cg;
        _patients = pt;
      });
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _deleteCaregiver(String id, String label) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove caretaker?'),
        content: Text(
          '$label\n\n'
          'All patients linked to this caretaker (and their elder logins) will be deleted.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: const Color(0xFFB53B34)),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    setState(() => _busy = true);
    try {
      await _client!.adminDeleteCaregiver(id);
      await _refreshAll();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _deletePatient(String id, String name) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remove patient?'),
        content: Text('$name — deletes devices/sessions/alerts and elder login if any.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: const Color(0xFFB53B34)),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok != true || !mounted) return;
    setState(() => _busy = true);
    try {
      await _client!.adminDeletePatient(id);
      await _refreshAll();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _showAddCaregiverDialog() async {
    final nameCtrl = TextEditingController();
    final emailCtrl = TextEditingController();
    final pwdCtrl = TextEditingController();
    final formKey = GlobalKey<FormState>();
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Add caretaker'),
        content: Form(
          key: formKey,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextFormField(
                  controller: nameCtrl,
                  decoration: const InputDecoration(labelText: 'Full name'),
                  validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null,
                ),
                TextFormField(
                  controller: emailCtrl,
                  decoration: const InputDecoration(labelText: 'Email'),
                  keyboardType: TextInputType.emailAddress,
                  validator: (v) => (v == null || !v.contains('@')) ? 'Valid email required' : null,
                ),
                TextFormField(
                  controller: pwdCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password'),
                  validator: (v) => (v == null || v.length < 6) ? 'Min 6 characters' : null,
                ),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(
            onPressed: () {
              if (formKey.currentState?.validate() ?? false) Navigator.pop(ctx, true);
            },
            child: const Text('Create'),
          ),
        ],
      ),
    );
    if (saved != true || !mounted) return;
    setState(() => _busy = true);
    try {
      await _client!.adminCreateCaregiver(
        fullName: nameCtrl.text.trim(),
        email: emailCtrl.text.trim(),
        password: pwdCtrl.text,
      );
      await _refreshAll();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
    nameCtrl.dispose();
    emailCtrl.dispose();
    pwdCtrl.dispose();
  }

  Future<void> _showAddPatientDialog() async {
    final nameCtrl = TextEditingController();
    final ageCtrl = TextEditingController();
    String? caregiverPick;
    final formKey = GlobalKey<FormState>();
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (context, setLocal) => AlertDialog(
          title: const Text('Add patient'),
          content: Form(
            key: formKey,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextFormField(
                    controller: nameCtrl,
                    decoration: const InputDecoration(labelText: 'Full name'),
                    validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null,
                  ),
                  TextFormField(
                    controller: ageCtrl,
                    decoration: const InputDecoration(labelText: 'Age (optional)'),
                    keyboardType: TextInputType.number,
                  ),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String?>(
                    value: caregiverPick,
                    decoration: const InputDecoration(labelText: 'Assign to caretaker (optional)'),
                    items: [
                      const DropdownMenuItem<String?>(value: null, child: Text('— None —')),
                      ..._caregivers.map(
                        (c) => DropdownMenuItem<String?>(
                          value: c['id'] as String?,
                          child: Text('${c['full_name']} (${c['email']})'),
                        ),
                      ),
                    ],
                    onChanged: (v) => setLocal(() => caregiverPick = v),
                  ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
            FilledButton(
              onPressed: () {
                if (formKey.currentState?.validate() ?? false) Navigator.pop(ctx, true);
              },
              child: const Text('Create'),
            ),
          ],
        ),
      ),
    );
    if (saved != true || !mounted) return;
    int? age;
    final ageText = ageCtrl.text.trim();
    if (ageText.isNotEmpty) age = int.tryParse(ageText);
    setState(() => _busy = true);
    try {
      await _client!.adminCreatePatient(
        fullName: nameCtrl.text.trim(),
        age: age,
        caregiverId: caregiverPick,
      );
      await _refreshAll();
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
    nameCtrl.dispose();
    ageCtrl.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: widget.onBack),
        title: const Text('Admin — SisFall'),
        actions: [
          if (_client != null)
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _busy ? null : _refreshAll,
              tooltip: 'Refresh',
            ),
        ],
      ),
      body: Column(
        children: [
          if (_busy && _dash != null) const LinearProgressIndicator(minHeight: 2),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
              TextField(
                controller: _urlCtrl,
                decoration: const InputDecoration(labelText: 'Backend base URL'),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _emailCtrl,
                decoration: const InputDecoration(labelText: 'Admin email'),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _passCtrl,
                obscureText: true,
                decoration: const InputDecoration(labelText: 'Password'),
              ),
              const SizedBox(height: 14),
              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Text(_error!, style: const TextStyle(color: Color(0xFFB53B34))),
                ),
              FilledButton(
                onPressed: _busy ? null : _loginAndLoad,
                child: Text(_busy && _dash == null ? 'Signing in…' : 'Sign in & load'),
              ),
              const SizedBox(height: 24),
              if (_dash != null) ...[
                const Text('Overview', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
                const SizedBox(height: 8),
                _tile('Caretakers', '${_dash!['caretakers']}'),
                _tile('Elders (accounts)', '${_dash!['elders_registered']}'),
                _tile('Patients', '${_dash!['patients']}'),
                _tile('Open alerts', '${_dash!['open_alerts']}'),
                _tile('Fall feedback rows', '${_dash!['fall_feedback_events']}'),
                _tile('Datasets', '${_dash!['datasets']}'),
                if (_dash!['note'] != null) Text('${_dash!['note']}'),
                const SizedBox(height: 28),
                Row(
                  children: [
                    const Expanded(
                      child: Text('Caretakers', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
                    ),
                    TextButton.icon(
                      onPressed: (_busy || _client == null) ? null : _showAddCaregiverDialog,
                      icon: const Icon(Icons.person_add_outlined),
                      label: const Text('Add'),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ..._caregivers.map((row) {
                  final id = row['id'] as String? ?? '';
                  final name = row['full_name'] as String? ?? '';
                  final email = row['email'] as String? ?? '';
                  return Card(
                    margin: const EdgeInsets.only(bottom: 8),
                    child: ListTile(
                      title: Text(name),
                      subtitle: Text(email),
                      trailing: IconButton(
                        icon: const Icon(Icons.delete_outline, color: Color(0xFFB53B34)),
                        onPressed: _busy ? null : () => _deleteCaregiver(id, name),
                      ),
                    ),
                  );
                }),
                const SizedBox(height: 20),
                Row(
                  children: [
                    const Expanded(
                      child: Text('Patients', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
                    ),
                    TextButton.icon(
                      onPressed: (_busy || _client == null) ? null : _showAddPatientDialog,
                      icon: const Icon(Icons.person_add_outlined),
                      label: const Text('Add'),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ..._patients.map((row) {
                  final id = row['id'] as String? ?? '';
                  final name = row['full_name'] as String? ?? '';
                  final cg = row['caregiver_id'] as String?;
                  final elder = row['elder_username'] as String?;
                  return Card(
                    margin: const EdgeInsets.only(bottom: 8),
                    child: ListTile(
                      title: Text(name),
                      subtitle: Text(
                        [
                          if (cg != null && cg.isNotEmpty) 'Caretaker: ${cg.substring(0, 8)}…',
                          if (elder != null && elder.isNotEmpty) 'Elder login: $elder',
                          if (cg == null || cg.isEmpty) 'No caretaker link',
                        ].join(' · '),
                      ),
                      isThreeLine: elder != null && elder.isNotEmpty,
                      trailing: IconButton(
                        icon: const Icon(Icons.delete_outline, color: Color(0xFFB53B34)),
                        onPressed: _busy ? null : () => _deletePatient(id, name),
                      ),
                    ),
                  );
                }),
              ],
            ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _tile(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            Expanded(child: Text(k)),
            Text(v, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
