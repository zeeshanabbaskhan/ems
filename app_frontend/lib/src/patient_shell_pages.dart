import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart' as gmap;
import 'package:latlong2/latlong.dart';

import 'monitoring_controller.dart';

/// Patient tab: recent accelerometer / gyroscope samples (local batch or server echo).
class PatientLiveSensorPage extends StatelessWidget {
  const PatientLiveSensorPage({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final c = controller;
        final rows = c.displayedLiveSensorRows;
        final det = c.lastDetection;
        final motion = c.lastMotionInference;

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Live sensors',
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            Text(
              c.isStreaming
                  ? 'Showing the latest batch from your phone (updates while Active protection is on).'
                  : 'Turn on Active protection on the Home tab to stream data, or view the last batch the server echoed.',
              style: TextStyle(fontSize: 14, color: Colors.grey.shade800),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 10,
              runSpacing: 8,
              children: [
                _StatChip(
                  label: 'Streaming',
                  value: c.isStreaming ? 'On' : 'Off',
                ),
                _StatChip(label: 'Batches sent', value: '${c.batchesSent}'),
                _StatChip(
                  label: 'Last batch',
                  value: '${c.lastBatchSize} samples',
                ),
                _StatChip(
                  label: 'Last send',
                  value: c.lastTransmissionAt == null
                      ? '—'
                      : _shortTime(c.lastTransmissionAt!),
                ),
              ],
            ),
            if (det != null) ...[
              const SizedBox(height: 14),
              _card(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Latest detection',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 6),
                    Text(det.message, style: const TextStyle(fontSize: 14)),
                    Text(
                      'Fall probability ${(det.fallProbability * 100).toStringAsFixed(1)}% · score ${det.score.toStringAsFixed(2)}',
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey.shade700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
            if (motion != null) ...[
              const SizedBox(height: 10),
              _card(
                child: Text(
                  motion.summaryLine,
                  style: const TextStyle(fontSize: 14),
                ),
              ),
            ],
            const SizedBox(height: 16),
            Text(
              rows.isEmpty ? 'No samples yet' : 'Recent samples (newest first)',
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
            ),
            const SizedBox(height: 8),
            if (rows.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 24),
                child: Center(
                  child: Text(
                    'Start Active protection to see live accelerometer and gyroscope values.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey.shade600),
                  ),
                ),
              )
            else
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: DataTable(
                  columnSpacing: 18,
                  columns: const [
                    DataColumn(label: Text('t(ms)')),
                    DataColumn(label: Text('acc x')),
                    DataColumn(label: Text('acc y')),
                    DataColumn(label: Text('acc z')),
                    DataColumn(label: Text('gyro x')),
                    DataColumn(label: Text('gyro y')),
                    DataColumn(label: Text('gyro z')),
                  ],
                  rows: [
                    for (final r in rows.reversed.take(16))
                      DataRow(
                        cells: [
                          DataCell(Text('${r.timestampMs}')),
                          DataCell(Text(r.accX.toStringAsFixed(2))),
                          DataCell(Text(r.accY.toStringAsFixed(2))),
                          DataCell(Text(r.accZ.toStringAsFixed(2))),
                          DataCell(Text(r.gyroX.toStringAsFixed(3))),
                          DataCell(Text(r.gyroY.toStringAsFixed(3))),
                          DataCell(Text(r.gyroZ.toStringAsFixed(3))),
                        ],
                      ),
                  ],
                ),
              ),
            const SizedBox(height: 12),
            Text(
              'Units: acceleration m/s², gyroscope rad/s (phone axes).',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ],
        );
      },
    );
  }

  static String _shortTime(DateTime t) {
    final l = t.toLocal();
    return '${l.hour.toString().padLeft(2, '0')}:${l.minute.toString().padLeft(2, '0')}';
  }
}

class _StatChip extends StatelessWidget {
  const _StatChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text('$label: $value', style: const TextStyle(fontSize: 12)),
      visualDensity: VisualDensity.compact,
    );
  }
}

Widget _card({required Widget child}) {
  return Container(
    width: double.infinity,
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(14),
      boxShadow: const [
        BoxShadow(
          color: Color(0x140F2E4D),
          blurRadius: 10,
          offset: Offset(0, 4),
        ),
      ],
    ),
    child: child,
  );
}

/// Patient tab: set home on map, distance from current GPS to home.
class PatientSettingsMapPage extends StatefulWidget {
  const PatientSettingsMapPage({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<PatientSettingsMapPage> createState() => _PatientSettingsMapPageState();
}

class _PatientSettingsMapPageState extends State<PatientSettingsMapPage> {
  gmap.LatLng? _mapPick;

  double _distanceKm(double lat1, double lon1, double lat2, double lon2) {
    const distance = Distance();
    return distance.as(
      LengthUnit.Kilometer,
      LatLng(lat1, lon1),
      LatLng(lat2, lon2),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final c = widget.controller;
        final pos = c.currentPosition;
        final hasHome = c.hasHomeLocation;
        final home = hasHome
            ? gmap.LatLng(c.homeLatitude!, c.homeLongitude!)
            : null;
        final pick = _mapPick;

        final center = pos != null
            ? gmap.LatLng(pos.latitude, pos.longitude)
            : (pick ?? home ?? const gmap.LatLng(20, 0));

        double? distKm;
        if (pos != null && hasHome) {
          distKm = _distanceKm(
            pos.latitude,
            pos.longitude,
            c.homeLatitude!,
            c.homeLongitude!,
          );
        }

        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Settings',
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            const Text(
              'Tap the map to place an orange pin, then save it as your home. '
              'Turn on “Share live location” on the Home tab so distance updates from your current GPS.',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 12),
            if (distKm != null)
              _card(
                child: Row(
                  children: [
                    const Icon(Icons.social_distance, color: Color(0xFF155A9B)),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Live distance to home: ${distKm.toStringAsFixed(2)} km',
                        style: const TextStyle(
                          fontWeight: FontWeight.w700,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ],
                ),
              )
            else if (c.hasElderSession)
              _card(
                child: Text(
                  hasHome
                      ? (pos == null
                            ? 'Enable Share live location on the Home tab to see live distance to home.'
                            : 'Orange pin: tap map to choose. Green: saved home. Blue: you (when GPS is on).')
                      : 'Tap the map to set home, or use “Use my GPS as home”.',
                  style: TextStyle(fontSize: 14, color: Colors.grey.shade800),
                ),
              ),
            const SizedBox(height: 12),
            SizedBox(
              height: 320,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: gmap.GoogleMap(
                  key: ValueKey(
                    '${center.latitude}_${center.longitude}_${pick?.latitude}_${pick?.longitude}',
                  ),
                  initialCameraPosition: gmap.CameraPosition(
                    target: center,
                    zoom: pos != null ? 15 : (hasHome || pick != null ? 12 : 3),
                  ),
                  onTap: (latlng) => setState(() => _mapPick = latlng),
                  myLocationButtonEnabled: false,
                  mapToolbarEnabled: false,
                  markers: {
                    if (pos != null)
                      gmap.Marker(
                        markerId: const gmap.MarkerId('current'),
                        position: gmap.LatLng(pos.latitude, pos.longitude),
                        icon: gmap.BitmapDescriptor.defaultMarkerWithHue(
                          gmap.BitmapDescriptor.hueAzure,
                        ),
                        infoWindow: const gmap.InfoWindow(
                          title: 'Current location',
                        ),
                      ),
                    if (home != null)
                      gmap.Marker(
                        markerId: const gmap.MarkerId('home'),
                        position: home,
                        icon: gmap.BitmapDescriptor.defaultMarkerWithHue(
                          gmap.BitmapDescriptor.hueGreen,
                        ),
                        infoWindow: const gmap.InfoWindow(title: 'Saved home'),
                      ),
                    if (pick != null)
                      gmap.Marker(
                        markerId: const gmap.MarkerId('pick'),
                        position: pick,
                        icon: gmap.BitmapDescriptor.defaultMarkerWithHue(
                          gmap.BitmapDescriptor.hueOrange,
                        ),
                        infoWindow: const gmap.InfoWindow(
                          title: 'Selected home pin',
                        ),
                      ),
                  },
                  polylines: {
                    if (pos != null && home != null)
                      gmap.Polyline(
                        polylineId: const gmap.PolylineId('home_line'),
                        points: [
                          gmap.LatLng(pos.latitude, pos.longitude),
                          home,
                        ],
                        width: 4,
                        color: const Color(0xFF2A7DA8),
                      ),
                  },
                ),
              ),
            ),
            const SizedBox(height: 6),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton(
                  onPressed: (c.isBusy || pick == null)
                      ? null
                      : () async {
                          await c.setHomeLocationFromCoordinates(
                            pick.latitude,
                            pick.longitude,
                          );
                          if (mounted) setState(() => _mapPick = null);
                        },
                  child: const Text('Save orange pin as home'),
                ),
                OutlinedButton(
                  onPressed: c.isBusy
                      ? null
                      : () async {
                          await c.setHomeLocationFromCurrent();
                          if (mounted) setState(() => _mapPick = null);
                        },
                  child: const Text('Use my GPS as home'),
                ),
                TextButton(
                  onPressed: c.isBusy || !hasHome
                      ? null
                      : () async {
                          await c.clearHomeLocation();
                          if (mounted) setState(() => _mapPick = null);
                        },
                  child: const Text('Clear saved home'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            if (pos != null && hasHome)
              FilledButton.tonalIcon(
                onPressed: c.isBusy
                    ? null
                    : () => c.openWalkingDirectionsHome(),
                icon: const Icon(Icons.directions_walk),
                label: const Text('Walking directions in Google Maps'),
              ),
            if (c.locationError != null) ...[
              const SizedBox(height: 12),
              Material(
                color: const Color(0xFFFFEBEE),
                borderRadius: BorderRadius.circular(12),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    c.locationError!,
                    style: const TextStyle(color: Color(0xFFB53B34)),
                  ),
                ),
              ),
            ],
          ],
        );
      },
    );
  }
}
