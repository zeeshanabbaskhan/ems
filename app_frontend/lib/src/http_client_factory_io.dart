import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:http/io_client.dart';

http.Client createBackendHttpClient() {
  final inner = HttpClient();
  inner.connectionTimeout = const Duration(seconds: 25);
  inner.idleTimeout = const Duration(seconds: 60);
  return IOClient(inner);
}
