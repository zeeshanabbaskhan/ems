import 'package:http/http.dart' as http;

import 'http_client_factory_io.dart' if (dart.library.html) 'http_client_factory_web.dart' as impl;

http.Client createBackendHttpClient() => impl.createBackendHttpClient();
