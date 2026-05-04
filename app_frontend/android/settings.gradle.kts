pluginManagement {
    val flutterSdkPath: String =
        System.getenv("FLUTTER_ROOT")?.trim()?.takeIf { it.isNotEmpty() }
            ?: run {
                val properties = java.util.Properties()
                val local = file("local.properties")
                require(local.exists()) {
                    "Missing android/local.properties. Set FLUTTER_ROOT or add flutter.sdk to local.properties."
                }
                local.inputStream().use { properties.load(it) }
                val fromFile = properties.getProperty("flutter.sdk")
                require(!fromFile.isNullOrBlank()) {
                    "Set FLUTTER_ROOT or flutter.sdk in android/local.properties (SDK root, not ...\\bin)."
                }
                fromFile
            }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    id("com.android.application") version "8.12.1" apply false
    id("org.jetbrains.kotlin.android") version "2.2.0" apply false
}

include(":app")
