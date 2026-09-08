plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.voiceguard.voice_guard"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        isCoreLibraryDesugaringEnabled = true
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.voiceguard.voice_guard"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // "standard" is the regular Play Store / sideloaded build — CAPTURE_AUDIO_OUTPUT
    // is not declared, so it behaves exactly as before.
    // "privileged" additionally declares CAPTURE_AUDIO_OUTPUT (see
    // src/privileged/AndroidManifest.xml). Declaring it in a normal install is a
    // no-op — Android only honors it for apps installed as privileged system
    // apps — so this flavor is only meaningful when installed via the
    // Magisk module in magisk-privileged-module/ (see that directory's README).
    // A privileged-flavor build must NEVER be distributed through the Play
    // Store or a normal sideload: it is not more capable there, and shipping a
    // build that merely *requests* a signature|privileged permission
    // pointlessly widens the app's declared permission surface for users who
    // get no benefit from it.
    flavorDimensions += "capture"
    productFlavors {
        create("standard") {
            dimension = "capture"
        }
        create("privileged") {
            dimension = "capture"
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.4")

    // flutter_webrtc declares this as `implementation` (not `api`), so its
    // org.webrtc.* classes aren't exposed on :app's compile classpath even
    // though the jar is already merged into the runtime/final APK via that
    // dependency. compileOnly avoids pulling in a second copy at package
    // time — RemoteAudioTap.kt only needs these types to compile.
    // Must match flutter_webrtc's own pinned version (see its
    // android/build.gradle) to guarantee binary compatibility at runtime.
    compileOnly("io.github.webrtc-sdk:android:125.6422.03")
}
