allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Some plugin dependencies (e.g. file_picker 8.x) hardcode compileSdk 34 in
// their own android/build.gradle, which is lower than what other plugin
// dependencies (e.g. flutter_plugin_android_lifecycle, via
// flutter.compileSdkVersion on this Flutter version) require at build time.
// Force every Android library subproject to compile against the same SDK
// level as the app module so AAR metadata checks don't fail on the mismatch.
subprojects {
    // Skip ":app" here: it's forced to evaluate eagerly above via
    // evaluationDependsOn(":app"), so by the time this block runs for it,
    // Project.afterEvaluate would throw ("already evaluated"). It doesn't
    // need this fix anyway (it's an application module, not a library, and
    // already sets compileSdk = 36 directly in its own build.gradle.kts).
    if (project.path != ":app") {
        afterEvaluate {
            extensions.findByType(com.android.build.gradle.LibraryExtension::class.java)?.apply {
                compileSdk = 36
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
