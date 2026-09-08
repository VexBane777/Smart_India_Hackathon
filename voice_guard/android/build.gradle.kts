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

    // flutter_webrtc's own android/build.gradle hardcodes compileSdkVersion 31,
    // which is too old for transitive androidx deps it pulls in (e.g.
    // androidx.fragment:1.7.1 needs 34+). Force every Android subproject to a
    // modern compileSdk so `flutter_webrtc:checkDebugAarMetadata` stops failing.
    afterEvaluate {
        extensions.findByType(com.android.build.gradle.BaseExtension::class.java)?.apply {
            compileSdkVersion(36)
        }
    }
}


tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
