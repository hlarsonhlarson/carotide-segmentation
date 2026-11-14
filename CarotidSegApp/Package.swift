// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "CarotidSegApp",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "CarotidSegApp",
            path: "Sources/CarotidSegApp"
        ),
        .testTarget(name: "CarotidSegAppTests", dependencies: ["CarotidSegApp"])
    ]
)
