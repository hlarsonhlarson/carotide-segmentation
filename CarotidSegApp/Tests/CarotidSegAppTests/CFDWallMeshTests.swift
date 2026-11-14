import SceneKit
import Testing
@testable import CarotidSegApp

struct CFDWallMeshTests {
    // Two right triangles, each with area 0.5 mm².
    private let points = [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]]
    private let triangles = [[0, 1, 2], [1, 3, 2]]

    @Test func testRecoloringPreservesVerticesAndTopology() throws {
        let mesh = CFDWallMesh(pointsMM: points, triangles: triangles)
        let values = [0.0, 0.25, 0.75, 1.0]
        let original = try #require(mesh.geometry(values: values, field: .tawss, revision: 0, range: 0...1, threshold: nil))
        let same = mesh.geometry(values: values, field: .tawss, revision: 0, range: 0...1, threshold: nil)
        #expect(original === same)

        let filtered = try #require(mesh.geometry(values: values, field: .tawss, revision: 0, range: 0...1, threshold: 0.2...0.8))
        #expect(!(original === filtered))
        #expect(original.sources(for: .vertex)[0] === filtered.sources(for: .vertex)[0])
        #expect(original.elements[0] === filtered.elements[0])
        #expect(original.elements[0].primitiveCount == 2)
        #expect(original.sources(for: .color)[0].data != filtered.sources(for: .color)[0].data)

        // Thresholding changes only colors; an out-of-band vertex is gray.
        let color = filtered.sources(for: .color)[0].data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(Array(color[0..<4]) == [0.22, 0.24, 0.27, 1])
        let vertices = original.sources(for: .vertex)[0].data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        #expect(Array(vertices[6..<9]) == [0, 0, 1])
    }

    @Test func testNewFrameAndFieldInvalidateColorsAndRange() throws {
        let mesh = CFDWallMesh(pointsMM: points, triangles: triangles)
        let first = [0.0, 1, 2, 3]
        let next = [8.0, 6, 4, 2]
        #expect(mesh.valueRange(values: first, field: .instantWSS, revision: 1) == 0...3)
        let original = try #require(mesh.geometry(values: first, field: .instantWSS, revision: 1, range: 0...8, threshold: nil))
        #expect(mesh.valueRange(values: next, field: .instantWSS, revision: 2) == 2...8)
        let changed = try #require(mesh.geometry(values: next, field: .instantWSS, revision: 2, range: 0...8, threshold: nil))
        #expect(original.sources(for: .color)[0].data != changed.sources(for: .color)[0].data)
        #expect(mesh.valueRange(values: first, field: .osi, revision: 0) == 0...3)
        #expect(mesh.valueRange(values: [2, 2, 2, 2], field: .meanPressure, revision: 0) == 0...1)
        #expect(mesh.geometry(values: [1], field: .tawss, revision: 0, range: 0...1, threshold: nil) == nil)
    }

    @Test func testThresholdAreasRetainAllVerticesConvention() {
        let mesh = CFDWallMesh(pointsMM: points, triangles: triangles)
        let values = [0.0, 1, 1, 2]
        let first = mesh.thresholdArea(values: values, range: 0...1)
        #expect(abs(first.area - 0.5) < 1e-12)
        #expect(abs(first.fraction - 0.5) < 1e-12)
        #expect(abs(mesh.thresholdArea(values: values, range: 0...2).area - 1) < 1e-12)
        #expect(mesh.thresholdArea(values: values, range: 0.9...1.1).area == 0)
    }
}
