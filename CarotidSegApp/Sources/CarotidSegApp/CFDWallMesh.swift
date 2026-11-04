import SceneKit

/// A results sheet owns one fixed wall mesh. Only its scalar colors change.
final class CFDWallMesh {
    private let vertexSource: SCNGeometrySource
    private let element: SCNGeometryElement
    private let material: SCNMaterial
    private let vertexCount: Int
    private let triangles: [[Int]]
    private let triangleAreas: [Double]
    private let totalArea: Double

    private struct FieldKey: Equatable {
        let field: CFDField
        let revision: Int
    }

    private struct ColorKey: Equatable {
        let field: FieldKey
        let range: ClosedRange<Double>
        let threshold: ClosedRange<Double>?
    }

    private var rangeKey: FieldKey?
    private var cachedRange: ClosedRange<Double> = 0...1
    private var colorKey: ColorKey?
    private var cachedGeometry: SCNGeometry?

    init(pointsMM: [[Double]], triangles: [[Int]]) {
        vertexCount = pointsMM.count
        vertexSource = SCNGeometrySource(vertices: pointsMM.map {
            SCNVector3(Float($0[0]), Float($0[2]), Float($0[1]))
        })
        self.triangles = triangles
        var indices: [Int32] = []
        indices.reserveCapacity(triangles.count * 3)
        var areas: [Double] = []
        areas.reserveCapacity(triangles.count)
        for tri in triangles {
            indices.append(contentsOf: tri.map { Int32($0) })
            let a = pointsMM[tri[0]], b = pointsMM[tri[1]], c = pointsMM[tri[2]]
            let u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
            let v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
            let x = u.1 * v.2 - u.2 * v.1
            let y = u.2 * v.0 - u.0 * v.2
            let z = u.0 * v.1 - u.1 * v.0
            areas.append(0.5 * (x * x + y * y + z * z).squareRoot())
        }
        triangleAreas = areas
        totalArea = areas.reduce(0, +)
        let data = indices.withUnsafeBytes { Data($0) }
        element = SCNGeometryElement(
            data: data, primitiveType: .triangles,
            primitiveCount: triangles.count, bytesPerIndex: MemoryLayout<Int32>.size
        )
        material = SCNMaterial()
        material.diffuse.contents = NSColor.white
        material.lightingModel = .physicallyBased
        material.isDoubleSided = true
    }

    func valueRange(values: [Double], field: CFDField, revision: Int) -> ClosedRange<Double> {
        let key = FieldKey(field: field, revision: revision)
        if rangeKey != key {
            if let lo = values.min(), let hi = values.max(), hi > lo {
                cachedRange = lo...hi
            } else {
                cachedRange = 0...1
            }
            rangeKey = key
        }
        return cachedRange
    }

    func geometry(values: [Double], field: CFDField, revision: Int,
                  range: ClosedRange<Double>, threshold: ClosedRange<Double>?) -> SCNGeometry? {
        guard values.count == vertexCount else { return nil }
        let key = ColorKey(field: FieldKey(field: field, revision: revision), range: range, threshold: threshold)
        if colorKey == key { return cachedGeometry }

        let span = range.upperBound - range.lowerBound
        var colors: [Float] = []
        colors.reserveCapacity(vertexCount * 4)
        for value in values {
            let inBand = threshold.map { $0.contains(value) } ?? true
            let rgb = inBand
                ? CFDColormap.color(span > 0 ? (value - range.lowerBound) / span : 0)
                : CFDColormap.dimmedGray
            colors.append(contentsOf: [Float(rgb.x), Float(rgb.y), Float(rgb.z), 1])
        }
        let data = colors.withUnsafeBytes { Data($0) }
        let colorSource = SCNGeometrySource(
            data: data, semantic: .color, vectorCount: vertexCount,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: MemoryLayout<Float>.size * 4
        )
        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        geometry.materials = [material]
        colorKey = key
        cachedGeometry = geometry
        return geometry
    }

    func thresholdArea(values: [Double], range: ClosedRange<Double>) -> (area: Double, fraction: Double) {
        guard values.count == vertexCount else { return (0, 0) }
        var area = 0.0
        for (index, tri) in triangles.enumerated() {
            if range.contains(values[tri[0]]) && range.contains(values[tri[1]]) && range.contains(values[tri[2]]) {
                area += triangleAreas[index]
            }
        }
        return (area, totalArea > 0 ? area / totalArea : 0)
    }
}
