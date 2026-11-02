import { describe, it, expect } from "vitest";
import { voxelAt } from "./PathEditor";
describe("display coordinates", () => {
  it("maps axial clicks to z,y,x", () =>
    expect(voxelAt("axial", 3, 4, 5, [10, 20, 30])).toEqual([3, 4, 5]));
  it("reverses superior-up coronal rows", () =>
    expect(voxelAt("coronal", 3, 4, 5, [10, 20, 30])).toEqual([5, 3, 5]));
  it("reverses superior-up sagittal rows", () =>
    expect(voxelAt("sagittal", 3, 4, 5, [10, 20, 30])).toEqual([5, 5, 3]));
});
