import { test, expect } from "@playwright/test";

test("orientation, right-drag zoom and middle-drag pan preserve voxel clicks", async ({
  page,
}) => {
  const study = {
    case_id: "navigation-test",
    shape: [10, 20, 30],
    spacing_mm: [2, 1, 1],
    modality: "CT",
    default_window: { wl: 300, ww: 800 },
    // Descending acquisition Z: upper anatomy is voxel Z=0.
    display_flips: {
      axial: { x: false, y: false },
      coronal: { x: false, y: true },
      sagittal: { x: false, y: true },
    },
  };
  const controlPoints: Record<string, number[][]> = {};
  const edits: number[][] = [];
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    let body: unknown = {};
    if (path.endsWith("/storage"))
      body = { path: ".", absolute_path: "/data/dicom", folders: [] };
    else if (path.endsWith("/series"))
      body = {
        series: [
          {
            series_uid: "synthetic",
            description: "Orientation fixture",
            modality: "CT",
            n_files: 10,
          },
        ],
      };
    else if (path === "/api/cases") body = study;
    else if (path.endsWith("/control_points") || path.endsWith("/centerline"))
      body = controlPoints;
    else if (path.endsWith("/points")) {
      const point = route.request().postDataJSON().point;
      edits.push(point);
      controlPoints["right:cca_ica"] = [...edits];
    } else if (path.endsWith("/slice")) {
      // Distinct top and bottom bands make orientation visible in screenshots.
      const axis = url.searchParams.get("axis");
      const width = axis === "sagittal" ? 20 : 30,
        height = axis === "axial" ? 20 : 10;
      return route.fulfill({
        contentType: "image/svg+xml",
        body: `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="100%" height="50%" fill="#555"/><rect y="50%" width="100%" height="50%" fill="#ddd"/></svg>`,
      });
    }
    await route.fulfill({ json: body });
  });
  await page.setViewportSize({ width: 1660, height: 1000 });
  await page.goto("http://127.0.0.1:8080");
  await page.getByRole("button", { name: "Выбрать папку DICOM…" }).click();
  await page.getByRole("button", { name: "Найти DICOM-серии" }).click();
  await page.locator(".series-item").click();
  await expect(page.locator(".slice-pane")).toHaveCount(3);
  await page
    .getByRole("button", { name: "Разметить путь", exact: true })
    .click();
  for (const [axis, label, rawX, rawY, expected] of [
    ["axial", "Аксиальная", 10.5, 8.5, [5, 8, 10]],
    ["coronal", "Корональная", 10.5, 11, [4, 10, 10]],
    ["sagittal", "Сагиттальная", 8.5, 11, [4, 8, 15]],
  ] as const) {
    const svg = page.locator(`.${axis} svg`),
      group = svg.locator("g");
    const point = () =>
      group.evaluate(
        (el, p) => {
          const matrix = (el as SVGGElement).getScreenCTM()!;
          const screen = new DOMPoint(...p).matrixTransform(matrix);
          return {
            x: screen.x,
            y: screen.y,
            scale: Math.abs(matrix.a),
            vertical: matrix.d,
          };
        },
        [rawX, rawY],
      );
    const before = await point();
    expect(before.vertical > 0).toBe(axis === "axial");
    const box = (await svg.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2 - 65, {
      steps: 8,
    });
    await page.mouse.up({ button: "right" });
    const zoomed = await point();
    expect(zoomed.scale).toBeGreaterThan(before.scale * 1.5);
    const editCount = edits.length;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down({ button: "middle" });
    await page.mouse.move(
      box.x + box.width / 2 + 36,
      box.y + box.height / 2 + 24,
      { steps: 5 },
    );
    await page.mouse.up({ button: "middle" });
    const panned = await point();
    expect(panned.x - zoomed.x).toBeCloseTo(36, 1);
    expect(panned.y - zoomed.y).toBeCloseTo(24, 1);
    expect(edits).toHaveLength(editCount);
    await page.mouse.click(panned.x, panned.y);
    await expect.poll(() => edits.length).toBe(editCount + 1);
    expect(edits.at(-1)).toEqual([...expected]);
    // Flipping the view also flips both markers and hit-testing, not the data.
    await page
      .getByRole("button", { name: `Перевернуть: ${label}`, exact: true })
      .click();
    const rotated = await point();
    await page.mouse.click(rotated.x, rotated.y);
    await expect.poll(() => edits.length).toBe(editCount + 2);
    expect(edits.at(-1)).toEqual([...expected]);
    await page
      .getByRole("button", { name: `Вписать: ${label}`, exact: true })
      .click();
    const reset = await point();
    expect(reset.x).toBeCloseTo(before.x, 1);
    expect(reset.y).toBeCloseTo(before.y, 1);
    expect(reset.scale).toBeCloseTo(before.scale, 2);
    // Pointer capture keeps a middle drag working outside the original pane.
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down({ button: "middle" });
    await page.mouse.move(box.x + box.width + 30, box.y + box.height / 2, {
      steps: 8,
    });
    await page.mouse.up({ button: "middle" });
    const captured = await point();
    expect(captured.x - reset.x).toBeCloseTo(box.width / 2 + 30, 1);
    await page
      .getByRole("button", { name: `Вписать: ${label}`, exact: true })
      .click();
  }
});
