import { test, expect } from "@playwright/test";
test("synthetic study: image, path, contours, mesh, persistence and UI", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  const host = "http://127.0.0.1:8080";
  async function get(path: string) {
    const r = await request.get(host + "/api" + path);
    expect(r.ok(), await r.text()).toBeTruthy();
    return r.json();
  }
  async function post(path: string, data: unknown = {}) {
    const r = await request.post(host + "/api" + path, { data });
    expect(r.ok(), await r.text()).toBeTruthy();
    return r.json();
  }
  const source = "/data/dicom/synthetic-demo";
  const series = await get("/series?path=" + encodeURIComponent(source));
  const study = await post("/cases", {
    path: source,
    series_uid: series.series[0].series_uid,
  });
  const base = "/cases/" + study.case_id;
  const image = await request.get(
    host + "/api" + base + "/slice?axis=axial&index=20",
  );
  expect(image.status()).toBe(200);
  expect(image.headers()["content-type"]).toBe("image/png");
  const project = "/data/projects/e2e-" + Date.now();
  await post(base + "/project", { path: project });
  for (const z of [8, 18, 28])
    await post(base + "/centerline/right/cca_ica/points", {
      point: [z, 16, 16],
    });
  const prepared = await post(base + "/segment/right/cca_ica/prepare");
  for (let i = 0; i < prepared.n_points; i++)
    await post(base + "/segment/right/cca_ica/point/" + i, {
      method: "circle",
      radius_mm: 3,
    });
  await post(base + "/segment/right/build_surface");
  await post(base + "/segment/right/su2_mesh");
  await expect
    .poll(
      async () => (await get(base + "/segment/right/su2_mesh/status")).state,
      { timeout: 60000 },
    )
    .toBe("done");
  await post(base + "/save/project");
  const opened = await post("/projects/open", { path: project });
  expect(opened.restored.right.mesh_files_exist).toBe(true);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.setViewportSize({ width: 1660, height: 1000 });
  await page.goto(host);
  await page
    .getByRole("button", { name: "Открыть сохранённый проект…" })
    .click();
  await page.getByLabel("Папка проекта").fill(project);
  await page
    .getByRole("button", { name: "Открыть проект", exact: true })
    .click();
  await expect(page.locator(".slice-pane")).toHaveCount(3);
  await expect(page.locator(".point-card")).toHaveCount(3);
  await expect(page.locator(".path-canvas canvas")).toBeVisible();
  await page.getByRole("button", { name: "Поверхность…", exact: true }).click();
  await page
    .getByRole("button", { name: "Показать сохранённую геометрию" })
    .click();
  await expect(page.locator("dialog canvas")).toBeVisible();
  await page.getByRole("button", { name: "Закрыть окно" }).click();
  await page
    .getByRole("button", { name: "CFD Computation…", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Запустить CFD", exact: true }),
  ).toBeDisabled();
  await page.keyboard.press("Escape");
  await expect(page.locator("dialog")).not.toBeVisible();
  await page.screenshot({
    path: "../data/runtime/native-style-workbench.png",
    fullPage: true,
  });
  // Switching branches keeps the three slice positions; left/right data stays independent.
  await page.getByRole("button", { name: "CCA → ECA", exact: true }).click();
  await page
    .getByRole("button", { name: "Разметить путь", exact: true })
    .click();
  const svg = page.locator(".coronal svg");
  // Convert voxel display coordinates through the actual SVG transform, including letterboxing.
  await page
    .getByRole("button", { name: "Увеличить: Корональная", exact: true })
    .click();
  const screen = await svg.locator("g").evaluate((el) => {
    const p = new DOMPoint(16.5, 19.5).matrixTransform(
      (el as SVGGElement).getScreenCTM()!,
    );
    return { x: p.x, y: p.y };
  });
  await page.mouse.click(screen.x, screen.y);
  await expect(page.locator(".point-card")).toHaveCount(4);
  const controls = await get("/cases/" + opened.case_id + "/control_points");
  expect(controls["right:cca_eca"]).toEqual([[20, 16, 16]]);
  await page.getByRole("button", { name: "Левая", exact: true }).click();
  await expect(page.locator(".point-card")).toHaveCount(0);
  await page.getByRole("button", { name: "Правая", exact: true }).click();
  await expect(page.locator(".point-card")).toHaveCount(4);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
  const mutation = await request.delete(
    host + "/api" + base + "/segment/right/cca_ica/point/0",
  );
  expect(mutation.ok()).toBeTruthy();
  expect((await get(base + "/workflow/right")).meshing).toBe("stale");
  const saved = await post("/projects/open", { path: project });
  expect(saved.restored.right.mesh_files_exist).toBe(false);
});
