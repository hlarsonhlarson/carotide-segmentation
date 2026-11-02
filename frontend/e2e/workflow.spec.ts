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
  await page.goto(host);
  await page.getByLabel("Папка проекта").fill(project);
  await page
    .getByRole("button", { name: "Открыть проект", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Путь сосуда", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Контрольные точки: 3")).toBeVisible();
  await page.getByLabel("Плоскость").selectOption("coronal");
  await expect(page.getByAltText("DICOM-срез")).toBeVisible();
  await page.getByRole("button", { name: "Геометрия", exact: true }).click();
  await page.getByRole("button", { name: "Показать сохранённую геометрию" }).click();
  await expect(page.locator("canvas")).toBeVisible();
  await page.screenshot({path: "/private/tmp/carotige-geometry.png", fullPage: true});
  await page.getByRole("button", { name: "Расчёт", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Запустить CFD", exact: true }),
  ).toBeDisabled();
  await page.screenshot({
    path: "/private/tmp/carotige-workbench.png",
    fullPage: true,
  });
  const mutation = await request.delete(
    host + "/api" + base + "/segment/right/cca_ica/point/0",
  );
  expect(mutation.ok()).toBeTruthy();
  expect((await get(base + "/workflow/right")).meshing).toBe("stale");
  const saved = await post("/projects/open", { path: project });
  expect(saved.restored.right.mesh_files_exist).toBe(false);
});
