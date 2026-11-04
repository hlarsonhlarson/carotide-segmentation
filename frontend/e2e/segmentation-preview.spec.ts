import { test, expect } from "@playwright/test";

test("segmentation: native wall preview, current ring and live contour edits", async ({
  page,
  request,
}) => {
  test.setTimeout(180000);
  page.setDefaultTimeout(15000);
  const host = "http://127.0.0.1:8080";
  async function post(path: string, data: unknown = {}) {
    const response = await request.post(host + "/api" + path, { data });
    expect(response.ok(), await response.text()).toBeTruthy();
    return response.json();
  }
  const source = "/data/dicom/synthetic-demo";
  const series = await (
    await request.get(host + "/api/series?path=" + encodeURIComponent(source))
  ).json();
  const study = await post("/cases", {
    path: source,
    series_uid: series.series[0].series_uid,
  });
  const base = "/cases/" + study.case_id;
  const project = "/data/projects/e2e-seg-preview-" + Date.now();
  await post(base + "/project", { path: project });
  for (const z of [8, 14])
    await post(base + "/centerline/right/cca_ica/points", {
      point: [z, 16, 16],
    });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.setViewportSize({ width: 1660, height: 1100 });
  await page.goto(host);
  await page
    .getByRole("button", { name: "Открыть сохранённый проект…" })
    .click();
  await page.getByLabel("Папка проекта").fill(project);
  await page
    .getByRole("button", { name: "Открыть проект", exact: true })
    .click();
  const mainPreview = page.locator(".preview-pane");
  await mainPreview
    .locator("canvas")
    .evaluate((el) => el.setAttribute("data-persistent", "yes"));
  await page.getByRole("button", { name: "Контуры вдоль пути…" }).click();
  const dialog = page.locator("dialog"),
    preview = dialog.locator(".wall-preview");
  await expect(preview).toBeVisible();
  await dialog
    .getByRole("button", { name: "Подготовить сечения", exact: true })
    .click();
  await expect(dialog.locator(".section-position")).toContainText(
    "Сечение 1 /",
  );
  await expect(preview.locator("canvas")).toBeVisible();
  await expect(preview).toContainText("0 контуров · 0 участков стенки");
  await expect(preview).toContainText("кольцо радиуса");
  // The original canvas stays mounted while data and current slice change.
  await preview
    .locator("canvas")
    .evaluate((el) => el.setAttribute("data-original", "yes"));
  await dialog.getByLabel("Метод", { exact: true }).selectOption("circle");
  await dialog.getByRole("button", { name: "Применить к сечению" }).click();
  await expect(preview).toContainText("1 контуров · 0 участков стенки");
  await expect(preview).toContainText("готовый контур");
  await dialog.getByRole("button", { name: "Далее ▶" }).click();
  await expect(preview).toContainText("Сечение 2 /");
  await expect(preview).toContainText("кольцо радиуса");
  await dialog.getByRole("button", { name: "Применить к сечению" }).click();
  await expect(preview).toContainText("2 контуров · 1 участков стенки");
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    "ICA: 1 участков стенки",
  );
  await expect(mainPreview.locator("canvas")).toHaveAttribute(
    "data-persistent",
    "yes",
  );
  await expect(preview.locator("canvas")).toHaveAttribute(
    "data-original",
    "yes",
  );
  // A real drag renders another view; middle/right retain their requested roles.
  const canvas = preview.locator("canvas"),
    box = (await canvas.boundingBox())!;
  const initialImage = await canvas.screenshot();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 60,
    box.y + box.height / 2 + 25,
    { steps: 8 },
  );
  await page.mouse.up();
  expect((await canvas.screenshot()).equals(initialImage)).toBe(false);
  await dialog.getByRole("button", { name: "Удалить контур" }).click();
  await expect(preview).toContainText("1 контуров · 0 участков стенки");
  await expect(preview).toContainText("кольцо радиуса");
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    "ICA: 0 участков стенки",
  );
  await dialog.getByRole("button", { name: "Обработать все сечения" }).click();
  await expect(
    dialog.getByRole("button", { name: "Обработать все сечения" }),
  ).toBeEnabled({ timeout: 120000 });
  const result = await (
    await request.get(host + "/api" + base + "/segment/right/cca_ica")
  ).json();
  const total = result.points.length;
  expect(total).toBeGreaterThan(2);
  expect(result.points.every((p: any) => p.contour_mm?.length)).toBe(true);
  await expect(preview).toContainText(
    `${total} контуров · ${total - 1} участков стенки`,
  );
  await dialog.getByRole("button", { name: "Вписать 3D-модель" }).click();
  await page.screenshot({
    path: "../data/runtime/segmentation-wall-preview.png",
    fullPage: true,
  });
  // The mask remains over the path after closing Segmentation.
  await dialog.getByRole("button", { name: "Закрыть окно" }).click();
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    `ICA: ${total - 1} участков стенки`,
  );
  await page.getByRole("button", { name: "Левая", exact: true }).click();
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    "ICA: 0 участков стенки",
  );
  await page.getByRole("button", { name: "Правая", exact: true }).click();
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    `ICA: ${total - 1} участков стенки`,
  );
  // Reopening the saved project reloads its contours for the main viewport.
  await page.reload();
  await page
    .getByRole("button", { name: "Открыть сохранённый проект…" })
    .click();
  await page.getByLabel("Папка проекта").fill(project);
  await page
    .getByRole("button", { name: "Открыть проект", exact: true })
    .click();
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    `ICA: ${total - 1} участков стенки`,
  );
  await page.screenshot({
    path: "../data/runtime/path-with-segmentation.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Контуры вдоль пути…" }).click();
  await expect(preview).toContainText(
    `${total} контуров · ${total - 1} участков стенки`,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(await dialog.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(
    true,
  );
  await dialog.getByRole("button", { name: "Закрыть окно" }).click();
  // Editing the centerline invalidates the segmentation rather than leaving a stale mask.
  await page
    .getByRole("button", { name: "Очистить ветвь", exact: true })
    .click();
  await expect(mainPreview.locator(".path-wall-status")).toContainText(
    "ICA: 0 участков стенки",
  );
  await expect(page.locator(".point-card")).toHaveCount(0);
  expect(errors).toEqual([]);
});
