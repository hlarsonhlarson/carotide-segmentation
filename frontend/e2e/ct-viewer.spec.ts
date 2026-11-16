import { test, expect } from "@playwright/test";
import fs from "node:fs";
// Real studies remain local. Supply an ignored JSON file with path and series_uid.
test("local CT: three planes, contrast, navigation and responsive layout", async ({
  page,
}) => {
  test.skip(!process.env.CAROTID_CT_TEST_INPUT, "Optional local CT study");
  test.setTimeout(180000);
  const input = JSON.parse(
    fs.readFileSync(process.env.CAROTID_CT_TEST_INPUT!, "utf8"),
  );
  const errors: string[] = [];
  const sliceResponses: number[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("response", (r) => {
    if (r.url().includes("/slice?")) sliceResponses.push(r.status());
  });
  await page.setViewportSize({ width: 1660, height: 1000 });
  await page.goto("http://127.0.0.1:8080");
  await page.getByRole("button", { name: "Выбрать папку DICOM…" }).click();
  await page.getByLabel("Папка снимков").fill(input.path);
  await page.getByRole("button", { name: "Найти DICOM-серии" }).click();
  await expect(page.locator(".series-item").first()).toBeVisible({
    timeout: 120000,
  });
  // The selected local fixture contains one CT series; require this to avoid guessing.
  await expect(page.locator(".series-item")).toHaveCount(1);
  await page.locator(".series-item").click();
  await expect(page.locator(".slice-pane")).toHaveCount(3, { timeout: 120000 });
  await expect.poll(() => sliceResponses.length).toBeGreaterThanOrEqual(3);
  const coronal = page.getByRole("slider", {
    name: "Срез: Корональная",
    exact: true,
  });
  await coronal.focus();
  await page.keyboard.press("ArrowRight");
  const before = await coronal.inputValue();
  await page.locator(".coronal svg").hover();
  await page.mouse.wheel(0, 100);
  await expect(coronal).toHaveValue(String(+before + 1));
  const level = page.getByRole("slider", { name: "Уровень окна", exact: true });
  const newLevel = +(await level.inputValue()) + 127;
  const changed = page.waitForResponse(
    (r) => r.url().includes("axis=axial") && r.url().includes("wl=" + newLevel),
  );
  await level.fill(String(newLevel));
  expect((await changed).status()).toBe(200);
  await expect(page.locator(".axial image")).toHaveAttribute(
    "href",
    new RegExp("wl=" + newLevel),
  );
  await page.getByRole("button", { name: "Сбросить", exact: true }).click();
  await page.screenshot({
    path: "../data/runtime/ct-native-style.png",
    fullPage: true,
  });
  for (const width of [1280, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      `overflow at ${width}`,
    ).toBe(true);
    for (const axis of ["axial", "coronal", "sagittal"]) {
      const box = await page.locator(`.${axis} svg`).boundingBox();
      expect(box!.width).toBeGreaterThan(60);
      expect(box!.height).toBeGreaterThan(60);
    }
  }
  expect(sliceResponses.every((s) => s === 200)).toBe(true);
  expect(errors).toEqual([]);
});
