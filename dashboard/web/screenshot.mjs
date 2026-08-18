import { chromium } from "playwright";

const [,, url, outPath] = process.argv;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForTimeout(1000);
await page.screenshot({ path: outPath, fullPage: true });
await browser.close();
console.log(`saved ${outPath}`);
