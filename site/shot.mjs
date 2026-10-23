// The site's layout check: headless Chrome through the DevTools protocol
// with device emulation. Prints the page's layout width against the viewport
// (a horizontal overflow shows as scrollW > innerW) and the first elements
// that reach past the viewport, then writes a full-page screenshot, or with
// SHOT_SELECTOR set, a screenshot of that element only.
//
//   node site/shot.mjs URL WIDTH HEIGHT OUT.png [mobile]
//
// Device emulation matters: a plain --window-size is clamped to about 500 px
// wide, which fakes an overflow at phone width. Each run gets a Chrome
// profile of its own, removed when it ends, and Chrome is stopped even when
// the run fails: with one shared profile, a run that died left its Chrome
// holding the profile lock and every later run failed to open the DevTools
// port. CHROME names the browser binary (default: the macOS install).
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, W, H, out, mobile] = process.argv.slice(2);
if (!url || !(+W > 0) || !(+H > 0) || !out) {
  console.error("usage: node site/shot.mjs URL WIDTH HEIGHT OUT.png [mobile]");
  process.exit(2);
}
const chrome = process.env.CHROME ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profile = mkdtempSync(join(tmpdir(), "shot-profile-"));
const port = 9333 + Math.floor(Math.random() * 600);
const proc = spawn(chrome, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
const exited = new Promise((r) => proc.once("exit", r));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function run() {
  let targets;
  for (let i = 0; i < 40 && !targets; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); }
    catch { await sleep(250); }
  }
  const page = (targets || []).find((t) => t.type === "page");
  if (!page) throw new Error(`Chrome opened no DevTools page on port ${port}`);
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  const events = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id) { pending.get(d.id)(d); pending.delete(d.id); } else events.push(d.method);
  };
  const send = (method, params = {}) => new Promise((r) => {
    ws.send(JSON.stringify({ id: ++id, method, params })); pending.set(id, r);
  });
  const evaluate = async (expression) =>
    (await send("Runtime.evaluate", { expression, returnByValue: true })).result.result.value;
  const metrics = (height) => send("Emulation.setDeviceMetricsOverride",
    { width: +W, height, deviceScaleFactor: 1, mobile: !!mobile });

  await send("Page.enable");
  await send("Runtime.enable");
  await metrics(+H);
  await send("Page.navigate", { url });
  for (let i = 0; i < 40 && !events.includes("Page.loadEventFired"); i++) await sleep(250);
  // the scroll-revealed elements, shown as a reader who scrolled sees them
  await evaluate("document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in'))");
  await sleep(2600);
  const fit = await evaluate(`JSON.stringify({scrollW: document.documentElement.scrollWidth,
    innerW: window.innerWidth, docH: document.documentElement.scrollHeight,
    wide: [...document.querySelectorAll('body *')]
      .filter(e => e.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 6).map(e => e.tagName + '.' + e.className)})`);
  console.log(url, W, fit);
  const { docH } = JSON.parse(fit);
  const sel = process.env.SHOT_SELECTOR;
  let clip;
  if (sel) {
    const box = JSON.parse(await evaluate(`(() => { const e = document.querySelector(${JSON.stringify(sel)});
      if (!e) return "null"; const r = e.getBoundingClientRect();
      return JSON.stringify({x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height}); })()`));
    if (!box) throw new Error(`no element matches ${sel}`);
    console.log("element", sel, JSON.stringify(box));
    clip = { ...box, height: Math.min(box.height, 6000), scale: 1 };
  }
  await metrics(Math.min(docH, sel ? 30000 : 12000));
  await sleep(500);
  const shot = await send("Page.captureScreenshot",
    { format: "png", captureBeyondViewport: true, ...(clip ? { clip } : {}) });
  writeFileSync(out, Buffer.from(shot.result.data, "base64"));
  ws.close();
}

let status = 0;
try {
  await run();
} catch (e) {
  console.error(String(e));
  status = 1;
} finally {
  proc.kill();
  await Promise.race([exited, sleep(5000)]);
  rmSync(profile, { recursive: true, force: true });
}
process.exit(status);
