import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../css/app.css", import.meta.url), "utf8");

function tokens(block) {
  return Object.fromEntries([...block.matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})/g)].map((m) => [m[1], m[2]]));
}
const light = tokens(css.slice(css.indexOf(":root {"), css.indexOf("}", css.indexOf(":root {"))));
const darkStart = css.indexOf(':root[data-theme="dark"]');
const dark = { ...light, ...tokens(css.slice(darkStart, css.indexOf("}", darkStart))) };

function lum(hex) {
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

const PAIRS = [["fg", "bg"], ["fg", "card"], ["mute", "card"], ["mute", "bg"], ["acc-fg", "acc"], ["ok", "card"], ["warn", "card"],
  ["err", "card"], ["chip-fg", "chip"], ["acc", "card"]];

for (const [name, t] of [["light", light], ["dark", dark]]) {
  test(`${name} theme text pairs meet WCAG AA 4.5:1`, () => {
    for (const [f, b] of PAIRS) {
      assert.ok(t[f] && t[b], `missing token ${f} or ${b}`);
      const r = ratio(t[f], t[b]);
      assert.ok(r >= 4.5, `${name}: --${f} on --${b} = ${r.toFixed(2)}`);
    }
  });
}

test("the prefers-color-scheme dark block equals the explicit data-theme=dark block", () => {
  const i = css.indexOf(':root:not([data-theme="light"])');
  assert.deepEqual(tokens(css.slice(i, css.indexOf("}", i))), tokens(css.slice(darkStart, css.indexOf("}", darkStart))));
});
