import test from "node:test";
import assert from "node:assert/strict";
import { appendToken, cutText, overflow, PLACEHOLDER, CAP } from "../js/turns.js";

test("first token replaces the placeholder, later tokens append", () => {
  assert.equal(appendToken(PLACEHOLDER, "The"), "The");
  assert.equal(appendToken("", "The"), "The");
  assert.equal(appendToken("The", " warranty"), "The warranty");
});

test("an interrupted answer ends with one ellipsis", () => {
  assert.equal(cutText("The warranty is"), "The warranty is…");
  assert.equal(cutText("The warranty is …"), "The warranty is…");
  assert.equal(cutText(PLACEHOLDER), PLACEHOLDER);
  assert.equal(cutText(""), PLACEHOLDER);
});

test("overflow counts bubbles past the cap, two per turn", () => {
  assert.equal(overflow(0), 0);
  assert.equal(overflow(CAP * 2), 0);
  assert.equal(overflow(CAP * 2 + 3), 3);
});
