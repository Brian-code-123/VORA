// Paste/inject into the page (browser pane / devtools). Returns a list of layout violations for the current viewport.
(() => {
  const out = [], vw = document.documentElement.clientWidth, vh = window.innerHeight;
  if (document.documentElement.scrollWidth > vw + 1) out.push(`horizontal scroll: ${document.documentElement.scrollWidth} > ${vw}`);
  for (const el of document.querySelectorAll("body *")) {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || el.closest("[hidden]") || el.classList.contains("vh") || el.closest(".filebtn input")) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    if (r.right > vw + 1 || r.left < -1) out.push(`past viewport: ${el.tagName}#${el.id}.${el.className} [${Math.round(r.left)}, ${Math.round(r.right)}]`);
    if (vw <= 768 && el.matches("button, select, .filebtn") && (r.height < 44 || r.width < 44)) out.push(`small target: ${el.tagName}#${el.id} ${Math.round(r.width)}x${Math.round(r.height)}`);
  }
  window.scrollTo(0, document.documentElement.scrollHeight);
  const dock = document.querySelector(".dock").getBoundingClientRect();
  const last = [...document.querySelectorAll("main .card, .attrib")].at(-1).getBoundingClientRect();
  if (last.bottom > dock.top + 1) out.push(`dock covers content: last bottom ${Math.round(last.bottom)} > dock top ${Math.round(dock.top)}`);
  window.scrollTo(0, 0);
  return { vw, vh, violations: out.slice(0, 20) };
})();
