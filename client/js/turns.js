// Pure helpers for the chat view. No DOM, importable in Node for tests.
export const PLACEHOLDER = "…";
export const CAP = 20;                                    // turns kept on screen (two bubbles each)

export const appendToken = (cur, tok) => (cur === "" || cur === PLACEHOLDER ? tok : cur + tok);

// An interrupted answer keeps what was said and ends with an ellipsis; nothing said yet leaves the placeholder.
export const cutText = (text) => (!text || text === PLACEHOLDER ? PLACEHOLDER : `${text.replace(/[\s…]+$/, "")}…`);

export const overflow = (nodes, cap = CAP) => Math.max(0, nodes - cap * 2);
