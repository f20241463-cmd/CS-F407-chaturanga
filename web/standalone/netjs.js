// Plain JavaScript forward pass of the folded 4x64 network (NHWC, weights in tfjs layout).
(function (root) {
  function decode(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
  // neighbour table: for each position p and kernel index k (0..8), source position or -1
  const NB = new Int16Array(64 * 9);
  for (let p = 0; p < 64; p++) for (let k = 0; k < 9; k++) {
    const r = (p >> 3) + ((k / 3) | 0) - 1, c = (p & 7) + (k % 3) - 1;
    NB[p * 9 + k] = r >= 0 && r < 8 && c >= 0 && c < 8 ? r * 8 + c : -1;
  }
  function makeNet(weights) {
    const W = {};
    for (const k in weights) W[k] = decode(weights[k][1]);
    const C = 64;
    const a = new Float32Array(64 * C), b = new Float32Array(64 * C), t = new Float32Array(64 * C);
    function conv3(x, Ci, w, bias, out, resid) {
      for (let p = 0; p < 64; p++) {
        const o = p * C;
        for (let co = 0; co < C; co++) out[o + co] = bias[co];
        for (let k = 0; k < 9; k++) {
          const q = NB[p * 9 + k];
          if (q < 0) continue;
          const xo = q * Ci, wk = k * Ci * C;
          for (let ci = 0; ci < Ci; ci++) {
            const v = x[xo + ci];
            if (v === 0) continue;
            const wo = wk + ci * C;
            for (let co = 0; co < C; co++) out[o + co] += v * w[wo + co];
          }
        }
        if (resid) for (let co = 0; co < C; co++) { const s = out[o + co] + resid[o + co]; out[o + co] = s > 0 ? s : 0; }
        else for (let co = 0; co < C; co++) if (out[o + co] < 0) out[o + co] = 0;
      }
    }
    const pfeat = new Float32Array(128), vfeat = new Float32Array(64), v1 = new Float32Array(64);
    return function (x) {
      conv3(x, 13, W["in.w"], W["in.b"], a, null);
      let h = a, other = b;
      for (let i = 0; i < 4; i++) {
        conv3(h, 64, W["r" + i + "a.w"], W["r" + i + "a.b"], t, null);
        conv3(t, 64, W["r" + i + "b.w"], W["r" + i + "b.b"], other, h);
        const s = h; h = other; other = s;
      }
      // 1x1 heads; NHWC flatten = p * Cout + co
      const pw = W["pc.w"], pb = W["pc.b"], vw = W["vc.w"], vb = W["vc.b"];
      for (let p = 0; p < 64; p++) {
        let s0 = pb[0], s1 = pb[1], sv = vb[0];
        const o = p * C;
        for (let ci = 0; ci < C; ci++) { const v = h[o + ci]; s0 += v * pw[ci * 2]; s1 += v * pw[ci * 2 + 1]; sv += v * vw[ci]; }
        pfeat[p * 2] = s0 > 0 ? s0 : 0; pfeat[p * 2 + 1] = s1 > 0 ? s1 : 0; vfeat[p] = sv > 0 ? sv : 0;
      }
      const logits = new Float32Array(W["pf.b"]);
      const fw = W["pf.w"];
      for (let i = 0; i < 128; i++) { const v = pfeat[i]; if (v === 0) continue; const o = i * 4096; for (let j = 0; j < 4096; j++) logits[j] += v * fw[o + j]; }
      const w1 = W["v1.w"], b1 = W["v1.b"];
      for (let j = 0; j < 64; j++) v1[j] = b1[j];
      for (let i = 0; i < 64; i++) { const v = vfeat[i]; if (v === 0) continue; for (let j = 0; j < 64; j++) v1[j] += v * w1[i * 64 + j]; }
      let val = W["v2.b"][0];
      for (let j = 0; j < 64; j++) if (v1[j] > 0) val += v1[j] * W["v2.w"][j];
      return { logits, value: Math.tanh(val) };
    };
  }
  const ORDER = ["in.w", "in.b"];
  for (let i = 0; i < 4; i++) ORDER.push("r" + i + "a.w", "r" + i + "a.b", "r" + i + "b.w", "r" + i + "b.b");
  ORDER.push("pc.w", "pc.b", "vc.w", "vc.b", "pf.w", "pf.b", "v1.w", "v1.b");
  // WebAssembly version: same arithmetic, several times faster. Resolves null if WebAssembly is unavailable.
  async function makeWasmNet(weights, wasmB64) {
    if (typeof WebAssembly === "undefined") return null;
    const bytes = decodeBytes(wasmB64);
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const e = instance.exports, mem = () => e.memory.buffer;
    e.init();
    const wPtr = e.get_w(), off = new Int32Array(mem(), e.get_off(), 26);
    let pos = 0;
    ORDER.forEach((k, i) => {
      const arr = decode(weights[k][1]);
      new Float32Array(mem(), wPtr + pos * 4, arr.length).set(arr);
      off[i] = pos; pos += arr.length;
    });
    const v2w = decode(weights["v2.w"][1]), v2b = decode(weights["v2.b"][1])[0];
    const inp = new Float32Array(mem(), e.get_in(), 64 * 13);
    const out = new Float32Array(mem(), e.get_out(), 4097);
    const v1 = new Float32Array(mem(), e.get_v1(), 64);
    return function (x) {
      inp.set(x);
      e.forward();
      let val = v2b;
      for (let j = 0; j < 64; j++) val += v1[j] * v2w[j];
      return { logits: out.slice(0, 4096), value: Math.tanh(val) };
    };
  }
  function decodeBytes(b64) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  if (typeof module !== "undefined") module.exports = { makeNet, makeWasmNet }; else root.CZNET = { makeNet, makeWasmNet };
})(typeof window !== "undefined" ? window : globalThis);
