const CZ = require('./engine.js');
const ref = require('./ref.json');
let bad = 0, n = 0;
const endings = {};
for (const g of ref.games) {
  let s = new CZ.GameState();
  for (let i = 0; i < g.plies.length; i++) {
    const [legal, result, check, ending, clock] = g.plies[i];
    const jl = s.legalMoves().map(m => m[0]*64+m[1]).sort((a,b)=>a-b);
    const r = s.result();
    const ok = JSON.stringify(jl) === JSON.stringify(legal) && r === result && s.isInCheck() === check && (ending === null || s.ending() === ending) && s.clock === clock;
    if (ending) endings[ending] = (endings[ending]||0)+1;
    n++;
    if (!ok) { if (bad++ < 5) console.log('mismatch', i, {jl: jl.length, legal: legal.length, r, result, ending, je: s.ending()}); break; }
    if (i < g.moves.length) s = s.apply(g.moves[i]);
  }
}
console.log('rules plies checked', n, 'mismatches', bad, endings);
function fake(x) {
  let h = 0; for (let i = 0; i < x.length; i++) h += x[i] * (i % 97);
  const logits = new Float32Array(4096);
  for (let a = 0; a < 4096; a++) logits[a] = Math.sin(a*0.37 + h*0.001)*3;
  return Promise.resolve({ logits, value: Math.fround(Math.tanh(Math.sin(h*0.013))) });
}
(async () => {
  let sbad = 0;
  for (const t of ref.search) {
    let s = new CZ.GameState();
    for (const m of t.seq) s = s.apply(m);
    const { root, visits } = await CZ.search(s, fake, 300, 1.5, null);
    const v = {}; for (const [a, c] of visits) if (c > 0) v[a] = c;
    const ch = CZ.chooseMove(s, visits).action;
    const same = JSON.stringify(v) === JSON.stringify(t.visits) && ch === t.choice && Math.abs(root.value - t.root_value) < 1e-9;
    if (!same) { sbad++; console.log('search mismatch', ch, t.choice, root.value, t.root_value); }
  }
  console.log('searches', ref.search.length, 'mismatches', sbad);
})();
