// ChaturangZero rules and search, ported line for line from chaturanga/ and mcts/.
// Board: Int8Array(64), 0 = empty, otherwise 1 + type + 6 * color.
// Types: 0 raja, 1 mantri, 2 ratha, 3 gaja, 4 ashva, 5 padati. Colors: 0 white, 1 black.
(function (root) {
  "use strict";
  const RAJA = 0, MANTRI = 1, RATHA = 2, GAJA = 3, ASHVA = 4, PADATI = 5;
  const WHITE = 0, BLACK = 1;
  const NO_PROGRESS_LIMIT = 100;
  const typeOf = (p) => (p - 1) % 6;
  const colorOf = (p) => (p - 1 >= 6 ? 1 : 0);
  const make = (t, c) => 1 + t + 6 * c;
  const onBoard = (r, c) => r >= 0 && r < 8 && c >= 0 && c < 8;

  const ORTHO = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const DIAG = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const ALL8 = ORTHO.concat(DIAG);
  const KNIGHT = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]];
  const GAJA_D = [[2, 2], [2, -2], [-2, 2], [-2, -2]];
  const fwd = (color) => (color === WHITE ? 1 : -1);

  function offsetTable(deltas) {
    const t = [];
    for (let sq = 0; sq < 64; sq++) {
      const r = sq >> 3, c = sq & 7, out = [];
      for (const [dr, dc] of deltas) if (onBoard(r + dr, c + dc)) out.push((r + dr) * 8 + c + dc);
      t.push(out);
    }
    return t;
  }
  function rayTable(deltas) {
    const t = [];
    for (let sq = 0; sq < 64; sq++) {
      const r = sq >> 3, c = sq & 7, rays = [];
      for (const [dr, dc] of deltas) {
        const ray = [];
        let nr = r + dr, nc = c + dc;
        while (onBoard(nr, nc)) { ray.push(nr * 8 + nc); nr += dr; nc += dc; }
        rays.push(ray);
      }
      t.push(rays);
    }
    return t;
  }
  const JUMP = { [RAJA]: offsetTable(ALL8), [MANTRI]: offsetTable(DIAG), [GAJA]: offsetTable(GAJA_D), [ASHVA]: offsetTable(KNIGHT) };
  const JUMP_ORDER = [RAJA, MANTRI, GAJA, ASHVA]; // insertion order of _JUMP_TABLES in moves.py
  const RAYS = rayTable(ORTHO);
  const PUSH = [WHITE, BLACK].map((col) => {
    const a = [];
    for (let sq = 0; sq < 64; sq++) { const r = (sq >> 3) + fwd(col), c = sq & 7; a.push(onBoard(r, c) ? r * 8 + c : -1); }
    return a;
  });
  const CAPS = [WHITE, BLACK].map((col) => offsetTable([[fwd(col), -1], [fwd(col), 1]]));
  const PAWN_ATTACKERS = [WHITE, BLACK].map((col) => offsetTable([[-fwd(col), -1], [-fwd(col), 1]]));

  function pseudoMoves(sqs, color) {
    const moves = [];
    for (let sq = 0; sq < 64; sq++) {
      const p = sqs[sq];
      if (!p || colorOf(p) !== color) continue;
      const pt = typeOf(p);
      if (pt === RATHA) {
        for (const ray of RAYS[sq]) for (const t of ray) {
          const o = sqs[t];
          if (!o) moves.push([sq, t]);
          else { if (colorOf(o) !== color) moves.push([sq, t]); break; }
        }
      } else if (pt === PADATI) {
        const push = PUSH[color][sq];
        if (push >= 0 && !sqs[push]) moves.push([sq, push]);
        for (const t of CAPS[color][sq]) { const o = sqs[t]; if (o && colorOf(o) !== color) moves.push([sq, t]); }
      } else {
        for (const t of JUMP[pt][sq]) { const o = sqs[t]; if (!o || colorOf(o) !== color) moves.push([sq, t]); }
      }
    }
    return moves;
  }

  function isAttacked(sqs, sq, by) {
    for (const o of PAWN_ATTACKERS[by][sq]) { const p = sqs[o]; if (p && colorOf(p) === by && typeOf(p) === PADATI) return true; }
    const target = sqs[sq];
    if (target && colorOf(target) === by) return false;
    for (const pt of JUMP_ORDER) for (const o of JUMP[pt][sq]) { const p = sqs[o]; if (p && colorOf(p) === by && typeOf(p) === pt) return true; }
    for (const ray of RAYS[sq]) for (const o of ray) {
      const p = sqs[o];
      if (p) { if (colorOf(p) === by && typeOf(p) === RATHA) return true; break; }
    }
    return false;
  }

  function findKing(sqs, color) {
    for (let i = 0; i < 64; i++) { const p = sqs[i]; if (p && colorOf(p) === color && typeOf(p) === RAJA) return i; }
    return -1;
  }

  function legalMoves(board, color) {
    const sqs = Int8Array.from(board);
    const king = findKing(sqs, color);
    const enemy = 1 - color, legal = [];
    for (const m of pseudoMoves(sqs, color)) {
      const [f, t] = m, moving = sqs[f], captured = sqs[t];
      sqs[f] = 0; sqs[t] = moving;
      const k = typeOf(moving) === RAJA ? t : king;
      if (k >= 0 && !isAttacked(sqs, k, enemy)) legal.push(m);
      sqs[f] = moving; sqs[t] = captured;
    }
    return legal;
  }

  // ---- GameState (immutable) ----
  class GameState {
    constructor(board, side, clock, history) {
      this.board = board || startBoard();
      this.side = side === undefined ? WHITE : side;
      this.clock = clock || 0;
      this.history = history || []; // snapshot keys of earlier positions
      this._legal = null; this._result = undefined; this._key = null;
    }
    key() { if (this._key === null) this._key = String.fromCharCode.apply(null, this.board) + this.side; return this._key; }
    legalMoves() { if (!this._legal) this._legal = legalMoves(this.board, this.side); return this._legal; }
    isInCheck(color) {
      color = color === undefined ? this.side : color;
      const k = findKing(this.board, color);
      return k < 0 ? false : isAttacked(this.board, k, 1 - color);
    }
    apply(move) {
      const [f, t] = move;
      const b = Int8Array.from(this.board);
      let piece = b[f];
      const capture = b[t] !== 0, padati = typeOf(piece) === PADATI;
      b[f] = 0;
      if (padati) { const row = t >> 3; if ((colorOf(piece) === WHITE && row === 7) || (colorOf(piece) === BLACK && row === 0)) piece = make(MANTRI, colorOf(piece)); }
      b[t] = piece;
      const clock = capture || padati ? 0 : this.clock + 1;
      return new GameState(b, 1 - this.side, clock, this.history.concat([this.key()]));
    }
    isThreefold() { const k = this.key(); let n = 1; for (const h of this.history) if (h === k) n++; return n >= 3; }
    result() {  // +1 side to move won, -1 lost, 0 draw, null ongoing
      if (this._result !== undefined) return this._result;
      let r = null;
      if (findKing(this.board, this.side) < 0) r = -1;
      else if (this.legalMoves().length === 0) r = this.isInCheck() ? -1 : 0;
      else if (this.clock >= NO_PROGRESS_LIMIT) r = 0;
      else if (this.isThreefold()) r = 0;
      this._result = r;
      return r;
    }
    isOver() { return this.result() !== null; }
    ending() {
      if (!this.isOver()) return "unfinished";
      if (findKing(this.board, this.side) < 0) return "king captured";
      if (this.legalMoves().length === 0) return this.isInCheck() ? "checkmate" : "stalemate";
      if (this.clock >= NO_PROGRESS_LIMIT) return "no-progress rule";
      return "threefold repetition";
    }
    // NHWC input for the network: [8, 8, 13]
    tensor() {
      const x = new Float32Array(8 * 8 * 13);
      for (let sq = 0; sq < 64; sq++) { const p = this.board[sq]; if (p) x[sq * 13 + (p - 1)] = 1; }
      if (this.side === BLACK) for (let sq = 0; sq < 64; sq++) x[sq * 13 + 12] = 1;
      return x;
    }
  }

  function startBoard() {
    const back = [RATHA, ASHVA, GAJA, RAJA, MANTRI, GAJA, ASHVA, RATHA];
    const b = new Int8Array(64);
    for (let c = 0; c < 8; c++) {
      b[c] = make(back[c], WHITE); b[56 + c] = make(back[c], BLACK);
      b[8 + c] = make(PADATI, WHITE); b[48 + c] = make(PADATI, BLACK);
    }
    return b;
  }

  const FILES = "abcdefgh";
  const alg = (sq) => FILES[sq & 7] + ((sq >> 3) + 1);
  const moveText = (m) => alg(m[0]) + alg(m[1]);

  // ---- MCTS (mcts/node.py + mcts/search.py) ----
  class Node {
    constructor(state, parent, prior, action, move) {
      this._state = state || null; this.parent = parent || null; this.prior = prior || 0;
      this.action = action === undefined ? null : action; this.move = move || null;
      this.actions = []; this.priors = []; this.made = new Map();
      this.N = 0; this.W = 0; this.expanded = false;
    }
    get state() { if (!this._state) this._state = this.parent.state.apply(this.move); return this._state; }
    get value() { return this.N === 0 ? 0 : this.W / this.N; }
    expand(actions, priors) {
      if (this.actions.length) { this.expanded = true; return; }
      this.actions = actions; this.priors = priors; this.expanded = true;
    }
    child(action, prior) {
      let c = this.made.get(action);
      if (!c) { c = new Node(null, this, prior, action, [action >> 6, action & 63]); this.made.set(action, c); }
      return c;
    }
    select(cpuct) {
      let best = -Infinity, bi = -1;
      const sq = Math.sqrt(this.N);
      for (let i = 0; i < this.actions.length; i++) {
        const c = this.made.get(this.actions[i]);
        let s;
        if (!c) s = cpuct * this.priors[i] * sq;
        else s = -c.value + cpuct * c.prior * sq / (1 + c.N);
        if (s > best) { best = s; bi = i; }
      }
      return this.child(this.actions[bi], this.priors[bi]);
    }
  }

  // net(Float32Array NHWC) -> Promise<{logits: Float32Array(4096), value: number}>
  async function evaluateAndExpand(node, net) {
    const st = node.state;
    const out = await net(st.tensor());
    const actions = st.legalMoves().map((m) => m[0] * 64 + m[1]).sort((a, b) => a - b);
    let mx = -Infinity;
    for (const a of actions) mx = Math.max(mx, out.logits[a]);
    const ex = actions.map((a) => Math.exp(out.logits[a] - mx));
    const tot = ex.reduce((s, v) => s + v, 0);
    const priors = tot > 0 ? ex.map((v) => v / tot) : actions.map(() => 1 / actions.length);
    node.expand(actions, priors);
    return out.value;
  }

  // Same steps as mcts_search; root may be a kept, already expanded node.
  async function search(rootState, net, sims, cpuct, root, onProgress) {
    if (!root || !root.expanded) { root = new Node(rootState); await evaluateAndExpand(root, net); }
    if (!root.actions.length) return { root, visits: null };
    for (let i = 0; i < sims; i++) {
      let node = root; const path = [node];
      while (node.expanded && node.actions.length) { node = node.select(cpuct); path.push(node); }
      let value;
      if (node.state.isOver()) value = node.state.result();
      else value = await evaluateAndExpand(node, net);
      for (let j = path.length - 1; j >= 0; j--) { path[j].N += 1; path[j].W += value; value = -value; }
      if (onProgress && (i & 63) === 63) await onProgress(i + 1, sims);
    }
    const visits = new Map();
    for (const a of root.actions) { const c = root.made.get(a); visits.set(a, c ? c.N : 0); }
    return { root, visits };
  }

  // filter_repeating_moves + argmax over action ids (first max wins, like np.argmax)
  function chooseMove(state, visits) {
    const filtered = new Map(visits);
    let any = 0;
    for (const [a, n] of filtered) {
      if (n > 0 && state.apply([a >> 6, a & 63]).isThreefold()) filtered.set(a, 0);
    }
    for (const n of filtered.values()) any += n;
    const use = any > 0 ? filtered : visits;
    let best = -1, bn = -1;
    for (const a of [...use.keys()].sort((x, y) => x - y)) if (use.get(a) > bn) { bn = use.get(a); best = a; }
    return { action: best, counts: use };
  }

  const api = { GameState, Node, search, chooseMove, legalMoves, startBoard, moveText, alg, typeOf, colorOf, WHITE, BLACK };
  if (typeof module !== "undefined") module.exports = api; else root.CZ = api;
})(typeof window !== "undefined" ? window : globalThis);
