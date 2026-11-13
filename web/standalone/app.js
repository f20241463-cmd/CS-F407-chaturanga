(function () {
  "use strict";
  const SIMS = 800;          // same search size as the earlier human games against round 560
  const CPUCT = 1.5;
  const MODEL = "models/chaturangzero_final.pt (iteration 1220, web)";
  const GLYPH = ["♚", "♛", "♜", "♝", "♞", "♟"];
  const NAMES = ["raja", "mantri", "ratha", "gaja", "ashva", "padati"];
  const FILES = "abcdefgh";
  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } },
  };

  // ---- network (WebAssembly, or plain JavaScript as a fallback) ----
  let net = null, netCheck = null, msPerEval = null;
  // Yield to the browser so the page can repaint; message events are not throttled like timers.
  const chan = new MessageChannel(), waiting = [];
  chan.port1.onmessage = () => { const r = waiting.shift(); if (r) r(); };
  const pause = () => new Promise((r) => { waiting.push(r); chan.port2.postMessage(0); });
  async function initNet() {
    let fn = null, backend = "javascript";
    try { fn = await CZNET.makeWasmNet(window.CZ_WEIGHTS, window.CZ_WASM); if (fn) backend = "wasm"; } catch (e) { fn = null; }
    if (!fn) fn = CZNET.makeNet(window.CZ_WEIGHTS);
    net = (x) => Promise.resolve(fn(x));
    // Check the network against outputs saved from the Python model.
    let s = new CZ.GameState(), ply = 0, maxDiff = 0;
    const refs = window.CZ_REFS;
    for (const r of refs.refs) {
      while (ply < r.moves_from_start) { s = s.apply(refs.seq[ply]); ply++; }
      const out = fn(s.tensor());
      maxDiff = Math.max(maxDiff, Math.abs(out.value - r.value));
      r.legal.forEach((a, i) => { maxDiff = Math.max(maxDiff, Math.abs(out.logits[a] - r.logits[i])); });
    }
    netCheck = { backend, maxDiff: Number(maxDiff.toExponential(2)) };
    const x = new CZ.GameState().tensor();
    for (let i = 0; i < 10; i++) fn(x);
    const t0 = performance.now();
    for (let i = 0; i < 40; i++) fn(x);
    msPerEval = (performance.now() - t0) / 40;
    window.CZ_STATUS = { netCheck, msPerEval };
    console.log("[ChaturangZero] network check", JSON.stringify(netCheck), "ms per evaluation", msPerEval.toFixed(2));
  }

  // ---- saving games ----
  let db = null;
  window.claude?.use?.("db").then((d) => { db = d; }).catch(() => { db = null; });
  let games = store.get("cz-games", []);
  function tally() {
    let w = 0, l = 0, d = 0;
    for (const g of games) { if (g.result === "human_win") w++; else if (g.result === "agent_win") l++; else if (g.result === "draw") d++; }
    $("tally").textContent = `${w} won, ${l} lost, ${d} drawn`;
    const box = $("codeBox");
    box.hidden = games.length === 0 || games.every((g) => g.saved);
    $("codes").value = games.map((g) => JSON.stringify(g.record)).join("\n");
  }
  async function record(result, ending) {
    const rec = {
      date: new Date().toISOString().slice(0, 19).replace("T", " "),
      player: ($("player").value || "").trim() || "guest",
      checkpoint: MODEL, simulations: SIMS, human_color: G.human === CZ.WHITE ? "white" : "black",
      result, ending, plies: G.moves.length, moves: G.moves.join(" "),
      backend: netCheck && netCheck.backend, net_check: netCheck && netCheck.maxDiff,
    };
    const entry = { record: rec, result, saved: false };
    games.push(entry);
    store.set("cz-games", games);
    tally();
    $("saveState").textContent = "Saving the game…";
    try {
      if (!db && window.claude && window.claude.use) db = await window.claude.use("db");
      if (!db) throw new Error("no online storage");
      await db.collection("games").add(rec);
      entry.saved = true;
      store.set("cz-games", games);
      $("saveState").innerHTML = '<span class="ok">Last game saved online.</span>';
    } catch (e) {
      $("saveState").textContent = "This game could not be saved online. Copy the text below and send it instead.";
    }
    tally();
  }

  // ---- game ----
  const G = { state: null, human: CZ.WHITE, moves: [], last: null, thinking: null, finished: null, kept: null, busy: false, selected: null, progress: 0 };
  function legalMap() {
    const m = {};
    if (G.finished || G.busy || G.state.side !== G.human) return m;
    for (const mv of G.state.legalMoves()) (m[mv[0]] = m[mv[0]] || []).push(mv[1]);
    return m;
  }
  function play(mv) {
    G.state = G.state.apply(mv);
    G.moves.push(CZ.moveText(mv));
    G.last = mv;
    if (G.state.isOver()) {
      const ending = G.state.ending();
      let result = "draw";
      if (G.state.result() !== 0) result = (1 - G.state.side) === G.human ? "human_win" : "agent_win";
      G.finished = { result, ending };
      record(result, ending);
    }
  }
  function newGame(color) {
    if (G.state && G.moves.length && !G.finished) record("unfinished", "quit");
    G.state = new CZ.GameState(); G.human = color; G.moves = []; G.last = null; G.thinking = null;
    G.finished = null; G.kept = null; G.selected = null;
    $("error").textContent = "";
    render();
    agentTurn();
  }
  async function agentTurn() {
    if (G.finished || G.state.side === G.human) return;
    G.busy = true; G.progress = 0; render();
    const t0 = performance.now();
    let root = null;
    if (G.kept && G.last) {
      const c = G.kept.made.get(G.last[0] * 64 + G.last[1]);
      if (c && c.expanded) { c.parent = null; root = c; }
    }
    G.kept = null;
    const res = await CZ.search(G.state, net, SIMS, CPUCT, root, async (i, n) => {
      G.progress = i / n;
      $("substatus").textContent = `Thinking: ${Math.round(G.progress * 100)}% of ${SIMS} search steps`;
      await pause();
    });
    const choice = CZ.chooseMove(G.state, res.visits);
    const a = choice.action;
    const child = res.root.made.get(a);
    if (child && child.expanded) G.kept = child;
    let total = 0;
    for (const n of choice.counts.values()) total += n;
    const top = [...choice.counts.entries()].filter((e) => e[1] > 0).sort((x, y) => y[1] - x[1] || x[0] - y[0]).slice(0, 3);
    G.thinking = { value: res.root.value, top: top.map(([act, n]) => ({ move: CZ.moveText([act >> 6, act & 63]), share: n / (total || 1) })), secs: (performance.now() - t0) / 1000 };
    G.busy = false;
    play([a >> 6, a & 63]);
    render();
  }
  function onSquare(sq) {
    if (!net || G.busy || G.finished || G.state.side !== G.human) return;
    const legal = legalMap();
    if (G.selected !== null && (legal[G.selected] || []).includes(sq)) {
      const mv = [G.selected, sq];
      G.selected = null;
      play(mv);
      render();
      setTimeout(agentTurn, 30);
      return;
    }
    G.selected = legal[sq] ? (G.selected === sq ? null : sq) : null;
    render();
  }

  // ---- drawing ----
  function render() {
    const white = G.human === CZ.WHITE;
    const rows = white ? [7, 6, 5, 4, 3, 2, 1, 0] : [0, 1, 2, 3, 4, 5, 6, 7];
    const cols = white ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
    $("ranks").innerHTML = rows.map((r) => `<span>${r + 1}</span>`).join("");
    $("files").innerHTML = cols.map((c) => `<span>${FILES[c]}</span>`).join("");
    const legal = legalMap();
    const targets = G.selected !== null ? legal[G.selected] || [] : [];
    const inCheck = !G.finished && G.state.isInCheck();
    const board = $("board");
    board.innerHTML = "";
    for (const r of rows) for (const c of cols) {
      const sq = r * 8 + c, p = G.state.board[sq];
      const el = document.createElement("div");
      el.className = "sq";
      if (G.last && (G.last[0] === sq || G.last[1] === sq)) el.classList.add("last");
      if (sq === G.selected) el.classList.add("sel");
      if (legal[sq]) el.classList.add("mine");
      if (targets.includes(sq)) { el.classList.add("target"); if (p) el.classList.add("capture"); }
      if (p && CZ.typeOf(p) === 0 && CZ.colorOf(p) === G.state.side && inCheck) el.classList.add("check");
      if (p) {
        const s = document.createElement("span");
        const col = CZ.colorOf(p) === 0 ? "white" : "black";
        s.className = "piece " + col;
        s.textContent = GLYPH[CZ.typeOf(p)] + "︎";
        s.title = `${col} ${NAMES[CZ.typeOf(p)]}`;
        el.appendChild(s);
      }
      el.title = CZ.alg(sq);
      el.addEventListener("click", () => onSquare(sq));
      board.appendChild(el);
    }
    const status = $("status"), sub = $("substatus");
    status.classList.toggle("done", !!G.finished);
    if (G.finished) {
      const f = G.finished;
      status.textContent = f.result === "human_win" ? "You win!" : f.result === "agent_win" ? "ChaturangZero wins" : f.result === "draw" ? "Draw" : "Game over";
      sub.textContent = `Game over by ${f.ending}. Start a new game with the buttons below.`;
    } else if (G.busy) {
      status.innerHTML = '<span class="spinner"></span>ChaturangZero is thinking';
      sub.textContent = `Thinking: ${Math.round(G.progress * 100)}% of ${SIMS} search steps`;
    } else {
      const yours = G.state.side === G.human;
      status.textContent = yours ? (inCheck ? "Your move: you are in check" : "Your move") : "Agent to move";
      sub.textContent = `You play ${white ? "White" : "Black"}. ${G.state.clock} of 100 moves without a capture or soldier move.`;
    }
    $("resign").disabled = !!G.finished || G.busy || !G.moves.length;
    $("newWhite").disabled = G.busy || !net;
    $("newBlack").disabled = G.busy || !net;
    const t = G.thinking;
    if (t) {
      $("meter").style.width = Math.round((t.value + 1) * 50) + "%";
      const word = t.value > 0.3 ? "it is winning" : t.value > 0.1 ? "it is slightly better" : t.value < -0.3 ? "it is losing" : t.value < -0.1 ? "you are slightly better" : "the game is even";
      $("thinkText").textContent = `After its last search (${t.secs.toFixed(1)} s), it thinks ${word} (value ${t.value >= 0 ? "+" : ""}${t.value.toFixed(2)} on a scale from -1 to +1).`;
      $("tops").innerHTML = t.top.map((m) => `<span class="chip">${m.move} ${Math.round(m.share * 100)}%</span>`).join("") + '<span class="sub" style="align-self:center">moves it considered most</span>';
    } else {
      $("meter").style.width = "50%"; $("thinkText").textContent = "It has not moved yet."; $("tops").innerHTML = "";
    }
    let html = "";
    for (let i = 0; i < G.moves.length; i += 2) html += `<span class="n">${i / 2 + 1}.</span><span>${G.moves[i]}</span><span>${G.moves[i + 1] || ""}</span>`;
    const box = $("moves");
    box.innerHTML = html || '<span class="sub" style="grid-column:1/-1">No moves yet.</span>';
    box.scrollTop = box.scrollHeight;
  }

  // A second click within a few seconds confirms, since the page cannot show dialogs.
  let pending = null;
  function confirmInline(msg) {
    if (pending && Date.now() - pending < 4000) { pending = null; return true; }
    pending = Date.now(); $("error").textContent = msg + " Click again to confirm."; return false;
  }
  $("newWhite").addEventListener("click", () => { if (G.moves.length && !G.finished && !confirmInline("Start a new game? This one will be recorded as unfinished.")) return; newGame(CZ.WHITE); });
  $("newBlack").addEventListener("click", () => { if (G.moves.length && !G.finished && !confirmInline("Start a new game? This one will be recorded as unfinished.")) return; newGame(CZ.BLACK); });
  $("resign").addEventListener("click", () => {
    if (G.busy || G.finished || !G.moves.length) return;
    if (!confirmInline("Resign this game?")) return;
    $("error").textContent = "";
    G.finished = { result: "agent_win", ending: "resigned" };
    record("agent_win", "resigned");
    render();
  });
  $("copy").addEventListener("click", () => {
    const ta = $("codes");
    navigator.clipboard.writeText(ta.value).then(() => { $("copy").textContent = "Copied"; }).catch(() => { ta.focus(); ta.select(); $("copy").textContent = "Selected: press Ctrl+C"; });
  });
  $("player").value = store.get("cz-player", "");
  $("player").addEventListener("input", () => store.set("cz-player", $("player").value));

  G.state = new CZ.GameState();
  render();
  tally();
  initNet().then(() => {
    $("about").textContent = `You are playing the final ChaturangZero model (round 1220), which searches ${SIMS} positions before each move. It learned only from games against itself.`;
    $("status").textContent = "Ready: start a game";
    $("substatus").textContent = "Play 5 games as White and 5 as Black if you can.";
    $("newWhite").disabled = false; $("newBlack").disabled = false;
  }).catch((e) => {
    $("status").textContent = "The network could not load";
    $("error").textContent = "Try reloading the page, or open it in Chrome or Edge. (" + e.message + ")";
  });
})();
