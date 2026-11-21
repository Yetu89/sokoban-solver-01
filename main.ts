// Sokoban push-optimal solver — A* over pushes, room by room, with a BFS fallback.
//
// Strategy (deterministic, work-budgeted, no clock / no randomness / no env):
//  1. Split the level into rooms: cut the hallway row's dead squares (squares from
//     which a box can never reach a goal) and take the connected components that
//     hold boxes. A box pushed into the hallway can never come back, so every box
//     stays in its room and the level optimum P* is the sum of the rooms' optima.
//  2. Solve each room with A* over pushes. A state is the sorted box squares plus
//     the player's reachable region (named by its lowest square). The heuristic is
//     the Hungarian (minimum-assignment) cost of box->goal BFS distances ignoring
//     other boxes: admissible, hence the first goal state popped is push-optimal.
//     Deadlock pruning: dead squares (pull-back from goals), frozen 2x2 blocks,
//     and boxes frozen against walls. Pushes are generated in fixed u/d/l/r order
//     and ties break by insertion order, so the search is fully deterministic.
//  3. If A* exhausts its expansion budget on a room, fall back to breadth-first
//     search over pushes (same state space, same pruning); BFS is optimal too and
//     the generator only keeps rooms this BFS solves, so the fallback is a safety
//     net, not the plan. If room decomposition fails, the same searches run on the
//     whole level (with a sum-of-minima heuristic, since Hungarian is only run for
//     small box counts).
//  4. Moves are reconstructed push by push: shortest walk to the push square
//     (BFS, fixed direction order) + the push itself.
//
// Budgets are counts of expanded states, never wall-clock time, so two runs on
// the same set write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (counts, not time): generous vs. what the practice set needs.
const ASTAR_EXPANSIONS = 300_000; // per room (or whole level)
const BFS_STATES = 2_000_000; // per room (or whole level), fallback only

const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  boxes: number[]; // sorted
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pulled back onto some goal. */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length || y - d < 0 || y - d >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood-fill from `from` with `occupied` in the way, marking `seen` with `stamp`;
 * returns the lowest square reached (the region's canonical name).
 */
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block (walls/boxes, a box off-goal). */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * Whether a box at `b` (off its goal) can never move again because of walls alone:
 * no vertical push possible (a wall above or below) AND no horizontal push
 * possible (a wall left or right). Walls never move, so this is sound.
 */
function wallFrozen(level: Level, b: number): boolean {
  const { w, wall } = level;
  const up = wall[b - w] === 1;
  const down = wall[b + w] === 1;
  const left = wall[b - 1] === 1;
  const right = wall[b + 1] === 1;
  return (up || down) && (left || right);
}

/** BFS distances (ignoring boxes) from each goal to every square; Infinity if unreachable. */
function goalDistances(level: Level): number[][] {
  const { w, wall, goals } = level;
  const dists: number[][] = [];
  for (const g of goals) {
    const dist = new Array<number>(wall.length).fill(Infinity);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (q < 0 || q >= wall.length || wall[q] || dist[q] !== Infinity) continue;
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
    dists.push(dist);
  }
  return dists;
}

/**
 * Minimum-assignment (Hungarian by brute force over permutations, n <= 5) cost of
 * matching boxes to goals on the distance matrix; Infinity when some box reaches
 * no goal. Admissible: every push moves one box one square, so the true remaining
 * push count is at least this.
 */
function hungarian(boxes: number[], dists: number[][], goals: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    // Too many boxes for permutations: fall back to the (still admissible)
    // sum of each box's nearest-goal distance.
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < goals.length; g++) m = Math.min(m, dists[g]![b]!);
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  }
  // dists are per-goal arrays; build cost[box][goal].
  const perm = goals.map((_, i) => i);
  let best = Infinity;
  const rec = (k: number, acc: number): void => {
    if (acc >= best) return;
    if (k === n) {
      best = acc;
      return;
    }
    for (let i = k; i < perm.length; i++) {
      [perm[k], perm[i]] = [perm[i]!, perm[k]!];
      const c = dists[perm[k]!]![boxes[k]!]!;
      if (c !== Infinity) rec(k + 1, acc + c);
      [perm[k], perm[i]] = [perm[i]!, perm[k]!];
    }
  };
  rec(0, 0);
  return best;
}

// Numeric state key: region's lowest square, then the sorted box squares, base `size`.
function keyOf(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

interface SearchCtx {
  level: Level;
  size: number;
  goal: Uint8Array;
  live: Uint8Array;
  dists: number[][];
  occupied: Uint8Array;
  seen: Int32Array;
  mark: Int32Array;
  stack: Int32Array;
  stamp: number;
}

function makeCtx(level: Level): SearchCtx {
  const size = level.wall.length;
  const goal = new Uint8Array(size);
  for (const g of level.goals) goal[g] = 1;
  return {
    level,
    size,
    goal,
    live: liveSquares(level),
    dists: goalDistances(level),
    occupied: new Uint8Array(size),
    seen: new Int32Array(size),
    mark: new Int32Array(size),
    stack: new Int32Array(size),
    stamp: 0,
  };
}

const solvedBoxes = (ctx: SearchCtx, boxes: number[]): boolean => boxes.every((b) => ctx.goal[b]);

/** Reconstruct the move string from the stored state chain. */
function reconstruct(
  ctx: SearchCtx,
  statesBoxes: number[][],
  statesPlayer: number[],
  parent: number[],
  found: number,
  startBoxes: number[],
  startPlayer: number,
): string {
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * ctx.level.w + dx === d)!;
    for (const b of boxes) ctx.occupied[b] = 1;
    moves += walkTo(ctx.level, ctx.occupied, player, from - d);
    for (const b of boxes) ctx.occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower LURD (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const { w, wall } = level;
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= wall.length || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Generate the pushes out of a state: [nextBoxes, playerAfterPush, childKey]. */
function* genPushes(
  ctx: SearchCtx,
  boxes: number[],
  player: number,
): Generator<{ next: number[]; player: number; key: number }> {
  const { level, occupied, seen } = ctx;
  const { w, wall } = level;
  for (const b of boxes) occupied[b] = 1;
  const stamp = ++ctx.stamp;
  reach(level, occupied, player, seen, stamp, ctx.stack);
  const here = stamp;
  for (let bi = 0; bi < boxes.length; bi++) {
    const b = boxes[bi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const t = b + d;
      if (t < 0 || t >= wall.length || b - d < 0 || b - d >= wall.length) continue;
      if (seen[b - d] !== here || wall[t] || occupied[t] || !ctx.live[t]) continue;
      if (!ctx.goal[t] && wallFrozen(level, t)) continue;
      occupied[b] = 0;
      occupied[t] = 1;
      const frozen = !ctx.goal[t] && frozenBlock(level, occupied, ctx.goal, t);
      let next: number[] | null = null;
      let key = -1;
      if (!frozen) {
        next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        // Key the child NOW, while `occupied` still holds the child's boxes:
        // the region the player stands in after the push.
        key = keyOf(next, reach(level, occupied, b, ctx.mark, ++ctx.stamp, ctx.stack), ctx.size);
      }
      occupied[t] = 0;
      occupied[b] = 1;
      if (next === null) continue;
      yield { next, player: b, key };
    }
  }
  for (const b of boxes) occupied[b] = 0;
}

/**
 * A* over pushes with the Hungarian heuristic. Returns {pushes, moves} or null
 * when the expansion budget runs out (or the level is unsolvable).
 */
function solveAStar(level: Level, maxExpansions: number): { pushes: number; moves: string } | null {
  const ctx = makeCtx(level);
  if (level.boxes.some((b) => !ctx.live[b])) return null;
  if (solvedBoxes(ctx, level.boxes)) return { pushes: 0, moves: '' };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const gScore: number[] = [0];
  const best = new Map<number, number>();
  for (const b of level.boxes) ctx.occupied[b] = 1;
  const low0 = reach(level, ctx.occupied, level.player, ctx.seen, ++ctx.stamp, ctx.stack);
  for (const b of level.boxes) ctx.occupied[b] = 0;
  best.set(keyOf(level.boxes, low0, ctx.size), 0);

  // Binary heap on [f, -g, seq, stateIndex]: lowest f first, deeper g first on ties.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = hungarian(level.boxes, ctx.dists, level.goals);
  if (h0 === Infinity) return null;
  const heapLess = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
        if (!heapLess(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  heapPush([h0, 0, seq++, 0]);

  let expanded = 0;
  while (heap.length > 0) {
    const [, negG, , s] = heapPop();
    const g = -negG;
    if (gScore[s] !== g) continue; // stale entry
    const boxes = statesBoxes[s]!;
    if (solvedBoxes(ctx, boxes)) {
      return { pushes: g, moves: reconstruct(ctx, statesBoxes, statesPlayer, parent, s, level.boxes, level.player) };
    }
    if (expanded >= maxExpansions) return null;
    expanded++;
    for (const { next, player, key } of genPushes(ctx, boxes, statesPlayer[s]!)) {
      const ng = g + 1;
      const prev = best.get(key);
      if (prev !== undefined && prev <= ng) continue;
      const h = hungarian(next, ctx.dists, level.goals);
      if (h === Infinity) continue;
      const t = statesBoxes.length;
      statesBoxes.push(next);
      statesPlayer.push(player);
      parent.push(s);
      gScore.push(ng);
      best.set(key, ng);
      heapPush([ng + h, -ng, seq++, t]);
    }
  }
  return null;
}

/**
 * Breadth-first search over pushes (fallback). Expands states in push order, so
 * the first solved state met is push-optimal. Same pruning as A*.
 */
function solveBFS(level: Level, maxStates: number): { pushes: number; moves: string } | null {
  const ctx = makeCtx(level);
  if (level.boxes.some((b) => !ctx.live[b])) return null;
  if (solvedBoxes(ctx, level.boxes)) return { pushes: 0, moves: '' };

  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of level.boxes) ctx.occupied[b] = 1;
  index.set(keyOf(level.boxes, reach(level, ctx.occupied, level.player, ctx.seen, ++ctx.stamp, ctx.stack), ctx.size), 0);
  for (const b of level.boxes) ctx.occupied[b] = 0;
  let found = -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const { next, player, key } of genPushes(ctx, boxes, statesPlayer[head]!)) {
      if (index.has(key)) continue;
      if (statesBoxes.length >= maxStates) return null;
      index.set(key, statesBoxes.length);
      statesBoxes.push(next);
      statesPlayer.push(player);
      parent.push(head);
      if (solvedBoxes(ctx, next)) {
        found = statesBoxes.length - 1;
        break;
      }
    }
  }
  if (found < 0) return null;
  return { pushes: found === 0 ? 0 : pathLen(parent, found), moves: reconstruct(ctx, statesBoxes, statesPlayer, parent, found, level.boxes, level.player) };
}

function pathLen(parent: number[], s: number): number {
  let n = 0;
  for (let t = s; t > 0; t = parent[t]!) n++;
  return n;
}

/** Solve one (sub-)level: A* first, BFS fallback. */
function solveOne(level: Level): { pushes: number; moves: string } | null {
  return solveAStar(level, ASTAR_EXPANSIONS) ?? solveBFS(level, BFS_STATES);
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway) out
 * of the floor; each remaining part holding boxes is a room. Null when a box
 * starts in the hallway or some part's boxes/goals don't match.
 */
function roomsOf(level: Level): { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve a level room by room; fall back to solving the whole level at once. */
function solveLevel(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return solveOne(level);
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms)
      if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solveOne(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

function solve(text: string): string {
  try {
    const r = solveLevel(parseLevel(text));
    return r === null ? '' : r.moves;
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
