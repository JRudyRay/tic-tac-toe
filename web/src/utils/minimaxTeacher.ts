import { computeWinner, isMoveValid, nextSymbol } from './gameLogic';

export type PlayerSymbol = 'X' | 'O';

export interface TeacherState {
  board: string;
  turn: PlayerSymbol;
  policy: number[]; // length 9, sums to 1 over legal moves
  mask: number[]; // length 9, 1 for legal, 0 for filled
  value: number; // expected outcome in [-1,1] for player-to-move
}

const SCORE_WIN = 10;

function applyMove(board: string, index: number, symbol: PlayerSymbol): string {
  return board.substring(0, index) + symbol + board.substring(index + 1);
}

function boardMask(board: string): number[] {
  return board.split('').map((c) => (c === '_' ? 1 : 0));
}

function terminalScore(board: string, depth: number): number | null {
  const winner = computeWinner(board);
  if (winner) {
    // If board is terminal, the side to move has lost (opponent just made winning move).
    return -(SCORE_WIN - depth);
  }
  if (!board.includes('_')) return 0;
  return null;
}

const negamaxCache = new Map<string, number>();

function negamax(board: string, playerToMove: PlayerSymbol, depth: number): number {
  const key = `${board}|${playerToMove}`;
  const cached = negamaxCache.get(key);
  if (cached !== undefined) return cached;

  const terminal = terminalScore(board, depth);
  if (terminal !== null) {
    negamaxCache.set(key, terminal);
    return terminal;
  }

  let best = -Infinity;
  for (let i = 0; i < 9; i++) {
    if (!isMoveValid(board, i)) continue;
    const nextBoard = applyMove(board, i, playerToMove);
    const score = -negamax(nextBoard, nextSymbol(playerToMove), depth + 1);
    if (score > best) best = score;
  }

  negamaxCache.set(key, best);
  return best;
}

const policyCache = new Map<string, { policy: number[]; value: number }>();

/**
 * Returns an optimal-policy distribution over moves for the player-to-move.
 * - Splits probability evenly among all optimal moves (ties).
 * - Uses depth-aware scoring to prefer quicker wins and slower losses.
 * - Provides a scalar value in [-1,1] for the player-to-move.
 */
export function minimaxPolicy(board: string, turn: PlayerSymbol): { policy: number[]; mask: number[]; value: number } {
  const key = `${board}|${turn}`;
  const cached = policyCache.get(key);
  if (cached) {
    return { policy: cached.policy.slice(), mask: boardMask(board), value: cached.value };
  }

  const mask = boardMask(board);
  const winner = computeWinner(board);
  if (winner || !board.includes('_')) {
    const zeros = Array(9).fill(0);
    policyCache.set(key, { policy: zeros, value: 0 });
    return { policy: zeros.slice(), mask, value: 0 };
  }

  let bestScore = -Infinity;
  const scores = Array(9).fill(-Infinity);

  for (let i = 0; i < 9; i++) {
    if (mask[i] === 0) continue;
    const nextBoard = applyMove(board, i, turn);
    const score = -negamax(nextBoard, nextSymbol(turn), 1);
    scores[i] = score;
    if (score > bestScore) bestScore = score;
  }

  const bestMoves = scores.map((s, i) => ({ s, i })).filter((x) => mask[x.i] === 1 && x.s === bestScore);
  const policy = Array(9).fill(0);
  const p = bestMoves.length > 0 ? 1 / bestMoves.length : 0;
  for (const m of bestMoves) policy[m.i] = p;

  const boundedValue = Math.max(-1, Math.min(1, bestScore / SCORE_WIN));
  policyCache.set(key, { policy, value: boundedValue });
  return { policy: policy.slice(), mask, value: boundedValue };
}

const TRANSFORMS: Array<{ name: string; map: number[] }> = [
  { name: 'id', map: [0, 1, 2, 3, 4, 5, 6, 7, 8] },
  { name: 'rot90', map: [2, 5, 8, 1, 4, 7, 0, 3, 6] },
  { name: 'rot180', map: [8, 7, 6, 5, 4, 3, 2, 1, 0] },
  { name: 'rot270', map: [6, 3, 0, 7, 4, 1, 8, 5, 2] },
  { name: 'flipTB', map: [6, 7, 8, 3, 4, 5, 0, 1, 2] },
  { name: 'flipLR', map: [2, 1, 0, 5, 4, 3, 8, 7, 6] },
  { name: 'diag', map: [0, 3, 6, 1, 4, 7, 2, 5, 8] },
  { name: 'anti', map: [8, 5, 2, 7, 4, 1, 6, 3, 0] },
];

function transformBoard(board: string, map: number[]): string {
  const out = Array(9).fill('_');
  for (let i = 0; i < 9; i++) {
    out[map[i]] = board[i];
  }
  return out.join('');
}

function transformDist(dist: number[], map: number[]): number[] {
  const out = Array(9).fill(0);
  for (let i = 0; i < 9; i++) {
    out[map[i]] += dist[i] ?? 0;
  }
  return out;
}

export function generateTeacherDataset(options?: { augmentSymmetries?: boolean }): TeacherState[] {
  const augmentSymmetries = options?.augmentSymmetries ?? true;

  const seen = new Set<string>();
  const states: Array<{ board: string; turn: PlayerSymbol }> = [];

  const walk = (board: string, turn: PlayerSymbol) => {
    const key = `${board}|${turn}`;
    if (seen.has(key)) return;
    seen.add(key);

    const winner = computeWinner(board);
    if (winner || !board.includes('_')) return;

    states.push({ board, turn });

    for (let i = 0; i < 9; i++) {
      if (!isMoveValid(board, i)) continue;
      walk(applyMove(board, i, turn), nextSymbol(turn));
    }
  };

  walk('_________', 'X');

  const out: TeacherState[] = [];
  const dedupe = new Set<string>();

  for (const s of states) {
    const { policy, mask, value } = minimaxPolicy(s.board, s.turn);

    if (!augmentSymmetries) {
      const k = `${s.board}|${s.turn}`;
      if (!dedupe.has(k)) {
        dedupe.add(k);
        out.push({ board: s.board, turn: s.turn, policy, mask, value });
      }
      continue;
    }

    for (const t of TRANSFORMS) {
      const b2 = transformBoard(s.board, t.map);
      const p2 = transformDist(policy, t.map);
      const m2 = boardMask(b2);
      const k = `${b2}|${s.turn}`;
      if (dedupe.has(k)) continue;
      dedupe.add(k);
      out.push({ board: b2, turn: s.turn, policy: p2, mask: m2, value });
    }
  }

  return out;
}
