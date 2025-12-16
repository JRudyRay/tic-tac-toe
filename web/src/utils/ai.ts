import { computeWinner, isMoveValid, nextSymbol } from './gameLogic';

export function aiMove(board: string, aiSymbol: 'X' | 'O', difficulty: 'EASY' | 'HARD'): number | null {
  if (difficulty === 'EASY') {
    const options = [...board].map((c, i) => (c === '_' ? i : -1)).filter((i) => i >= 0);
    if (options.length === 0) return null;
    return options[Math.floor(Math.random() * options.length)];
  }
  const best = minimax(board, aiSymbol, aiSymbol);
  return best.index;
}

function minimax(board: string, player: 'X' | 'O', ai: 'X' | 'O'): { score: number; index: number | null } {
  const winner = computeWinner(board);
  if (winner) return { score: winner === ai ? 10 : -10, index: null };
  if (!board.includes('_')) return { score: 0, index: null };

  let bestScore = -Infinity;
  let bestIndex: number | null = null;
  for (let i = 0; i < 9; i++) {
    if (!isMoveValid(board, i)) continue;
    const nextBoard = board.substring(0, i) + player + board.substring(i + 1);
    const nextPlayer = nextSymbol(player);
    const result = minimaxOpponent(nextBoard, nextPlayer, ai);
    if (result.score > bestScore) {
      bestScore = result.score;
      bestIndex = i;
    }
  }
  return { score: bestScore, index: bestIndex };
}

function minimaxOpponent(board: string, player: 'X' | 'O', ai: 'X' | 'O'): { score: number; index: number | null } {
  const winner = computeWinner(board);
  if (winner) return { score: winner === ai ? 10 : -10, index: null };
  if (!board.includes('_')) return { score: 0, index: null };

  let bestScore = Infinity;
  let bestIndex: number | null = null;
  for (let i = 0; i < 9; i++) {
    if (!isMoveValid(board, i)) continue;
    const nextBoard = board.substring(0, i) + player + board.substring(i + 1);
    const nextPlayer = nextSymbol(player);
    const result = minimax(nextBoard, nextPlayer, ai);
    if (result.score < bestScore) {
      bestScore = result.score;
      bestIndex = i;
    }
  }
  return { score: bestScore, index: bestIndex };
}
