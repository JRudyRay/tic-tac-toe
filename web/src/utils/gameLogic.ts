const WINNING_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
  [0, 4, 8], [2, 4, 6]             // diagonals
];

export function computeWinner(board: string): 'X' | 'O' | null {
  for (const [a, b, c] of WINNING_LINES) {
    if (board[a] !== '_' && board[a] === board[b] && board[b] === board[c]) {
      return board[a] as 'X' | 'O';
    }
  }
  return null;
}

export function isMoveValid(board: string, index: number) {
  return board[index] === '_';
}

export function nextSymbol(s: 'X' | 'O'): 'X' | 'O' {
  return s === 'X' ? 'O' : 'X';
}
