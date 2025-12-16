import { useMemo, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { GameBoard } from '../components/GameBoard';
import { aiMove } from '../utils/ai';
import { computeWinner, isMoveValid } from '../utils/gameLogic';
import { useTheme } from '../context/ThemeContext';

export default function SinglePlayerPage() {
  const navigate = useNavigate();
  const { config } = useTheme();
  const [difficulty, setDifficulty] = useState<'EASY' | 'HARD'>('EASY');
  const [board, setBoard] = useState('_________');
  const [turn, setTurn] = useState<'X' | 'O'>('X');
  const [status, setStatus] = useState<'ACTIVE' | 'FINISHED'>('ACTIVE');
  const [moves, setMoves] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (status === 'FINISHED') return;
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(timer);
  }, [status]);

  const winner = useMemo(() => computeWinner(board), [board]);
  const isMyTurn = useMemo(() => status === 'ACTIVE' && turn === 'X', [status, turn]);

  const reset = () => {
    setBoard('_________');
    setTurn('X');
    setStatus('ACTIVE');
    setMoves(0);
    setElapsed(0);
  };

  const statusText = useMemo(() => {
    if (status === 'FINISHED') {
      if (!winner) return '🤝 Draw!';
      return winner === 'X' ? '🎉 You win!' : '🤖 AI wins!';
    }
    return isMyTurn ? '👤 Your turn' : '⏳ AI thinking...';
  }, [status, winner, isMyTurn]);

  const handleHumanMove = (index: number) => {
    if (!isMyTurn || !isMoveValid(board, index)) return;
    const b1 = board.substring(0, index) + 'X' + board.substring(index + 1);
    const w1 = computeWinner(b1);
    const full1 = !b1.includes('_');
    
    setBoard(b1);
    setMoves((m) => m + 1);

    if (w1 || full1) {
      setTurn('X');
      setStatus('FINISHED');
      return;
    }

    // Switch to AI turn immediately
    setTurn('O');

    // Simulate AI thinking delay
    setTimeout(() => {
      const aiIndex = aiMove(b1, 'O', difficulty);
      const b2 = aiIndex !== null ? b1.substring(0, aiIndex) + 'O' + b1.substring(aiIndex + 1) : b1;
      setBoard(b2);
      const w2 = computeWinner(b2);
      const full2 = !b2.includes('_');
      if (w2 || full2) {
        setStatus('FINISHED');
      } else {
        setTurn('X');
      }
      setMoves((m) => m + 1);
    }, 500);
  };

  return (
    <div className={`min-h-screen ${config.bg} p-4 md:p-8 transition-all duration-500 flex flex-col items-center justify-center`}>
      <div className={`glass-card p-6 md:p-8 max-w-md w-full border ${config.card}`}>
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className={`text-3xl font-bold ${config.text}`}>Tic-Tac-Toe</h1>
            <p className={`${config.text}/70 text-sm`}>{difficulty === 'EASY' ? '🟢 Easy' : '🔴 Hard'}</p>
          </div>
          <button onClick={() => navigate('/')} className={`px-3 py-2 rounded-lg ${config.button} transition hover:scale-105`}>
            ← Home
          </button>
        </div>

        <div className="grid grid-cols-3 gap-2 mb-6 text-center text-sm">
          <div className={`${config.text}/70`}>
            <div className="font-bold">Moves</div>
            <div className={`text-2xl font-bold ${config.text}`}>{moves}</div>
          </div>
          <div className={`${config.text}/70`}>
            <div className="font-bold">Time</div>
            <div className={`text-2xl font-bold ${config.text}`}>{Math.floor(elapsed / 60)}:{(elapsed % 60).toString().padStart(2, '0')}</div>
          </div>
          <div className={`${config.text}/70`}>
            <div className="font-bold">Turn</div>
            <div className={`text-2xl font-bold ${config.text}`}>{turn}</div>
          </div>
        </div>

        <div className="flex items-center gap-3 mb-6 flex-wrap">
          <span className={`${config.text}/80 text-sm font-semibold`}>Difficulty:</span>
          <button
            onClick={() => {
              setDifficulty('EASY');
              reset();
            }}
            className={`px-3 py-1 rounded-lg font-semibold transition-all ${difficulty === 'EASY' ? `${config.accent} scale-105` : config.button}`}
          >
            Easy
          </button>
          <button
            onClick={() => {
              setDifficulty('HARD');
              reset();
            }}
            className={`px-3 py-1 rounded-lg font-semibold transition-all ${difficulty === 'HARD' ? `${config.accent} scale-105` : config.button}`}
          >
            Hard
          </button>
          <button onClick={reset} className={`ml-auto px-3 py-1 rounded-lg ${config.button} transition hover:scale-105`}>
            Reset
          </button>
        </div>

        <GameBoard board={board} disabled={status === 'FINISHED' || !isMyTurn} onPick={handleHumanMove} />
        <p className={`text-center mt-6 text-xl font-bold ${config.text} animate-pulse`}>{statusText}</p>
      </div>
    </div>
  );
}
