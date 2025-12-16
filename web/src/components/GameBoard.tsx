import { useTheme } from '../context/ThemeContext';

interface Props {
  board: string;
  disabled?: boolean;
  onPick: (index: number) => void;
}

export function GameBoard({ board, disabled, onPick }: Props) {
  const { config } = useTheme();
  
  return (
    <div className="grid grid-cols-3 gap-3 w-full max-w-md mx-auto">
      {board.split('').map((mark, idx) => (
        <button
          key={idx}
          disabled={disabled || mark !== '_'}
          onClick={() => onPick(idx)}
          className={`aspect-square flex items-center justify-center text-4xl md:text-5xl font-semibold rounded-xl border transition-all duration-200 transform ${
            mark === '_' 
              ? `${config.button} hover:scale-110 hover:shadow-lg cursor-pointer` 
              : `${config.accent} shadow-lg`
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {mark === '_' ? '' : mark}
        </button>
      ))}
    </div>
  );
}
