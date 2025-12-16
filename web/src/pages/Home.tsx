import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';

export default function Home() {
  const navigate = useNavigate();
  const { config } = useTheme();

  return (
    <div className={`min-h-screen ${config.bg} flex flex-col items-center justify-center p-4 transition-all duration-500`}>
      <div className={`glass-card p-8 max-w-md w-full text-center border ${config.card}`}>
        <h1 className={`text-5xl font-bold ${config.text} mb-4`}>Tic-Tac-Toe</h1>
        <p className={`${config.text}/80 mb-8 text-lg`}>Challenge the AI</p>

        <div className="space-y-3">
          <button
            onClick={() => navigate('/single')}
            className={`w-full py-3 rounded-lg font-bold text-lg ${config.accent} hover:scale-105 hover:shadow-lg transition-all duration-200`}
          >
            ▶ Play Now
          </button>
          <button
            onClick={() => navigate('/train-setup')}
            className={`w-full py-3 rounded-lg font-bold text-lg ${config.accent} hover:scale-105 hover:shadow-lg transition-all duration-200`}
          >
            🧠 Train Your Own AI
          </button>
        </div>
      </div>
    </div>
  );
}
