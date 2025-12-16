import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { NeuralNetwork, type NetworkConfig } from '../utils/neuralNetwork';

export default function TrainSetup() {
  const navigate = useNavigate();
  const { config } = useTheme();
  const [networkName, setNetworkName] = useState('MyAI');
  const [networkSize, setNetworkSize] = useState<'small' | 'medium' | 'large'>('medium');
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Optimized defaults for novices - stable learning with good convergence
  const [learningRate, setLearningRate] = useState(0.015); // Slightly slower but more stable
  const [dropout, setDropout] = useState(0.10); // Lower dropout for simpler task
  const [weightDecay, setWeightDecay] = useState(0.0001); // Light regularization
  const [isLoading, setIsLoading] = useState(false);

  // Preset configurations - optimized for tic-tac-toe
  const presets = {
    small: { layers: [32], description: 'Fast training, good for quick demos', icon: '🚀' },
    medium: { layers: [64, 32], description: 'Balanced - recommended for most users', icon: '⚖️' },
    large: { layers: [128, 64, 32], description: 'Higher capacity, takes longer to train', icon: '🧠' },
  };

  const handleStart = () => {
    setIsLoading(true);

    if (!networkName.trim()) {
      alert('Please enter a name for your AI');
      setIsLoading(false);
      return;
    }

    const preset = presets[networkSize];
    const netConfig: NetworkConfig = {
      inputSize: 10, // 9 board cells + 1 turn indicator
      hiddenLayers: preset.layers,
      outputSize: 9, // 9 possible moves
      learningRate,
      weightDecay,
      dropout,
      valueLossWeight: 0.5, // Fixed optimal value
      gradientClip: 5.0, // Prevent training instability
    };

    try {
      const network = new NeuralNetwork(netConfig);
      const snapshot = network.getSnapshot();

      sessionStorage.setItem('trainingSnapshot', JSON.stringify(snapshot));
      sessionStorage.setItem('trainingNetworkName', networkName);

      navigate('/train');
    } catch (e) {
      console.error('Error creating network:', e);
      alert('Error: ' + (e as Error).message);
      setIsLoading(false);
    }
  };

  return (
    <div className={`min-h-screen ${config.bg} p-4 md:p-8 transition-all duration-500 flex flex-col items-center justify-center`}>
      <div className={`glass-card p-8 max-w-lg w-full border ${config.card}`}>
        <h1 className={`text-4xl font-bold ${config.text} mb-2`}>🧠 Train Your AI</h1>
        <p className={`${config.text}/70 mb-8 text-base`}>
          Create a neural network that learns to play Tic-Tac-Toe!
        </p>

        <div className="space-y-6">
          {/* Network Name */}
          <div>
            <label className={`block ${config.text} font-semibold mb-2 text-sm`}>Name Your AI</label>
            <input
              type="text"
              value={networkName}
              onChange={(e) => setNetworkName(e.target.value)}
              className={`w-full px-4 py-3 rounded-lg ${config.button} text-base focus:outline-none focus:ring-2 focus:ring-blue-500`}
              placeholder="e.g., Champion, SmartBot, MyFirstAI"
            />
          </div>

          {/* Network Size - Simple Preset Selection */}
          <div>
            <label className={`block ${config.text} font-semibold mb-3 text-sm`}>Brain Size</label>
            <div className="grid grid-cols-3 gap-3">
              {(Object.keys(presets) as Array<keyof typeof presets>).map((size) => (
                <button
                  key={size}
                  onClick={() => setNetworkSize(size)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    networkSize === size
                      ? `border-blue-500 ${config.accent}`
                      : `border-transparent ${config.button} hover:border-blue-500/50`
                  }`}
                >
                  <div className="text-2xl mb-1">{presets[size].icon}</div>
                  <div className={`font-semibold ${config.text} capitalize`}>{size}</div>
                  <div className={`text-xs ${config.text}/60 mt-1`}>
                    {presets[size].layers.length} layer{presets[size].layers.length > 1 ? 's' : ''}
                  </div>
                </button>
              ))}
            </div>
            <p className={`${config.text}/60 text-xs mt-2 text-center`}>
              {presets[networkSize].description}
            </p>
          </div>

          {/* How It Works - Educational */}
          <div className={`p-4 rounded-lg ${config.button} border border-white/10`}>
            <h3 className={`${config.text} font-semibold text-sm mb-2`}>💡 How It Works</h3>
            <ol className={`${config.text}/70 text-xs space-y-1 list-decimal list-inside`}>
              <li>Your AI watches perfect games played by Minimax</li>
              <li>It learns which moves lead to wins vs losses</li>
              <li>You can play against it anytime to test progress!</li>
            </ol>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`w-full text-left text-sm ${config.text}/60 hover:${config.text}/80 transition`}
          >
            {showAdvanced ? '▼' : '▶'} Advanced Settings
          </button>

          {showAdvanced && (
            <div className={`p-4 rounded-lg ${config.button} space-y-4`}>
              {/* Learning Rate */}
              <div>
                <label className={`block ${config.text} font-semibold mb-2 text-sm`}>
                  Learning Rate: <span className="text-blue-400">{learningRate.toFixed(3)}</span>
                </label>
                <input
                  type="range"
                  min="0.005"
                  max="0.1"
                  step="0.005"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className={`${config.text}/60 text-xs mt-1`}>
                  How fast the network learns. Higher = faster but less stable.
                </p>
              </div>

              {/* Dropout */}
              <div>
                <label className={`block ${config.text} font-semibold mb-2 text-sm`}>
                  Dropout: <span className="text-blue-400">{(dropout * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.05"
                  value={dropout}
                  onChange={(e) => setDropout(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className={`${config.text}/60 text-xs mt-1`}>
                  Randomly disables neurons during training. Prevents overfitting.
                </p>
              </div>

              {/* Weight Decay */}
              <div>
                <label className={`block ${config.text} font-semibold mb-2 text-sm`}>
                  Weight Decay: <span className="text-blue-400">{weightDecay.toFixed(4)}</span>
                </label>
                <input
                  type="range"
                  min="0"
                  max="0.01"
                  step="0.0005"
                  value={weightDecay}
                  onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className={`${config.text}/60 text-xs mt-1`}>
                  L2 regularization. Keeps weights small to prevent overfitting.
                </p>
              </div>

              <div className={`pt-2 border-t border-white/10`}>
                <p className={`${config.text}/50 text-xs`}>
                  Architecture: 10 → {presets[networkSize].layers.join(' → ')} → 9
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="flex gap-3 mt-8">
          <button
            onClick={() => navigate('/')}
            className={`flex-1 py-3 rounded-lg ${config.button} transition hover:scale-105 font-semibold`}
            disabled={isLoading}
          >
            ← Back
          </button>
          <button
            onClick={handleStart}
            disabled={isLoading}
            className={`flex-1 py-3 rounded-lg font-bold text-lg ${isLoading ? 'opacity-50 cursor-not-allowed' : config.accent} transition hover:scale-105`}
          >
            {isLoading ? 'Creating...' : 'Start Training →'}
          </button>
        </div>
      </div>
    </div>
  );
}
