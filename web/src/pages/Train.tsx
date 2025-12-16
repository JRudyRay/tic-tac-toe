import { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { GameBoard } from '../components/GameBoard';
import { NetworkVisualization } from '../components/NetworkVisualization';
import { computeWinner, isMoveValid } from '../utils/gameLogic';
import { useTheme } from '../context/ThemeContext';
import { NeuralNetwork, type NetworkSnapshot, type TrainingExample } from '../utils/neuralNetwork';
import { generateTeacherDataset, minimaxPolicy } from '../utils/minimaxTeacher';

interface GameRecord {
  moves: Array<{ board: string; move: number; turn: 'X' | 'O' }>;
  winner: string | null;
  outcome: number; // 1 for AI win, 0 for draw, -1 for player win
}

interface TestStats {
  vsMinimaxWins: number;
  vsMinimaxLosses: number;
  vsMinimaxDraws: number;
  vsRandomWins: number;
  vsRandomLosses: number;
  vsRandomDraws: number;
  vsEasyWins: number;
  vsEasyLosses: number;
  vsEasyDraws: number;
}

type AppMode = 'train' | 'test';
type TestOpponent = 'you' | 'minimax' | 'random' | 'easy';

export default function TrainPage() {
  const navigate = useNavigate();
  const { config } = useTheme();
  const [network, setNetwork] = useState<NeuralNetwork | null>(null);
  const [networkName, setNetworkName] = useState('');
  const [board, setBoard] = useState('_________');
  const [turn, setTurn] = useState<'X' | 'O'>('X');
  const [status, setStatus] = useState<'ACTIVE' | 'FINISHED'>('ACTIVE');
  const [gameCount, setGameCount] = useState(0);
  const [losses, setLosses] = useState<number[]>([]);
  const [gameRecord, setGameRecord] = useState<GameRecord | null>(null);
  const [animating, setAnimating] = useState(false);
  const [layerDeltas, setLayerDeltas] = useState<Array<{ weights: number[][]; biases: number[] }>>([]);
  const [initError, setInitError] = useState<string>('');
  const aiMoveTimeout = useRef<number | null>(null);
  const sessionRef = useRef(0);
  const [simulatedGames, setSimulatedGames] = useState(5);
  const [simulationMessage, setSimulationMessage] = useState('');
  const [baseLearningRate, setBaseLearningRate] = useState<number | null>(null);
  const [testStats, setTestStats] = useState<TestStats>({ 
    vsMinimaxWins: 0, vsMinimaxLosses: 0, vsMinimaxDraws: 0,
    vsRandomWins: 0, vsRandomLosses: 0, vsRandomDraws: 0,
    vsEasyWins: 0, vsEasyLosses: 0, vsEasyDraws: 0,
  });
  const [explorationTemp] = useState(0.3);
  const [mode, setMode] = useState<AppMode>('train');
  const [testOpponent, setTestOpponent] = useState<TestOpponent>('you');
  const [vsPlayerWins, setVsPlayerWins] = useState(0);
  const [vsPlayerLosses, setVsPlayerLosses] = useState(0);
  const [vsPlayerDraws, setVsPlayerDraws] = useState(0);
  const [forwardPassData, setForwardPassData] = useState<{
    activations: number[][];
    activeLayer: number;
  } | null>(null);

  const manualTeacherExamples = useMemo<TrainingExample[]>(() => {
    const dataset = generateTeacherDataset({ augmentSymmetries: false });
    return dataset.map((s) => ({
      input: NeuralNetwork.boardToInputWithTurn(s.board, s.turn),
      target: s.policy,
      mask: s.mask,
      weight: 1,
      value: s.value,
    }));
  }, []);

  const clearAiTimeout = () => {
    if (aiMoveTimeout.current !== null) {
      clearTimeout(aiMoveTimeout.current);
      aiMoveTimeout.current = null;
    }
  };

  // Initialize network from sessionStorage only (robust across navigation/state loss)
  useEffect(() => {
    console.log('Train useEffect running - loading snapshot from sessionStorage');
    const snapshotStr = sessionStorage.getItem('trainingSnapshot');
    const name = sessionStorage.getItem('trainingNetworkName');

    if (snapshotStr && name) {
      try {
        const snapshot: NetworkSnapshot = JSON.parse(snapshotStr);
        console.log('Parsed snapshot:', snapshot);

        setNetworkName(name);
        const net = new NeuralNetwork(snapshot.config);
        console.log('Network created, layers:', net.layers.length);

        // Prefer a previously-trained snapshot for continuity; fall back to fresh one
        const stored = localStorage.getItem(`network_${name}`);
        if (stored) {
          try {
            const trainedSnapshot: NetworkSnapshot = JSON.parse(stored);
            net.loadSnapshot(trainedSnapshot);
            console.log('Loaded trained snapshot from localStorage');
          } catch (err) {
            console.warn('Failed to load trained snapshot, using fresh config', err);
            net.loadSnapshot(snapshot);
          }
        } else {
          net.loadSnapshot(snapshot);
        }
        console.log('Snapshot loaded');

        setNetwork(net);
        setLayerDeltas(net.layers.map(() => ({ weights: [], biases: [] })));
        setBaseLearningRate(net.config.learningRate);

        // Load history if available
        const historyKey = `training_${name}`;
        const savedLosses = localStorage.getItem(historyKey);
        if (savedLosses) {
          try {
            setLosses(JSON.parse(savedLosses));
          } catch (e) {
            console.log('Could not parse saved losses');
          }
        }
        
        // Load test stats if available
        const savedStats = localStorage.getItem(`testStats_${name}`);
        if (savedStats) {
          try {
            setTestStats(JSON.parse(savedStats));
          } catch (e) {
            console.log('Could not parse saved test stats');
          }
        }
      } catch (e) {
        console.error('Failed to load network:', e);
        setInitError('Failed to initialize network. Returning to setup...');
        setTimeout(() => navigate('/train-setup'), 1500);
      }
    } else {
      console.log('No snapshot found in sessionStorage');
      setInitError('No network configuration found. Returning to setup...');
      setTimeout(() => navigate('/train-setup'), 1000);
    }
    return () => {
      clearAiTimeout();
    };
  }, [navigate]);

  const winner = useMemo(() => computeWinner(board), [board]);
  const isPlayerTurn = useMemo(() => status === 'ACTIVE' && turn === 'X', [status, turn]);

  const reset = () => {
    sessionRef.current += 1; // invalidate any pending AI moves
    clearAiTimeout();
    setBoard('_________');
    setTurn('X');
    setStatus('ACTIVE');
    setGameRecord(null);
    setLayerDeltas(network!.layers.map(() => ({ weights: [], biases: [] })));
    setAnimating(false);
  };
  const runTeacherTraining = async (dataset: TrainingExample[], gamesToTrain: number, delayMs = 120, countGames = 0) => {
    if (!network) return 0;
    if (dataset.length === 0 || gamesToTrain <= 0) return 0;

    const lrDropAt = 200; // games trained
    const lrDropFactor = 0.25; // e.g., 0.02 -> 0.005
    const scheduleLearningRate = (totalGamesCompleted: number) => {
      if (!network) return;
      const base = baseLearningRate ?? network.getLearningRate();
      const target = totalGamesCompleted >= lrDropAt ? base * lrDropFactor : base;
      network.setLearningRate(target);
    };

    const batchesPerGame = 14;
    const batchSize = 10;
    let totalLossAcrossGames = 0;
    let gamesTracked = 0;

    const sampleBatch = (): TrainingExample[] => {
      const batch: TrainingExample[] = [];
      for (let k = 0; k < batchSize; k += 1) {
        batch.push(dataset[Math.floor(Math.random() * dataset.length)]);
      }
      return batch;
    };

    setAnimating(true);
    try {
      for (let g = 0; g < gamesToTrain; g += 1) {
        let lossThisGame = 0;
        let batchesThisGame = 0;

        scheduleLearningRate(gameCount + gamesTracked);

        for (let b = 0; b < batchesPerGame; b += 1) {
          const batch = sampleBatch();
          const rawLoss = network.train(batch);
          const loss = Number.isFinite(rawLoss) ? rawLoss : 0;
          lossThisGame += loss;
          batchesThisGame += 1;

          // Update visualization after each batch
          setLayerDeltas(network.layers.map((_, idx) => network.getLayerDeltas(idx)));

          if (delayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, delayMs));
          }
        }

        const avgLossThisGame = batchesThisGame > 0 ? lossThisGame / batchesThisGame : 0;
        totalLossAcrossGames += avgLossThisGame;
        gamesTracked += 1;

        setLosses((prev) => {
          const updated = [...prev, avgLossThisGame];
          localStorage.setItem(`training_${networkName}`, JSON.stringify(updated));
          return updated;
        });

        if (countGames > 0) {
          setGameCount((prev) => prev + 1);
        }
      }

      setLayerDeltas(network.layers.map((_, idx) => network.getLayerDeltas(idx)));
      const networkSnapshot = network.getSnapshot();
      localStorage.setItem(`network_${networkName}`, JSON.stringify(networkSnapshot));
      return gamesTracked > 0 ? totalLossAcrossGames / gamesTracked : 0;
    } finally {
      setAnimating(false);
    }
  };

  const trainNetwork = async (record: GameRecord) => {
    setGameRecord(record);
    setSimulationMessage('');
    await runTeacherTraining(manualTeacherExamples, 1, 120, 1);
  };

  const simulateMinimaxRollouts = async () => {
    if (animating) return;
    setSimulationMessage('Playing minimax rollouts...');

    const episodes: TrainingExample[] = [];

    // Randomly choose among moves with equal (optimal) probability
    const chooseMove = (policy: number[], mask: number[]) => {
      // Find the max probability among valid moves
      let maxProb = -Infinity;
      for (let i = 0; i < policy.length; i++) {
        if (mask[i] === 1 && policy[i] > maxProb) {
          maxProb = policy[i];
        }
      }
      // Collect all moves with that probability (optimal moves)
      const optimalMoves: number[] = [];
      for (let i = 0; i < policy.length; i++) {
        if (mask[i] === 1 && Math.abs(policy[i] - maxProb) < 1e-6) {
          optimalMoves.push(i);
        }
      }
      if (optimalMoves.length === 0) return -1;
      // Randomly pick one of the optimal moves
      return optimalMoves[Math.floor(Math.random() * optimalMoves.length)];
    };

    for (let g = 0; g < simulatedGames; g += 1) {
      let b = '_________';
      let t: 'X' | 'O' = 'X';

      while (true) {
        const { policy, mask, value } = minimaxPolicy(b, t);
        episodes.push({
          input: NeuralNetwork.boardToInputWithTurn(b, t),
          target: policy,
          mask,
          weight: 1,
          value,
        });

        const move = chooseMove(policy, mask);
        if (move === -1) break;
        b = b.substring(0, move) + t + b.substring(move + 1);

        const w = computeWinner(b);
        if (w || !b.includes('_')) break;
        t = t === 'X' ? 'O' : 'X';
      }
    }

    if (episodes.length === 0) {
      setSimulationMessage('No rollout states generated.');
      return;
    }

    const avgLoss = await runTeacherTraining(episodes, simulatedGames, 40, simulatedGames);
    setSimulationMessage(`Trained on ${simulatedGames} games · avg loss ${avgLoss.toFixed(4)}`);
  };

  // Test AI vs different opponents
  const testAIvsOpponent = async (opponent: 'minimax' | 'random' | 'easy', numGames: number = 10) => {
    if (!network || animating) return;
    
    const opponentNames = { minimax: 'Perfect AI', random: 'Random Player', easy: 'Easy AI' };
    setSimulationMessage(`Testing AI against ${opponentNames[opponent]}...`);
    setAnimating(true);

    let wins = 0, losses = 0, draws = 0;

    // Helper to pick best move from policy
    const pickBestMove = (policy: number[], mask: number[]): number => {
      let best = -1, bestScore = -Infinity;
      for (let i = 0; i < policy.length; i++) {
        if (mask[i] === 0) continue;
        if (policy[i] > bestScore) { bestScore = policy[i]; best = i; }
      }
      return best;
    };

    // Random player - picks any valid move randomly
    const randomMove = (mask: number[]): number => {
      const validMoves = mask.map((m, i) => m === 1 ? i : -1).filter(i => i >= 0);
      if (validMoves.length === 0) return -1;
      return validMoves[Math.floor(Math.random() * validMoves.length)];
    };

    // Easy AI - 50% chance to play optimally, 50% random
    const easyMove = (b: string, t: 'X' | 'O'): number => {
      const mask = NeuralNetwork.boardToMask(b);
      if (Math.random() < 0.5) {
        // Play optimally
        const { policy } = minimaxPolicy(b, t);
        return pickBestMove(policy, mask);
      } else {
        // Play randomly
        return randomMove(mask);
      }
    };

    try {
      for (let g = 0; g < numGames; g++) {
        let b = '_________';
        let t: 'X' | 'O' = 'X';

        while (true) {
          let move: number;
          const mask = NeuralNetwork.boardToMask(b);
          
          if (t === 'X') {
            // Opponent plays X
            if (opponent === 'minimax') {
              const { policy } = minimaxPolicy(b, t);
              move = pickBestMove(policy, mask);
            } else if (opponent === 'random') {
              move = randomMove(mask);
            } else {
              move = easyMove(b, t);
            }
          } else {
            // AI plays O
            const input = NeuralNetwork.boardToInputWithTurn(b, 'O');
            const { policy } = network.forward(input, mask);
            move = pickBestMove(policy, mask);
          }

          if (move === -1 || !isMoveValid(b, move)) break;
          b = b.substring(0, move) + t + b.substring(move + 1);

          const w = computeWinner(b);
          if (w) {
            if (w === 'O') wins++;
            else losses++;
            break;
          }
          if (!b.includes('_')) {
            draws++;
            break;
          }
          t = t === 'X' ? 'O' : 'X';
        }

        if (g % 5 === 0) {
          await new Promise(r => setTimeout(r, 10));
        }
      }

      // Update stats based on opponent type
      const newStats = { ...testStats };
      if (opponent === 'minimax') {
        newStats.vsMinimaxWins += wins;
        newStats.vsMinimaxLosses += losses;
        newStats.vsMinimaxDraws += draws;
      } else if (opponent === 'random') {
        newStats.vsRandomWins += wins;
        newStats.vsRandomLosses += losses;
        newStats.vsRandomDraws += draws;
      } else {
        newStats.vsEasyWins += wins;
        newStats.vsEasyLosses += losses;
        newStats.vsEasyDraws += draws;
      }
      setTestStats(newStats);
      localStorage.setItem(`testStats_${networkName}`, JSON.stringify(newStats));

      const winRate = ((wins + draws * 0.5) / numGames * 100).toFixed(1);
      setSimulationMessage(`${wins}W / ${draws}D / ${losses}L vs ${opponentNames[opponent]} (${winRate}% score)`);
    } finally {
      setAnimating(false);
    }
  };

  const handlePlayerMove = (index: number) => {
    if (!isPlayerTurn || !isMoveValid(board, index) || !network || animating) return;

    const moves: GameRecord['moves'] = gameRecord?.moves ?? [];
    const b1 = board.substring(0, index) + 'X' + board.substring(index + 1);
    const w1 = computeWinner(b1);
    const full1 = !b1.includes('_');

    const newMoves: GameRecord['moves'] = [...moves, { board, move: index, turn: 'X' as const }];

    if (w1 || full1) {
      setBoard(b1);
      setTurn('X');
      setStatus('FINISHED');
      const outcome = w1 === 'X' ? -1 : 0;
      const record: GameRecord = { moves: newMoves, winner: w1, outcome };
      setGameRecord(record);
      
      // Update player stats (player won means AI lost)
      if (w1 === 'X') setVsPlayerWins(prev => prev + 1);
      else setVsPlayerDraws(prev => prev + 1);
      
      // Only train in train mode
      if (mode === 'train') {
        trainNetwork(record);
      }
      return;
    }

    setBoard(b1);
    setTurn('O');

    // AI move with optional forward pass visualization
    const sessionId = sessionRef.current;
    
    // Helper to animate forward pass layer by layer
    const animateForwardPass = (
      net: NeuralNetwork, 
      input: number[], 
      mask: number[], 
      onComplete: (policy: number[]) => void
    ) => {
      const layerSizes = [net.config.inputSize, ...net.config.hiddenLayers, net.config.outputSize];
      const totalLayers = layerSizes.length;
      
      // First, run the actual forward pass to get results
      const { policy } = net.forward(input, mask);
      const activations = net.getLayerActivations(input);
      
      // Animate layer by layer with delays
      let currentLayer = 0;
      const animateLayer = () => {
        if (sessionRef.current !== sessionId) {
          setForwardPassData(null);
          return;
        }
        
        if (currentLayer < totalLayers) {
          setForwardPassData({
            activations,
            activeLayer: currentLayer
          });
          currentLayer++;
          setTimeout(animateLayer, 400); // 400ms per layer
        } else {
          // Animation complete, clear after a brief pause
          setTimeout(() => {
            setForwardPassData(null);
            onComplete(policy);
          }, 300);
        }
      };
      
      animateLayer();
    };
    
    // Determine if we should animate the forward pass
    const shouldAnimate = mode === 'test' && testOpponent === 'you';
    
    aiMoveTimeout.current = window.setTimeout(() => {
      if (sessionRef.current !== sessionId) {
        return; // stale timeout after reset
      }
      const input = NeuralNetwork.boardToInputWithTurn(b1, 'O');
      const mask = NeuralNetwork.boardToMask(b1);
      
      // Temperature-based move selection for more varied play
      const sampleWithTemp = (policy: number[], maskArr: number[], temp: number): number => {
        if (temp <= 0.01) {
          // Greedy
          let best = -1, bestScore = -Infinity;
          for (let i = 0; i < policy.length; i++) {
            if (maskArr[i] === 0) continue;
            const score = Number.isFinite(policy[i]) ? policy[i] : -Infinity;
            if (score > bestScore) { bestScore = score; best = i; }
          }
          return best;
        }
        // Temperature sampling - higher temp = more exploration
        const validProbs = policy.map((p, i) => maskArr[i] === 1 ? Math.pow(Math.max(p, 1e-8), 1 / temp) : 0);
        const sum = validProbs.reduce((a, b) => a + b, 0);
        if (sum <= 0) return maskArr.indexOf(1);
        const normalized = validProbs.map(p => p / sum);
        const rand = Math.random();
        let cumulative = 0;
        for (let i = 0; i < normalized.length; i++) {
          cumulative += normalized[i];
          if (rand < cumulative && maskArr[i] === 1) return i;
        }
        return maskArr.lastIndexOf(1);
      };
      
      const executeMove = (output: number[]) => {
        // Use low temperature for mostly-greedy play with occasional exploration
        let bestMove = sampleWithTemp(output, mask, explorationTemp);

        // Fallback: pick first available spot if network output is invalid
        if (bestMove === -1) {
          bestMove = b1.indexOf('_');
        }

        if (bestMove === -1) {
          // No valid moves - draw
          setTurn('O');
          setStatus('FINISHED');
          const record: GameRecord = { moves: newMoves, winner: null, outcome: 0 };
          setGameRecord(record);
          setVsPlayerDraws(prev => prev + 1);
          if (mode === 'train') {
            trainNetwork(record);
          }
          aiMoveTimeout.current = null;
          return;
        }

        const b2 = b1.substring(0, bestMove) + 'O' + b1.substring(bestMove + 1);
        const w2 = computeWinner(b2);
        const full2 = !b2.includes('_');

        const aiMoves: GameRecord['moves'] = [...newMoves, { board: b1, move: bestMove, turn: 'O' as const }];

        if (w2 || full2) {
          setBoard(b2);
          setTurn('O');
          setStatus('FINISHED');
          const outcome = w2 === 'O' ? 1 : 0;
          const record: GameRecord = { moves: aiMoves, winner: w2, outcome };
          setGameRecord(record);
          
          // Update player stats (AI won means player lost)
          if (w2 === 'O') setVsPlayerLosses(prev => prev + 1);
          else setVsPlayerDraws(prev => prev + 1);
          
          // Only train in train mode
          if (mode === 'train') {
            trainNetwork(record);
          }
        } else {
          setBoard(b2);
          setTurn('X');
        }
        aiMoveTimeout.current = null;
      };
      
      if (shouldAnimate) {
        // Animate forward pass, then execute move
        animateForwardPass(network!, input, mask, executeMove);
      } else {
        // Direct execution without animation
        const { policy: output } = network!.forward(input, mask);
        executeMove(output);
      }
    }, shouldAnimate ? 100 : 300); // Shorter initial delay when animating
  };

  const safeLosses = losses.filter((l) => typeof l === 'number' && Number.isFinite(l));
  const avgLoss = safeLosses.length > 0 ? (safeLosses.reduce((a, b) => a + b, 0) / safeLosses.length).toFixed(4) : 'N/A';
  const plot = useMemo(() => {
    const width = 480;
    const height = 300;
    const pad = 18;
    if (safeLosses.length === 0) return { points: '', min: 0, max: 0, width, height, pad };
    const max = Math.max(...safeLosses);
    const min = Math.min(...safeLosses);
    const range = max - min || 1;
    const step = safeLosses.length > 1 ? (width - pad * 2) / (safeLosses.length - 1) : 0;
    const points = safeLosses
      .map((loss, i) => {
        const x = pad + i * step;
        const y = height - pad - ((loss - min) / range) * (height - pad * 2);
        return `${x},${y}`;
      })
      .join(' ');
    return { points, min, max, width, height, pad };
  }, [safeLosses]);

  // Status text changes based on mode
  const statusText =
    status === 'FINISHED'
      ? !winner
        ? mode === 'train' ? '🤝 Draw! AI is learning...' : '🤝 Draw!'
        : winner === 'X'
          ? mode === 'train' ? '🎉 You won! AI is learning...' : '🎉 You won!'
          : mode === 'train' ? '🤖 AI won! Training on this game.' : '🤖 AI won!'
      : isPlayerTurn
        ? '👤 Your turn'
        : '🧠 AI thinking...';

  if (!network) {
    return (
      <div className={`min-h-screen ${config.bg} flex items-center justify-center p-4 transition-all duration-500`}>
        <div className={`glass-card p-8 max-w-md w-full border ${config.card} text-center`}>
          <h2 className={`text-2xl font-bold ${config.text} mb-4`}>Loading Network...</h2>
          {initError ? (
            <p className={`${config.text} mb-4`}>⚠️ {initError}</p>
          ) : (
            <>
              <p className={`${config.text}/70 mb-4`}>Initializing your neural network architecture</p>
              <div className={`h-2 bg-gray-300 rounded-full overflow-hidden`}>
                <div className={`h-full ${config.accent} animate-pulse`}></div>
              </div>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${config.bg} p-4 md:p-6 transition-all duration-500`}>
      <div className={`max-w-7xl mx-auto`}>
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div>
            <h1 className={`text-3xl font-bold ${config.text} mb-1`}>
              {mode === 'train' ? '🎓 Training' : '🧪 Testing'}: <span className={config.accent}>{networkName}</span>
            </h1>
            <p className={`${config.text}/70 text-sm`}>
              Games Trained: <span className="font-semibold">{gameCount}</span> | Avg Loss: <span className="font-semibold">{avgLoss}</span>
            </p>
          </div>
          <button onClick={() => navigate('/')} className={`px-4 py-2 rounded-lg ${config.button} transition hover:scale-105 font-semibold`}>
            ← Home
          </button>
        </div>

        {/* Mode Tabs */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setMode('train')}
            className={`px-6 py-3 rounded-lg font-bold transition ${
              mode === 'train' 
                ? `${config.accent} shadow-lg` 
                : `${config.button} border border-white/20 hover:border-white/40`
            }`}
          >
            🎓 Train Mode
          </button>
          <button
            onClick={() => setMode('test')}
            className={`px-6 py-3 rounded-lg font-bold transition ${
              mode === 'test' 
                ? `${config.accent} shadow-lg` 
                : `${config.button} border border-white/20 hover:border-white/40`
            }`}
          >
            🧪 Test Mode
          </button>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6 auto-rows-max">
          {/* Left Column - Changes based on mode */}
          <div className={`lg:col-span-1 glass-card p-6 border ${config.card} flex flex-col`}>
            
            {mode === 'train' ? (
              /* TRAIN MODE UI */
              <>
                <h2 className={`text-lg font-bold ${config.text} mb-4 flex items-center gap-2`}>
                  <span>🎓</span> Train Your AI
                </h2>

                {/* Skill Level Indicator */}
                <div className={`mb-4 p-3 rounded-lg border ${config.card}`}>
                  <div className={`${config.text}/70 text-xs font-semibold mb-2`}>AI SKILL LEVEL</div>
                  <div className="flex items-center gap-2">
                    <div className={`flex-1 h-2 rounded-full bg-gray-700 overflow-hidden`}>
                      <div 
                        className={`h-full transition-all duration-500 ${
                          gameCount < 20 ? 'bg-red-500' : 
                          gameCount < 50 ? 'bg-orange-500' : 
                          gameCount < 100 ? 'bg-yellow-500' : 
                          gameCount < 200 ? 'bg-green-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${Math.min(100, gameCount / 2)}%` }}
                      />
                    </div>
                    <span className={`text-sm font-bold ${config.text}`}>
                      {gameCount < 20 ? '🔴 Novice' : 
                       gameCount < 50 ? '🟠 Beginner' : 
                       gameCount < 100 ? '🟡 Amateur' : 
                       gameCount < 200 ? '🟢 Skilled' : '🔵 Expert'}
                    </span>
                  </div>
                  <p className={`${config.text}/50 text-xs mt-1`}>
                    {gameCount < 50 
                      ? `Train ${50 - gameCount} more games to reach Amateur level`
                      : gameCount < 200 
                        ? 'Keep training to approach perfect play!'
                        : 'Your AI should now play optimally!'}
                  </p>
                </div>

                {/* Training Controls */}
                <div className={`mb-4 border rounded-lg p-4 ${config.card}`}>
                  <div className={`flex items-center justify-between mb-2 text-xs ${config.text}/70`}>
                    <span>Training rounds</span>
                    <span className="font-semibold">{simulatedGames}</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={50}
                    step={5}
                    value={simulatedGames}
                    onChange={(event) => setSimulatedGames(Number(event.target.value))}
                    className="w-full mb-4"
                  />
                  
                  <button
                    onClick={simulateMinimaxRollouts}
                    disabled={animating}
                    className={`w-full py-3 rounded-lg font-bold text-base transition ${
                      animating ? 'bg-gray-400 text-gray-800 cursor-not-allowed' : config.accent
                    }`}
                  >
                    {animating ? '⏳ Training...' : `🧠 Train ${simulatedGames} Games`}
                  </button>
                  
                  <p className={`${config.text}/50 text-xs mt-3 text-center`}>
                    AI learns from optimal Minimax play with varied openings
                  </p>
                  
                  {simulationMessage && (
                    <p className={`mt-3 text-sm ${config.text} text-center p-3 rounded-lg ${config.button} border border-white/10`}>
                      {simulationMessage}
                    </p>
                  )}
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-2 gap-3 text-center text-sm">
                  <div className={`${config.button} p-3 rounded-lg`}>
                    <div className={`${config.text}/70 text-xs font-semibold`}>GAMES TRAINED</div>
                    <div className={`text-2xl font-bold ${config.text}`}>{gameCount}</div>
                  </div>
                  <div className={`${config.button} p-3 rounded-lg`}>
                    <div className={`${config.text}/70 text-xs font-semibold`}>AVG LOSS</div>
                    <div className={`text-2xl font-bold ${config.text}`}>
                      {Number.isFinite(parseFloat(avgLoss)) ? (parseFloat(avgLoss) || 0).toFixed(3) : 'N/A'}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              /* TEST MODE UI */
              <>
                <h2 className={`text-lg font-bold ${config.text} mb-3 flex items-center gap-2`}>
                  <span>🧪</span> Test Your AI
                </h2>
                
                <div className={`${config.button} rounded-lg p-3 mb-4 border border-white/10`}>
                  <p className={`${config.text}/80 text-sm`}>
                    ⚡ <strong>No learning</strong> in test mode — see your AI's true skill!
                  </p>
                </div>

                {/* Opponent Selection - Horizontal tabs */}
                <div className={`mb-4`}>
                  <div className={`${config.text}/70 text-xs font-semibold mb-2`}>CHOOSE OPPONENT</div>
                  <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-black/20">
                    <button
                      onClick={() => setTestOpponent('you')}
                      className={`py-2 px-1 rounded text-center text-xs font-semibold transition ${
                        testOpponent === 'you' ? config.accent : 'hover:bg-white/10'
                      }`}
                    >
                      👤 You
                    </button>
                    <button
                      onClick={() => setTestOpponent('random')}
                      className={`py-2 px-1 rounded text-center text-xs font-semibold transition ${
                        testOpponent === 'random' ? config.accent : 'hover:bg-white/10'
                      }`}
                    >
                      🎲 Random
                    </button>
                    <button
                      onClick={() => setTestOpponent('easy')}
                      className={`py-2 px-1 rounded text-center text-xs font-semibold transition ${
                        testOpponent === 'easy' ? config.accent : 'hover:bg-white/10'
                      }`}
                    >
                      🤖 Easy
                    </button>
                    <button
                      onClick={() => setTestOpponent('minimax')}
                      className={`py-2 px-1 rounded text-center text-xs font-semibold transition ${
                        testOpponent === 'minimax' ? config.accent : 'hover:bg-white/10'
                      }`}
                    >
                      🧠 Perfect
                    </button>
                  </div>
                </div>

                {/* Test Actions */}
                {testOpponent === 'you' ? (
                  <div className={`mb-4 border rounded-lg p-4 ${config.card}`}>
                    <GameBoard board={board} disabled={status === 'FINISHED' || !isPlayerTurn || animating} onPick={handlePlayerMove} />
                    <p className={`text-center text-sm font-bold ${config.text} mt-4`}>{statusText}</p>
                    <button
                      onClick={reset}
                      disabled={animating}
                      className={`w-full mt-3 py-2 rounded-lg font-bold ${animating ? 'opacity-50 cursor-not-allowed' : config.accent} transition`}
                    >
                      {status === 'FINISHED' ? '▶ Play Again' : 'Reset Game'}
                    </button>
                    
                    {/* Player Stats */}
                    <div className={`mt-4 pt-3 border-t border-white/10`}>
                      <div className={`${config.text}/70 text-xs font-semibold mb-2`}>Your Record vs AI</div>
                      <div className="flex justify-between text-sm">
                        <span className="text-green-400">{vsPlayerWins}W</span>
                        <span className="text-yellow-400">{vsPlayerDraws}D</span>
                        <span className="text-red-400">{vsPlayerLosses}L</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className={`mb-4 border rounded-lg p-4 ${config.card}`}>
                    <p className={`${config.text}/70 text-sm mb-3 text-center`}>
                      {testOpponent === 'random' && '🎲 Random picks any valid move'}
                      {testOpponent === 'easy' && '🤖 Easy AI: 50% smart, 50% random'}
                      {testOpponent === 'minimax' && '🧠 Perfect AI never makes mistakes'}
                    </p>
                    <button
                      onClick={() => testAIvsOpponent(testOpponent as 'minimax' | 'random' | 'easy', 10)}
                      disabled={animating}
                      className={`w-full py-3 rounded-lg font-bold text-base transition ${
                        animating ? 'bg-gray-400 text-gray-800 cursor-not-allowed' : config.accent
                      }`}
                    >
                      {animating ? '⏳ Running Tests...' : `▶ Run 10 Games`}
                    </button>
                    
                    {simulationMessage && (
                      <p className={`mt-3 text-sm ${config.text} text-center p-3 rounded-lg ${config.button} border border-white/10`}>
                        {simulationMessage}
                      </p>
                    )}
                  </div>
                )}

                {/* Test Results */}
                <div className={`border rounded-lg p-4 ${config.card}`}>
                  <div className={`${config.text}/70 text-xs font-semibold mb-3`}>TEST RESULTS</div>
                  
                  {/* vs Random */}
                  <div className={`mb-3 p-2 rounded ${config.button}`}>
                    <div className={`text-xs ${config.text}/70 mb-1`}>🎲 vs Random Player</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-green-400">{testStats.vsRandomWins}W</span>
                      <span className="text-yellow-400">{testStats.vsRandomDraws}D</span>
                      <span className="text-red-400">{testStats.vsRandomLosses}L</span>
                      <span className={config.text}>
                        {(testStats.vsRandomWins + testStats.vsRandomLosses + testStats.vsRandomDraws) > 0 
                          ? `${((testStats.vsRandomWins + testStats.vsRandomDraws * 0.5) / (testStats.vsRandomWins + testStats.vsRandomLosses + testStats.vsRandomDraws) * 100).toFixed(0)}%`
                          : '--'}
                      </span>
                    </div>
                  </div>
                  
                  {/* vs Easy */}
                  <div className={`mb-3 p-2 rounded ${config.button}`}>
                    <div className={`text-xs ${config.text}/70 mb-1`}>🤖 vs Easy AI</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-green-400">{testStats.vsEasyWins}W</span>
                      <span className="text-yellow-400">{testStats.vsEasyDraws}D</span>
                      <span className="text-red-400">{testStats.vsEasyLosses}L</span>
                      <span className={config.text}>
                        {(testStats.vsEasyWins + testStats.vsEasyLosses + testStats.vsEasyDraws) > 0 
                          ? `${((testStats.vsEasyWins + testStats.vsEasyDraws * 0.5) / (testStats.vsEasyWins + testStats.vsEasyLosses + testStats.vsEasyDraws) * 100).toFixed(0)}%`
                          : '--'}
                      </span>
                    </div>
                  </div>
                  
                  {/* vs Minimax */}
                  <div className={`p-2 rounded ${config.button}`}>
                    <div className={`text-xs ${config.text}/70 mb-1`}>🧠 vs Perfect AI</div>
                    <div className="flex justify-between text-sm">
                      <span className="text-green-400">{testStats.vsMinimaxWins}W</span>
                      <span className="text-yellow-400">{testStats.vsMinimaxDraws}D</span>
                      <span className="text-red-400">{testStats.vsMinimaxLosses}L</span>
                      <span className={config.text}>
                        {(testStats.vsMinimaxWins + testStats.vsMinimaxLosses + testStats.vsMinimaxDraws) > 0 
                          ? `${((testStats.vsMinimaxWins + testStats.vsMinimaxDraws * 0.5) / (testStats.vsMinimaxWins + testStats.vsMinimaxLosses + testStats.vsMinimaxDraws) * 100).toFixed(0)}%`
                          : '--'}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Visualization & Analytics - Right Columns */}
          <div className={`lg:col-span-2 glass-card p-6 border ${config.card}`}>
            <h2 className={`text-lg font-bold ${config.text} mb-4 flex items-center gap-2`}>
              <span>🧠</span> Neural Network Learning
            </h2>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Network Activity Visualization */}
              <div>
                <h3 className={`text-sm font-semibold ${config.text} mb-3 flex items-center gap-2`}>
                  <span>⚡</span> Network Activity
                </h3>
                <div className={`border rounded-lg p-4 ${config.card} min-h-[500px]`}>
                  <NetworkVisualization
                    layerSizes={[network.config.inputSize, ...network.config.hiddenLayers, network.config.outputSize]}
                    layerDeltas={layerDeltas}
                    animating={animating}
                    forwardPassData={forwardPassData}
                  />
                </div>
              </div>

              {/* Loss History Chart */}
              <div>
                <h3 className={`text-sm font-semibold ${config.text} mb-3 flex items-center gap-2`}>
                  <span>📊</span> Training Progress
                </h3>
                <div className={`border rounded-lg ${config.card} p-4 h-[400px] flex flex-col`}>
                  {safeLosses.length === 0 ? (
                    <div className={`flex-1 flex items-center justify-center ${config.text}/50`}>
                      <div className="text-center">
                        <div className="text-4xl mb-3">📈</div>
                        <p className="text-sm mb-1">No training data yet</p>
                        <p className="text-xs opacity-70">Train some games to see progress</p>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* Stats header */}
                      <div className="flex justify-between items-center mb-3 pb-3 border-b border-white/10">
                        <div className="text-center">
                          <div className={`text-lg font-bold ${config.accent}`}>{safeLosses.length}</div>
                          <div className={`text-xs ${config.text}/60`}>Games</div>
                        </div>
                        <div className="text-center">
                          <div className={`text-lg font-bold text-blue-400`}>{safeLosses[safeLosses.length - 1]?.toFixed(3) ?? '—'}</div>
                          <div className={`text-xs ${config.text}/60`}>Latest Loss</div>
                        </div>
                        <div className="text-center">
                          <div className={`text-lg font-bold text-green-400`}>{plot.min.toFixed(3)}</div>
                          <div className={`text-xs ${config.text}/60`}>Best Loss</div>
                        </div>
                      </div>
                      
                      {/* Chart */}
                      <div className="flex-1 relative">
                        <svg
                          viewBox={`0 0 ${plot.width} ${plot.height}`}
                          className="w-full h-full"
                          preserveAspectRatio="none"
                        >
                          {/* Grid lines */}
                          <g stroke="currentColor" className={`${config.text}/10`}>
                            <line x1={plot.pad} y1={plot.pad} x2={plot.pad} y2={plot.height - plot.pad} />
                            <line x1={plot.pad} y1={plot.height - plot.pad} x2={plot.width - plot.pad} y2={plot.height - plot.pad} />
                            {/* Horizontal grid lines */}
                            {[0.25, 0.5, 0.75].map(pct => (
                              <line key={pct} x1={plot.pad} y1={plot.pad + (plot.height - plot.pad * 2) * pct} x2={plot.width - plot.pad} y2={plot.pad + (plot.height - plot.pad * 2) * pct} strokeDasharray="4 4" />
                            ))}
                          </g>
                          {/* Gradient fill under line */}
                          <defs>
                            <linearGradient id="lossGradient" x1="0" x2="0" y1="0" y2="1">
                              <stop offset="0%" stopColor="rgb(59, 130, 246)" stopOpacity="0.3" />
                              <stop offset="100%" stopColor="rgb(59, 130, 246)" stopOpacity="0" />
                            </linearGradient>
                          </defs>
                          {plot.points && (
                            <polygon
                              points={`${plot.pad},${plot.height - plot.pad} ${plot.points} ${plot.pad + (plot.width - plot.pad * 2)},${plot.height - plot.pad}`}
                              fill="url(#lossGradient)"
                            />
                          )}
                          {/* Line */}
                          <polyline
                            points={plot.points}
                            fill="none"
                            stroke="rgb(59, 130, 246)"
                            strokeWidth={2.5}
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          {/* Last point highlight */}
                          {plot.points && (() => {
                            const pts = plot.points.split(' ');
                            const last = pts[pts.length - 1];
                            if (!last) return null;
                            const [x, y] = last.split(',').map(Number);
                            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                            return (
                              <>
                                <circle cx={x} cy={y} r={6} fill="rgb(59, 130, 246)" opacity={0.3} />
                                <circle cx={x} cy={y} r={4} fill="rgb(59, 130, 246)" />
                              </>
                            );
                          })()}
                        </svg>
                      </div>
                      
                      {/* Legend */}
                      <div className={`text-xs ${config.text}/50 text-center mt-2 pt-2 border-t border-white/5`}>
                        Lower loss = better learning • Target: &lt; 0.5
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Network Architecture Info */}
            <div className="mt-6 pt-6 border-t border-white/10">
              <h3 className={`text-sm font-semibold ${config.text} mb-3 flex items-center gap-2`}>
                <span>🏗️</span> Architecture
              </h3>
              <div className={`grid grid-cols-2 md:grid-cols-5 gap-3 ${config.button} p-4 rounded-lg`}>
                <div>
                  <p className={`${config.text}/70 text-xs font-semibold mb-1`}>STRUCTURE</p>
                  <p className={`${config.text} font-mono text-xs`}>
                    {network.config.inputSize}→{network.config.hiddenLayers.join('→')}→{network.config.outputSize}
                  </p>
                </div>
                <div>
                  <p className={`${config.text}/70 text-xs font-semibold mb-1`}>LEARN RATE</p>
                  <p className={`${config.text} font-mono text-sm`}>{network.config.learningRate}</p>
                </div>
                <div>
                  <p className={`${config.text}/70 text-xs font-semibold mb-1`}>WEIGHT DECAY</p>
                  <p className={`${config.text} font-mono text-sm`}>{network.config.weightDecay ?? 0}</p>
                </div>
                <div>
                  <p className={`${config.text}/70 text-xs font-semibold mb-1`}>DROPOUT</p>
                  <p className={`${config.text} font-mono text-sm`}>{(network.config.dropout ?? 0.15) * 100}%</p>
                </div>
                <div>
                  <p className={`${config.text}/70 text-xs font-semibold mb-1`}>VALUE WT</p>
                  <p className={`${config.text} font-mono text-sm`}>{network.config.valueLossWeight ?? 0.5}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
