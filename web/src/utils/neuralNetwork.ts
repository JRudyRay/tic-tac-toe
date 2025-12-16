/**
 * Simple Multi-Layer Perceptron for Tic-Tac-Toe
 * Learns to play through supervised learning on game outcomes
 */

export interface NetworkConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  learningRate: number;
  valueLossWeight?: number;
  weightDecay?: number; // L2 style decay on weights
  gradientClip?: number; // Maximum gradient magnitude
  dropout?: number; // Dropout rate (0-0.5), applied during training only
}

export interface TrainingExample {
  input: number[]; // includes turn feature
  target: number[]; // probability distribution over valid moves (masked one-hot)
  mask: number[]; // 1 for valid moves, 0 for filled cells
  weight: number; // sample weight (e.g., outcome-based)
  value: number; // expected outcome in [-1,1] from the current player's perspective
}

export interface NetworkSnapshot {
  config: NetworkConfig;
  weights: number[][][]; // trunk + policy head
  biases: number[][]; // trunk + policy head
  valueHeadWeights?: number[][];
  valueHeadBiases?: number[];
  trainingCount: number;
}

class Layer {
  weights: number[][];
  biases: number[];
  weightDeltas: number[][];
  biasDeltas: number[];
  inputActivation: number[];
  outputActivation: number[]; // for output layer this is softmax probabilities

  constructor(inputSize: number, outputSize: number) {
    const limit = Math.sqrt(6 / (inputSize + outputSize)); // Glorot uniform
    this.weights = Array(inputSize)
      .fill(0)
      .map(() => Array(outputSize).fill(0).map(() => (Math.random() * 2 - 1) * limit));
    this.biases = Array(outputSize).fill(0);
    this.weightDeltas = Array(inputSize).fill(0).map(() => Array(outputSize).fill(0));
    this.biasDeltas = Array(outputSize).fill(0);
    this.inputActivation = [];
    this.outputActivation = [];
  }
}

export class NeuralNetwork {
  config: NetworkConfig;
  layers: Layer[]; // trunk + policy head (used for visualization)
  valueHead: Layer;
  trainingCount: number = 0;
  private lastHiddenActivation: number[] = [];
  private valueLossWeight: number;
  private weightDecay: number;
  private gradientClip: number;
  private currentLearningRate: number;
  private dropout: number;
  private isTraining: boolean = false;
  private dropoutMasks: number[][] = [];

  constructor(config: NetworkConfig) {
    this.config = config;
    this.layers = [];
    this.valueLossWeight = config.valueLossWeight ?? 0.5;
    this.weightDecay = config.weightDecay ?? 0.0005;
    this.gradientClip = config.gradientClip ?? 5.0; // Prevent exploding gradients
    this.currentLearningRate = config.learningRate;
    this.dropout = config.dropout ?? 0.15; // Default 15% dropout

    const sizes = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    const hiddenSizes = sizes.slice(0, -1);
    for (let i = 0; i < hiddenSizes.length - 1; i++) {
      this.layers.push(new Layer(hiddenSizes[i], hiddenSizes[i + 1]));
    }

    // Policy head maps from last hidden to move logits
    const lastHidden = hiddenSizes.length > 0 ? hiddenSizes[hiddenSizes.length - 1] : this.config.inputSize;
    this.layers.push(new Layer(lastHidden, this.config.outputSize));

    // Value head maps from last hidden to scalar value
    this.valueHead = new Layer(lastHidden, 1);
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  private clipGradient(grad: number): number {
    return Math.max(-this.gradientClip, Math.min(this.gradientClip, grad));
  }

  private softmax(logits: number[], mask?: number[]): number[] {
    const adjusted = logits.map((v, i) => (mask && mask[i] === 0 ? -1e9 : v));
    const maxLogit = Math.max(...adjusted);
    const exps = adjusted.map((v) => Math.exp(v - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((v) => v / sum);
  }

  // Set training mode (enables dropout)
  setTraining(training: boolean): void {
    this.isTraining = training;
  }

  // Generate dropout mask for a layer
  private generateDropoutMask(size: number): number[] {
    if (!this.isTraining || this.dropout <= 0) {
      return Array(size).fill(1);
    }
    const scale = 1 / (1 - this.dropout); // Inverted dropout scaling
    return Array(size).fill(0).map(() => Math.random() > this.dropout ? scale : 0);
  }

  forward(input: number[], mask?: number[]): { policy: number[]; value: number } {
    let activation = input;
    this.dropoutMasks = [];

    // Hidden trunk
    for (let l = 0; l < this.layers.length - 1; l++) {
      const layer = this.layers[l];
      const next: number[] = [];

      layer.inputActivation = activation;

      for (let j = 0; j < layer.weights[0].length; j++) {
        let sum = layer.biases[j];
        for (let i = 0; i < activation.length; i++) {
          sum += activation[i] * layer.weights[i][j];
        }
        next.push(this.relu(sum));
      }

      // Apply dropout to hidden layer outputs
      const dropoutMask = this.generateDropoutMask(next.length);
      this.dropoutMasks.push(dropoutMask);
      const droppedNext = next.map((v, i) => v * dropoutMask[i]);

      layer.outputActivation = droppedNext;
      activation = droppedNext; // Use dropped output for next layer
    }

    // Cache last hidden activation (could be input if no hidden layers)
    this.lastHiddenActivation = activation;

    // Policy head (last layer in layers array)
    const policyLayer = this.layers[this.layers.length - 1];
    policyLayer.inputActivation = this.lastHiddenActivation;
    const policyLogits: number[] = [];
    for (let j = 0; j < policyLayer.weights[0].length; j++) {
      let sum = policyLayer.biases[j];
      for (let i = 0; i < this.lastHiddenActivation.length; i++) {
        sum += this.lastHiddenActivation[i] * policyLayer.weights[i][j];
      }
      policyLogits.push(sum);
    }
    const policy = this.softmax(policyLogits, mask);
    policyLayer.outputActivation = policy;

    // Value head
    this.valueHead.inputActivation = this.lastHiddenActivation;
    let valueSum = this.valueHead.biases[0];
    for (let i = 0; i < this.lastHiddenActivation.length; i++) {
      valueSum += this.lastHiddenActivation[i] * this.valueHead.weights[i][0];
    }
    // Bound value to [-1,1]
    const value = Math.tanh(valueSum);
    this.valueHead.outputActivation = [value];

    return { policy, value };
  }

  // Get all layer activations for visualization (call after forward())
  getLayerActivations(input: number[]): number[][] {
    const activations: number[][] = [];
    
    // Input layer activations (just the input values, normalized to 0-1 range)
    activations.push(input.map(v => Math.max(0, Math.min(1, (v + 1) / 2))));
    
    // Hidden layer activations (from cached output activations)
    for (let l = 0; l < this.layers.length - 1; l++) {
      const layer = this.layers[l];
      if (layer.outputActivation && layer.outputActivation.length > 0) {
        // Normalize ReLU outputs to 0-1 using softmax-like normalization
        const maxAct = Math.max(...layer.outputActivation, 0.001);
        activations.push(layer.outputActivation.map(v => v / maxAct));
      } else {
        activations.push([]);
      }
    }
    
    // Output layer (policy probabilities, already 0-1)
    const policyLayer = this.layers[this.layers.length - 1];
    if (policyLayer.outputActivation && policyLayer.outputActivation.length > 0) {
      activations.push(policyLayer.outputActivation);
    } else {
      activations.push([]);
    }
    
    return activations;
  }

  private backward(policyDelta: number[], valueDelta: number[]): void {
    const lr = this.currentLearningRate;
    // valueDelta is length 1 (dLoss/dValue)

    // Backprop heads
    const policyLayer = this.layers[this.layers.length - 1];
    const prevHidden = this.lastHiddenActivation;

    const policyWeightsSnapshot = policyLayer.weights.map((row) => row.slice());
    const valueWeightsSnapshot = this.valueHead.weights.map((row) => row.slice());

    // Initialize visualization deltas for policy head
    policyLayer.weightDeltas = Array(prevHidden.length)
      .fill(0)
      .map(() => Array(policyDelta.length).fill(0));
    policyLayer.biasDeltas = Array(policyDelta.length).fill(0);

    for (let i = 0; i < prevHidden.length; i++) {
      for (let j = 0; j < policyDelta.length; j++) {
        const grad = this.clipGradient(policyDelta[j] * prevHidden[i]);
        // L2 decay + gradient step
        policyLayer.weights[i][j] = policyLayer.weights[i][j] * (1 - lr * this.weightDecay) - lr * grad;
        policyLayer.weightDeltas[i][j] = Math.abs(grad);
      }
    }
    for (let j = 0; j < policyDelta.length; j++) {
      const grad = this.clipGradient(policyDelta[j]);
      policyLayer.biases[j] -= lr * grad;
      policyLayer.biasDeltas[j] = Math.abs(grad);
    }

    // Value head update (not shown in viz)
    this.valueHead.weightDeltas = Array(prevHidden.length)
      .fill(0)
      .map(() => Array(1).fill(0));
    this.valueHead.biasDeltas = Array(1).fill(0);

    for (let i = 0; i < prevHidden.length; i++) {
      const grad = this.clipGradient(valueDelta[0] * prevHidden[i]);
      this.valueHead.weights[i][0] = this.valueHead.weights[i][0] * (1 - lr * this.weightDecay) - lr * grad;
      this.valueHead.weightDeltas[i][0] = Math.abs(grad);
    }
    const valueGrad = this.clipGradient(valueDelta[0]);
    this.valueHead.biases[0] -= lr * valueGrad;
    this.valueHead.biasDeltas[0] = Math.abs(valueGrad);

    // Backprop into shared hidden from both heads
    let sharedDelta = Array(prevHidden.length).fill(0);

    // Contribution from policy head (use pre-update weights)
    for (let i = 0; i < prevHidden.length; i++) {
      let sum = 0;
      for (let j = 0; j < policyDelta.length; j++) {
        sum += policyDelta[j] * policyWeightsSnapshot[i][j];
      }
      sharedDelta[i] += sum;
    }

    // Contribution from value head (use pre-update weights)
    for (let i = 0; i < prevHidden.length; i++) {
      sharedDelta[i] += valueDelta[0] * valueWeightsSnapshot[i][0];
    }

    // Backprop through hidden layers (if any)
    for (let l = this.layers.length - 2; l >= 0; l--) {
      const layer = this.layers[l];
      const prevActivation = l === 0 ? this.layers[l].inputActivation : this.layers[l - 1].outputActivation;

      const delta = layer.outputActivation.map((act, idx) => sharedDelta[idx] * this.reluDerivative(act));

      // Store for visualization
      layer.weightDeltas = Array(prevActivation.length)
        .fill(0)
        .map(() => Array(delta.length).fill(0));
      layer.biasDeltas = Array(delta.length).fill(0);

      for (let i = 0; i < prevActivation.length; i++) {
        for (let j = 0; j < delta.length; j++) {
          const grad = this.clipGradient(delta[j] * prevActivation[i]);
          layer.weights[i][j] = layer.weights[i][j] * (1 - lr * this.weightDecay) - lr * grad;
          layer.weightDeltas[i][j] = Math.abs(grad);
        }
      }

      for (let j = 0; j < delta.length; j++) {
        const grad = this.clipGradient(delta[j]);
        layer.biases[j] -= lr * grad;
        layer.biasDeltas[j] = Math.abs(grad);
      }

      // Prepare sharedDelta for next layer up (previous hidden)
      const nextShared: number[] = Array(prevActivation.length).fill(0);
      for (let i = 0; i < prevActivation.length; i++) {
        let sum = 0;
        for (let j = 0; j < delta.length; j++) {
          sum += delta[j] * layer.weights[i][j];
        }
        nextShared[i] = sum;
      }
      sharedDelta = nextShared;
    }
  }

  train(examples: TrainingExample[]): number {
    this.setTraining(true); // Enable dropout during training
    let totalLoss = 0;
    let totalWeight = 0;

    for (const example of examples) {
      const { policy, value } = this.forward(example.input, example.mask);
      const eps = 1e-9;

      // Policy loss (cross-entropy)
      const policyLoss = -example.target.reduce((sum, t, i) => sum + t * Math.log(Math.max(policy[i], eps)), 0);

      // Value loss (MSE) bounded by tanh in forward
      const valueError = value - example.value;
      const valueLoss = valueError * valueError;

      const sampleLoss = policyLoss + this.valueLossWeight * valueLoss;
      totalLoss += sampleLoss * example.weight;
      totalWeight += example.weight;

      // Gradients
      const policyGrad = policy.map((p, i) => (p - example.target[i]) * example.weight);
      const dValue = 2 * valueError * this.valueLossWeight * example.weight * (1 - value * value); // derivative through tanh

      this.backward(policyGrad, [dValue]);
    }

    this.setTraining(false); // Disable dropout after training
    this.trainingCount += examples.length;
    return totalWeight > 0 ? totalLoss / totalWeight : 0;
  }

  predict(input: number[], mask?: number[]): number {
    const { policy } = this.forward(input, mask);
    let best = -1;
    let score = -Infinity;
    for (let i = 0; i < policy.length; i++) {
      const s = policy[i];
      if (mask && mask[i] === 0) continue;
      if (s > score) {
        score = s;
        best = i;
      }
    }
    return best;
  }

  getSnapshot(): NetworkSnapshot {
    // Persist the current learning rate so resumed training uses the latest schedule
    this.config.learningRate = this.currentLearningRate;
    return {
      config: this.config,
      weights: this.layers.map((l) => l.weights),
      biases: this.layers.map((l) => l.biases),
      valueHeadWeights: this.valueHead.weights,
      valueHeadBiases: this.valueHead.biases,
      trainingCount: this.trainingCount,
    };
  }

  loadSnapshot(snapshot: NetworkSnapshot): void {
    this.trainingCount = snapshot.trainingCount;
    console.log('loadSnapshot: layers.length =', this.layers.length, 'snapshot.weights.length =', snapshot.weights.length);
    for (let i = 0; i < this.layers.length; i++) {
      if (!snapshot.weights[i] || !snapshot.biases[i]) {
        throw new Error('Snapshot layer missing data');
      }
      const expectedIn = this.layers[i].weights.length;
      const expectedOut = this.layers[i].weights[0].length;
      const gotIn = snapshot.weights[i].length;
      const gotOut = snapshot.weights[i][0]?.length;
      if (expectedIn !== gotIn || expectedOut !== gotOut) {
        throw new Error(`Snapshot shape mismatch at layer ${i}: expected ${expectedIn}x${expectedOut}, got ${gotIn}x${gotOut}`);
      }
      console.log(`Layer ${i}: loading weights[${gotIn}][${gotOut}] and biases[${snapshot.biases[i].length}]`);
      this.layers[i].weights = JSON.parse(JSON.stringify(snapshot.weights[i]));
      this.layers[i].biases = JSON.parse(JSON.stringify(snapshot.biases[i]));
    }
    if (snapshot.valueHeadWeights && snapshot.valueHeadBiases) {
      this.valueHead.weights = JSON.parse(JSON.stringify(snapshot.valueHeadWeights));
      this.valueHead.biases = JSON.parse(JSON.stringify(snapshot.valueHeadBiases));
    }
    console.log('loadSnapshot complete');
    this.currentLearningRate = this.config.learningRate;
  }

  getLayerDeltas(layerIndex: number): { weights: number[][]; biases: number[] } {
    const layer = this.layers[layerIndex];
    return {
      weights: layer.weightDeltas,
      biases: layer.biasDeltas,
    };
  }

  /**
   * Convert board state to network input (9 values: 0=empty, 1=X, -1=O)
   */
  static boardToInputWithTurn(board: string, turn: 'X' | 'O'): number[] {
    const turnVal = turn === 'X' ? 1 : -1;
    const cells = board.split('').map((c) => {
      if (c === 'X') return 1;
      if (c === 'O') return -1;
      return 0;
    });
    return [...cells, turnVal];
  }

  /**
   * Convert move index to network target (one-hot encoded)
   */
  static moveToTarget(moveIndex: number): number[] {
    const target = Array(9).fill(0);
    target[moveIndex] = 1;
    return target;
  }

  static boardToMask(board: string): number[] {
    return board.split('').map((c) => (c === '_' ? 1 : 0));
  }

  setLearningRate(lr: number): void {
    this.currentLearningRate = lr;
    this.config.learningRate = lr;
  }

  getLearningRate(): number {
    return this.currentLearningRate;
  }
}
