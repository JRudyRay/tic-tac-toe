import { useEffect, useRef, useCallback } from 'react';
import { useTheme } from '../context/ThemeContext';

interface NetworkVizProps {
  layerSizes: number[];
  layerDeltas: Array<{ weights: number[][]; biases: number[] }>;
  animating: boolean;
  // New props for forward propagation animation
  forwardPassData?: {
    activations: number[][];  // Activation values for each layer
    activeLayer: number;      // Currently highlighted layer during animation
  } | null;
}

export function NetworkVisualization({ 
  layerSizes, 
  layerDeltas, 
  animating,
  forwardPassData 
}: NetworkVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { config } = useTheme();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Background gradient
    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, 'rgba(12, 18, 32, 0.95)');
    bg.addColorStop(1, 'rgba(8, 12, 24, 0.88)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fillRect(8, 8, width - 16, height - 16);

    const padding = 60;
    const usableWidth = width - padding * 2;
    const usableHeight = height - padding * 2;

    // Normalize deltas for training visualization
    let maxWeightDelta = 0.001;
    let maxBiasDelta = 0.001;
    layerDeltas.forEach((d) => {
      if (d.weights) d.weights.forEach((row) => row.forEach((v) => { maxWeightDelta = Math.max(maxWeightDelta, Math.abs(v)); }));
      if (d.biases) d.biases.forEach((v) => { maxBiasDelta = Math.max(maxBiasDelta, Math.abs(v)); });
    });
    const normW = (v: number) => Math.min(Math.abs(v) / maxWeightDelta, 1);
    const normB = (v: number) => Math.min(Math.abs(v) / maxBiasDelta, 1);

    // Helper to get node position
    const getNodePos = (layerIdx: number, nodeIdx: number, layerSize: number) => {
      const x = padding + usableWidth * (layerIdx / (layerSizes.length - 1));
      const nodeSpacing = layerSize > 1 ? usableHeight / (layerSize - 1) : 0;
      const y = layerSize > 1 ? padding + nodeIdx * nodeSpacing : padding + usableHeight / 2;
      return { x, y };
    };

    // Draw connections first (behind nodes)
    // Note: layerDeltas[i] represents weights FROM layer i TO layer i+1 in the network
    // But network.layers doesn't include input layer, so layerDeltas[0] is input->hidden1
    // This means layerDeltas[layerIdx] corresponds to connection from layerSizes[layerIdx] to layerSizes[layerIdx+1]
    for (let layerIdx = 0; layerIdx < layerSizes.length - 1; layerIdx++) {
      const currentSize = layerSizes[layerIdx];
      const nextSize = layerSizes[layerIdx + 1];

      for (let i = 0; i < currentSize; i++) {
        const { x: x1, y: y1 } = getNodePos(layerIdx, i, currentSize);
        
        for (let j = 0; j < nextSize; j++) {
          const { x: x2, y: y2 } = getNodePos(layerIdx + 1, j, nextSize);
          
          let alpha = 0.1;
          let lineWidth = 0.8;
          let color = 'rgba(100, 100, 100, 0.2)';

          // During training - show weight updates
          // layerDeltas[layerIdx] has weights from layer layerIdx to layerIdx+1
          const deltaIdx = layerIdx; // Connection from layerIdx to layerIdx+1
          if (animating && deltaIdx < layerDeltas.length && layerDeltas[deltaIdx]?.weights?.[i]?.[j] !== undefined) {
            const weight = normW(layerDeltas[deltaIdx].weights[i][j]);
            alpha = 0.15 + weight * 0.85;
            lineWidth = 0.8 + weight * 3;
            // Green for updates
            color = `rgba(34, 197, 94, ${alpha})`;
          }
          
          // During forward pass - highlight active connections
          if (forwardPassData && forwardPassData.activeLayer >= layerIdx) {
            const sourceActivation = forwardPassData.activations[layerIdx]?.[i] ?? 0;
            if (forwardPassData.activeLayer > layerIdx && sourceActivation > 0.1) {
              alpha = 0.3 + sourceActivation * 0.7;
              lineWidth = 1 + sourceActivation * 2.5;
              color = `rgba(251, 191, 36, ${alpha})`; // Amber/gold for forward pass
            }
          }

          ctx.strokeStyle = color;
          ctx.lineWidth = lineWidth;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    layerSizes.forEach((size, layerIdx) => {
      for (let i = 0; i < size; i++) {
        const { x, y } = getNodePos(layerIdx, i, size);

        let intensity = 0;
        let nodeColor = 'rgba(100, 100, 100, 0.5)';
        let radius = 7;
        
        // Get default color based on layer type
        const getLayerColor = (alpha: number) => {
          if (layerIdx === 0) {
            return `rgba(129, 230, 217, ${alpha})`; // Teal for input
          } else if (layerIdx === layerSizes.length - 1) {
            return `rgba(251, 146, 60, ${alpha})`; // Orange for output
          } else {
            return `rgba(99, 179, 237, ${alpha})`; // Blue for hidden
          }
        };

        // During training - show bias updates
        // layerDeltas[i] contains biases for layer i+1 (since input layer has no biases)
        // So for layerSizes[1] (first hidden), we use layerDeltas[0]
        const deltaIdx = layerIdx - 1; // Offset by 1 since input layer has no deltas
        if (animating && layerIdx > 0 && deltaIdx >= 0 && deltaIdx < layerDeltas.length && layerDeltas[deltaIdx]?.biases?.[i] !== undefined) {
          intensity = normB(layerDeltas[deltaIdx].biases[i]);
          radius = 7 + intensity * 6;
          const alpha = 0.4 + intensity * 0.6;
          nodeColor = getLayerColor(alpha);
        } else if (animating && layerIdx === 0) {
          // Input layer gets default animated color during training
          nodeColor = getLayerColor(0.7);
        }
        
        // During forward pass - show activations
        if (forwardPassData && forwardPassData.activations[layerIdx]) {
          const activation = forwardPassData.activations[layerIdx][i] ?? 0;
          const isActive = forwardPassData.activeLayer >= layerIdx;
          
          if (isActive) {
            intensity = activation;
            radius = 7 + activation * 7;
            const alpha = 0.4 + activation * 0.6;
            
            if (layerIdx === 0) {
              nodeColor = `rgba(129, 230, 217, ${alpha})`; // Teal for input
            } else if (layerIdx === layerSizes.length - 1) {
              nodeColor = `rgba(251, 146, 60, ${alpha})`; // Orange for output
            } else {
              nodeColor = `rgba(251, 191, 36, ${alpha})`; // Gold for hidden during forward
            }
          }
        }
        
        // Default colors when idle or when no intensity was set
        if (!animating && !forwardPassData) {
          radius = 7;
          nodeColor = getLayerColor(0.45);
        } else if (animating && intensity === 0) {
          // During training but no updates for this node - show dimmed layer color
          nodeColor = getLayerColor(0.35);
        }

        // Draw node
        ctx.fillStyle = nodeColor;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Pulse ring for high intensity
        if (intensity > 0.3) {
          ctx.strokeStyle = nodeColor.replace(/[\d.]+\)$/, `${intensity * 0.4})`);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, radius + 5, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    });

    // Draw layer labels
    ctx.fillStyle = 'rgba(229, 231, 235, 0.85)';
    ctx.font = '600 11px "JetBrains Mono", "Fira Code", monospace';
    ctx.textAlign = 'center';
    
    const labels = ['Input', ...layerSizes.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), 'Output'];
    layerSizes.forEach((size, layerIdx) => {
      const { x } = getNodePos(layerIdx, 0, size);
      ctx.fillText(labels[layerIdx], x, height - 15);
      ctx.font = '700 13px "JetBrains Mono", "Fira Code", monospace';
      ctx.fillText(`${size}`, x, padding - 22);
      ctx.font = '600 11px "JetBrains Mono", "Fira Code", monospace';
    });

    // Status indicator
    if (forwardPassData) {
      ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
      ctx.font = '700 12px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('▶ Forward Pass', 15, 25);
    } else if (animating) {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.9)';
      ctx.font = '700 12px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('◀ Learning', 15, 25);
    }

  }, [layerSizes, layerDeltas, animating, forwardPassData]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div className="w-full flex flex-col gap-2">
      <canvas
        ref={canvasRef}
        width={720}
        height={650}
        className={`w-full border rounded-lg ${config.card}`}
        style={{ background: 'transparent' }}
      />
      {/* Legend */}
      <div className="flex justify-between items-center px-2">
        <div className={`text-xs ${config.text}/60 flex items-center gap-4`}>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-teal-400"></span> Input
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-400"></span> Hidden
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-400"></span> Output
          </span>
        </div>
        <div className={`text-xs ${config.text}/50`}>
          {forwardPassData ? '🟡 Signal flow' : animating ? '🟢 Weight updates' : '○ Idle'}
        </div>
      </div>
    </div>
  );
}
