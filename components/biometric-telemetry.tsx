import { useEffect, useState } from 'react';

interface BiometricTelemetryProps {
  isScanning: boolean;
}

interface TelemetryMetrics {
  heartRate: number | null;
  spatialCoherence: number | null;
  spoofProbability: number | null;
}

export function BiometricTelemetry({ isScanning }: BiometricTelemetryProps) {
  const [metrics, setMetrics] = useState<TelemetryMetrics>({
    heartRate: null,
    spatialCoherence: null,
    spoofProbability: null,
  });

  useEffect(() => {
    if (!isScanning) {
      setMetrics({
        heartRate: null,
        spatialCoherence: null,
        spoofProbability: null,
      });
      return;
    }

    // Simulate telemetry data updates during scan
    const interval = setInterval(() => {
      setMetrics({
        heartRate: Math.floor(Math.random() * 40 + 60),
        spatialCoherence: Math.floor(Math.random() * 30 + 70),
        spoofProbability: Math.floor(Math.random() * 15),
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isScanning]);

  return (
    <div className="bg-card border border-border rounded-lg p-4 mb-6 font-mono text-sm">
      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            Heart Rate
          </p>
          <p className="text-lg font-semibold text-foreground">
            {metrics.heartRate !== null ? `${metrics.heartRate} BPM` : '-- BPM'}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            Spatial Coherence
          </p>
          <p className="text-lg font-semibold text-accent">
            {metrics.spatialCoherence !== null ? `${metrics.spatialCoherence}%` : '--%'}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            Spoof Probability
          </p>
          <p
            className={`text-lg font-semibold ${
              metrics.spoofProbability !== null && metrics.spoofProbability > 20
                ? 'text-destructive'
                : 'text-foreground'
            }`}
          >
            {metrics.spoofProbability !== null ? `${metrics.spoofProbability}%` : '--%'}
          </p>
        </div>
      </div>
    </div>
  );
}
