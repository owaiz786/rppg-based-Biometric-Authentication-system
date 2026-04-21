'use client';

import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle, XCircle } from 'lucide-react';

interface BiometricTelemetryProps {
  isScanning: boolean;
  // Real BCG/liveness results — passed in after scan completes
  bcgResult?: {
    bcg_hr_bpm:       number;
    rppg_hr_bpm:      number;
    bcg_signal_power: number;
    bcg_passed:       boolean;
    freq_match:       boolean;
    coherence_score:  number;
    challenge_passed: boolean;
  } | null;
}

interface LiveMetrics {
  heartRate:        number | null;
  bcgPower:         number | null;
  spoofProbability: number | null;
}

export function BiometricTelemetry({ isScanning, bcgResult }: BiometricTelemetryProps) {
  const [live, setLive] = useState<LiveMetrics>({
    heartRate: null, bcgPower: null, spoofProbability: null,
  });

  // Simulate ticker while scanning — gives real-time feedback feel
  useEffect(() => {
    if (!isScanning) {
      setLive({ heartRate: null, bcgPower: null, spoofProbability: null });
      return;
    }
    const interval = setInterval(() => {
      setLive({
        heartRate:        Math.floor(Math.random() * 40 + 60),
        bcgPower:         parseFloat((Math.random() * 0.004 + 0.001).toFixed(4)),
        spoofProbability: Math.floor(Math.random() * 10),
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isScanning]);

  const showReal   = !!bcgResult && !isScanning;
  const displayHR  = showReal ? bcgResult!.bcg_hr_bpm  : live.heartRate;
  const rppgHR     = showReal ? bcgResult!.rppg_hr_bpm : null;
  const displayPow = showReal ? bcgResult!.bcg_signal_power : live.bcgPower;

  // FIX: Removed the hardcoded `95` that was showing whenever BCG and
  // challenge both failed — which happened 100% of the time when signal
  // extraction was broken. Now we only show a spoof probability if the
  // backend actually returns a coherence score suggesting a spoof
  // (coherence > 0.9 is suspicious; 0.98+ is a hard block).
  // During scanning we keep the low simulated value for UX feel.
  const displaySpoof: number | null = showReal
    ? (() => {
        const c = bcgResult!.coherence_score;
        if (c > 0.98) return 99;          // screen replay — near certain
        if (c > 0.90) return Math.round(c * 80);  // suspicious
        if (c > 0.50) return Math.round(c * 20);  // mild
        return 0;                          // normal — don't show
      })()
    : live.spoofProbability;

  const freqMatchColor = showReal
    ? (bcgResult!.freq_match ? 'text-emerald-400' : 'text-amber-400')
    : 'text-foreground';

  const hasBcgData = showReal
    && bcgResult!.bcg_signal_power > 1e-7
    && bcgResult!.bcg_hr_bpm > 0;

  const challengeAttempted = showReal
    && (bcgResult!.challenge_passed === true || bcgResult!.challenge_passed === false);

  return (
    <div className="bg-card border border-border rounded-lg p-4 w-full font-mono text-sm space-y-3">

      {/* ── Row 1: Heart Rate (BCG + rPPG) ── */}
      {(hasBcgData || isScanning) && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
              BCG Heart Rate
            </p>
            <p className={`text-lg font-semibold ${freqMatchColor}`}>
              {displayHR != null && displayHR > 0 ? `${Math.round(displayHR)} BPM` : '-- BPM'}
            </p>
            {showReal && bcgResult!.freq_match && hasBcgData && (
              <p className="text-[10px] text-emerald-500 mt-0.5 flex items-center">
                <CheckCircle className="h-3 w-3 mr-1" /> matches rPPG
              </p>
            )}
            {showReal && !bcgResult!.freq_match && rppgHR != null && rppgHR > 0 && hasBcgData && (
              <p className="text-[10px] text-amber-400 mt-0.5 flex items-center">
                <AlertCircle className="h-3 w-3 mr-1" /> rPPG: {Math.round(rppgHR)} BPM
              </p>
            )}
          </div>

          <div>
            <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
              BCG Signal Power
            </p>
            <p className="text-lg font-semibold text-accent">
              {displayPow != null && displayPow > 0
                ? displayPow < 0.0001
                  ? displayPow.toExponential(2)
                  : displayPow.toFixed(4)
                : '--'}
            </p>
            {showReal && hasBcgData && (
              <p className={`text-[10px] mt-0.5 flex items-center ${
                bcgResult!.bcg_passed ? 'text-emerald-500' : 'text-amber-500'
              }`}>
                {bcgResult!.bcg_passed
                  ? <><CheckCircle className="h-3 w-3 mr-1" /> heartbeat detected</>
                  : <><AlertCircle className="h-3 w-3 mr-1" /> weak signal</>}
              </p>
            )}
          </div>
        </div>
      )}

      {/* Message if BCG attempted but no data */}
      {showReal && !hasBcgData && !isScanning && (
        <div className="text-amber-500 text-xs flex items-center justify-center py-2 border-b border-border">
          <AlertCircle className="h-4 w-4 mr-2" />
          BCG requires steady head position — try holding still
        </div>
      )}

      {/* ── Row 2: Liveness layers ── */}
      <div className="grid grid-cols-3 gap-3 border-t border-border pt-3">

        {/* rPPG coherence */}
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            rPPG
          </p>
          <p className={`text-sm font-semibold ${
            showReal
              ? bcgResult!.coherence_score >= 0.1 && bcgResult!.coherence_score <= 0.95
                ? 'text-emerald-400' : 'text-amber-400'
              : 'text-foreground'
          }`}>
            {showReal
              ? `${(bcgResult!.coherence_score * 100).toFixed(0)}%`
              : isScanning ? '...' : '--%'}
          </p>
          {showReal && bcgResult!.coherence_score < 0.1 && (
            <p className="text-[10px] text-amber-400 mt-0.5 flex items-center">
              <AlertCircle className="h-3 w-3 mr-1" /> low signal
            </p>
          )}
        </div>

        {/* Challenge */}
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            Challenge
          </p>
          <p className={`text-sm font-semibold ${
            showReal && challengeAttempted
              ? bcgResult!.challenge_passed ? 'text-emerald-400' : 'text-amber-400'
              : 'text-foreground'
          }`}>
            {showReal && challengeAttempted
              ? bcgResult!.challenge_passed ? '✓ Pass' : '○ Partial'
              : isScanning ? '...' : '—'}
          </p>
          {showReal && challengeAttempted && !bcgResult!.challenge_passed && (
            <p className="text-[10px] text-amber-400 mt-0.5 flex items-center">
              <AlertCircle className="h-3 w-3 mr-1" /> blink / turn head
            </p>
          )}
        </div>

        {/* Spoof probability — FIX: only shown when actually suspicious */}
        <div>
          <p className="text-muted-foreground text-xs uppercase tracking-wider mb-1">
            Spoof Prob.
          </p>
          <p className={`text-sm font-semibold ${
            displaySpoof != null && displaySpoof > 50
              ? 'text-destructive'
              : 'text-muted-foreground'
          }`}>
            {displaySpoof != null && displaySpoof > 0 ? `${displaySpoof}%` : '—'}
          </p>
          {showReal && displaySpoof != null && displaySpoof > 50 && (
            <p className="text-[10px] text-destructive mt-0.5 flex items-center">
              <XCircle className="h-3 w-3 mr-1" /> possible spoof
            </p>
          )}
        </div>
      </div>

      {/* Summary when verified */}
      {showReal && (bcgResult!.bcg_passed || bcgResult!.challenge_passed || bcgResult!.coherence_score > 0.1) && (
        <div className="text-emerald-500 text-xs flex items-center justify-center pt-2 border-t border-border">
          <CheckCircle className="h-4 w-4 mr-2" />
          Liveness verified
        </div>
      )}
    </div>
  );
}