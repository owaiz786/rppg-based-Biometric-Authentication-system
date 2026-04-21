'use client';
import { useState, useRef } from 'react';
import BiometricWebcamArea from './biometric-webcam-area';
import { BiometricTelemetry } from './biometric-telemetry';
import BiometricControls from './biometric-controls';
import { BiometricFeedback } from './biometric-feedback';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const ENROLL_DURATION = 5000;

// Human-readable labels and timing for each challenge type
const CHALLENGE_META: Record<string, { label: string; duration: number }> = {
  blink:     { label: "👁  Blink your eyes clearly",  duration: 3000 },
  head_turn: { label: "↩↪  Turn head LEFT then RIGHT", duration: 4000 },
  smile:     { label: "😊  Smile broadly",             duration: 3000 },
  look_up:   { label: "⬆️  Look UP briefly",            duration: 3000 },
};

// Build the step sequence dynamically from the server-issued challenge list.
// Always start with a "hold still" warm-up and end with a BCG capture.
function buildSteps(challenges: string[]) {
  return [
    { label: "Hold still — scanning face...", duration: 2000 },
    ...challenges.map((c) => CHALLENGE_META[c] ?? { label: c, duration: 3000 }),
    { label: "✓  Hold still — BCG capture...", duration: 2000 },
  ];
}

type BcgResult = {
  bcg_hr_bpm:       number;
  rppg_hr_bpm:      number;
  bcg_signal_power: number;
  bcg_passed:       boolean;
  freq_match:       boolean;
  coherence_score:  number;
  challenge_passed: boolean;
};

export function BiometricAuth() {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const recorderRef   = useRef<MediaRecorder | null>(null);
  const chunksRef     = useRef<BlobPart[]>([]);
  const stepTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);

  const [status,           setStatus]           = useState("Initializing camera...");
  const [isScanning,       setIsScanning]       = useState(false);
  const [scanResult,       setScanResult]       = useState<"success" | "fail" | null>(null);
  const [username,         setUsername]         = useState("");
  const [usernameError,    setUsernameError]    = useState("");
  const [challengeStep,    setChallengeStep]    = useState<string | null>(null);
  const [bcgResult,        setBcgResult]        = useState<BcgResult | null>(null);
  // Pending challenges shown to user before recording starts
  const [pendingChallenges, setPendingChallenges] = useState<string[] | null>(null);

  // Token issued by the server for this login session
  const challengeTokenRef   = useRef<string>("");
  const activeChallengesRef = useRef<string[]>([]);

  // ── Challenge instruction ticker ─────────────────────────────────────────
  const runChallengeInstructions = (challenges: string[]) => {
    const steps = buildSteps(challenges);
    let elapsed = 0;
    setChallengeStep(steps[0].label);

    steps.forEach((step) => {
      const t = setTimeout(() => {
        setChallengeStep(step.label);
        setStatus(step.label);
      }, elapsed);
      stepTimersRef.current.push(t);
      elapsed += step.duration;
    });
  };

  const clearTimers = () => {
    stepTimersRef.current.forEach(clearTimeout);
    stepTimersRef.current = [];
    setChallengeStep(null);
  };

  // ── Fetch a random challenge token from the server ────────────────────────
  const fetchChallengeToken = async (): Promise<boolean> => {
    try {
      setStatus("Generating random challenges...");
      const res  = await fetch("http://localhost:8000/api/auth/challenge-token");
      const data = await res.json();

      if (!data.token || !data.challenges) {
        setStatus("Failed to get challenge token — try again.");
        return false;
      }

      challengeTokenRef.current   = data.token;
      activeChallengesRef.current = data.challenges;
      setPendingChallenges(data.challenges);
      return true;
    } catch (err: any) {
      setStatus(`Server error: ${err.message}`);
      return false;
    }
  };

  // ── Start recording ───────────────────────────────────────────────────────
  const handleStartScan = async (action: "enroll" | "login") => {
    if (!username.trim()) {
      setUsernameError("Please enter a username");
      return;
    }
    setUsernameError("");

    if (!videoRef.current?.srcObject) return;

    // For login: fetch a random challenge token first.
    // The token determines which challenges the user must perform.
    // A pre-recorded video cannot know these challenges in advance.
    if (action === "login") {
      const ok = await fetchChallengeToken();
      if (!ok) return;
    }

    setIsScanning(true);
    setScanResult(null);
    setBcgResult(null);
    setPendingChallenges(null);
    chunksRef.current = [];

    if (action === "enroll") {
      setStatus("Hold still — enrolling face (5s)...");
    } else {
      setStatus("Preparing...");
      runChallengeInstructions(activeChallengesRef.current);
    }

    const steps    = action === "login" ? buildSteps(activeChallengesRef.current) : [];
    const duration = action === "login"
      ? steps.reduce((s, c) => s + c.duration, 0)
      : ENROLL_DURATION;

    const stream   = videoRef.current.srcObject as MediaStream;
    const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    recorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data?.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      setIsScanning(false);
      clearTimers();
      setStatus("Analysing rPPG + Challenges + BCG...");
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      sendToBackend(blob, action);
    };

    recorder.start();
    setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, duration);
  };

  // ── Send to backend ───────────────────────────────────────────────────────
  const sendToBackend = async (blob: Blob, action: "enroll" | "login") => {
    const formData = new FormData();
    formData.append("video", blob, "rppg_sample.webm");
    formData.append("username", username.trim());

    // For login: always send the challenge token so the backend can verify
    // that the video corresponds to the specific challenges that were issued.
    if (action === "login") {
      formData.append("challenge_token", challengeTokenRef.current);
    }

    try {
      const endpoint = action === "enroll"
        ? "http://localhost:8000/api/auth/enroll-video"
        : "http://localhost:8000/api/auth/login-video";

      const res  = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json().catch(async () => {
        const txt = await res.text();
        throw new Error(`Non-JSON response (${res.status}): ${txt}`);
      });

      if (data.success) {
        setScanResult("success");

        if (action === "login") {
          setBcgResult({
            bcg_hr_bpm:       data.bcg_hr_bpm       ?? 0,
            rppg_hr_bpm:      data.rppg_hr_bpm       ?? 0,
            bcg_signal_power: data.bcg_signal_power  ?? 0,
            bcg_passed:       data.bcg_passed        ?? false,
            freq_match:       data.bcg_freq_match    ?? false,
            coherence_score:  data.coherence_score   ?? 0,
            challenge_passed: data.challenge_passed  ?? false,
          });

          const bcgInfo = data.bcg_hr_bpm ? ` | BCG: ${data.bcg_hr_bpm} BPM` : '';
          setStatus(`Welcome ${username}! Liveness verified.${bcgInfo}`);
        } else {
          setStatus(`Enrollment successful for ${username}. Face data stored.`);
        }
      } else {
        setScanResult("fail");
        setStatus(data.message || data.spoof_reason || "Access Denied");
      }

    } catch (err: any) {
      console.error("Backend error:", err);
      setStatus(`Error: ${err.message}`);
      setScanResult("fail");
    }
  };

  return (
    <div className="space-y-6">
      {/* Username */}
      <div className="space-y-2">
        <Label htmlFor="username">Username</Label>
        <Input
          id="username"
          type="text"
          placeholder="Enter your username"
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            setUsernameError("");
          }}
          disabled={isScanning}
          className={usernameError ? "border-red-500" : ""}
        />
        {usernameError && (
          <p className="text-sm text-red-500">{usernameError}</p>
        )}
      </div>

      <BiometricWebcamArea
        videoRef={videoRef}
        status={status}
        setStatus={setStatus}
      />

      {/* Pre-scan challenge preview — shown after token fetch, before recording */}
      {pendingChallenges && !isScanning && (
        <div className="w-full bg-blue-950 border border-blue-700 rounded-lg px-4 py-3">
          <p className="text-blue-300 font-mono text-xs font-semibold mb-2 uppercase tracking-wider">
            🎲 Random Challenges Assigned
          </p>
          <ol className="list-decimal list-inside space-y-1">
            {pendingChallenges.map((c) => (
              <li key={c} className="text-blue-100 text-sm font-mono">
                {CHALLENGE_META[c]?.label ?? c}
              </li>
            ))}
          </ol>
          <p className="text-xs text-blue-400 mt-2">
            Recording will start when you click Login. Complete all steps in order.
          </p>
        </div>
      )}

      {/* Live challenge step banner */}
      {isScanning && challengeStep && (
        <div className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-4 py-3 text-center">
          <p className="text-emerald-400 font-mono text-sm font-semibold animate-pulse">
            {challengeStep}
          </p>
          <p className="text-xs text-zinc-400 mt-1">
            Follow the prompts — camera is recording rPPG, challenges & BCG
          </p>
        </div>
      )}

      <BiometricTelemetry isScanning={isScanning} bcgResult={bcgResult} />

      <BiometricControls
        isScanning={isScanning}
        onEnroll={() => handleStartScan("enroll")}
        onLogin={() => handleStartScan("login")}
      />

      {scanResult && (
        <BiometricFeedback
          result={scanResult}
          message={status}
          onClose={() => {
            setScanResult(null);
            setBcgResult(null);
          }}
        />
      )}
    </div>
  );
}