'use client';

import { useState, useEffect, useRef } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { BiometricHeader } from './biometric-header';
import BiometricWebcamArea from './biometric-webcam-area';
import { BiometricTelemetry } from './biometric-telemetry';
import BiometricControls from './biometric-controls';
import { BiometricFeedback } from './biometric-feedback';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

type ScanState = 'idle' | 'scanning' | 'success' | 'failed';
type AuthMode = 'enrollment' | 'login';

export function BiometricAuth() {
  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // UI States
  const [status, setStatus] = useState("Initializing camera...");
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<"success" | "fail" | null>(null);
  const [scanAction, setScanAction] = useState<"enroll" | "login" | null>(null);
  const [username, setUsername] = useState("");
  const [usernameError, setUsernameError] = useState("");

  // 1. Recording Logic
  const handleStartScan = (action: "enroll" | "login") => {
    // Validate username
    if (!username.trim()) {
      setUsernameError("Please enter a username");
      return;
    }
    setUsernameError("");
    
    if (!videoRef.current || !videoRef.current.srcObject) return;
    
    setScanAction(action);
    setIsScanning(true);
    setStatus(action === "enroll" ? "Hold still - Enrolling Face (5s)..." : "Hold still - Extracting rPPG (5s)...");
    chunksRef.current = []; // Reset previous recording

    const stream = videoRef.current.srcObject as MediaStream;
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      setIsScanning(false);
      setStatus("Analyzing spatial coherence & liveness...");
      const videoBlob = new Blob(chunksRef.current, { type: "video/webm" });
      sendVideoToBackend(videoBlob, action);
    };

    mediaRecorder.start();

    // Stop recording automatically after 5 seconds
    setTimeout(() => {
      if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
    }, 5000);
  };

  // 2. API Logic
  const sendVideoToBackend = async (videoBlob: Blob, action: "enroll" | "login") => {
    const formData = new FormData();
    formData.append("video", videoBlob, "rppg_sample.webm");
    formData.append("username", username.trim());

    try {
      const endpoint = action === "enroll" 
        ? "http://localhost:8000/api/auth/enroll-video" 
        : "http://localhost:8000/api/auth/login-video";

      console.log(`Sending to ${endpoint} for user: ${username}`);

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData
      });

      // Try to parse the response as JSON
      let data;
      const contentType = response.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        console.error("Non-JSON response:", text);
        throw new Error(`Server responded with ${response.status}: ${text}`);
      }

      if (!response.ok) {
        console.error("Server error details:", data);
        throw new Error(data.message || `Server responded with ${response.status}`);
      }

      if (data.success) {
        setScanResult("success");
        if (action === "enroll") {
          setStatus(`Enrollment Successful for ${username}. Face data stored.`);
        } else {
          setStatus(`Authentication Successful. Welcome ${username}! Liveness: ${data.is_real ? 'Verified' : 'Failed'}`);
        }
      } else {
        setScanResult("fail");
        setStatus(data.message || "Access Denied");
      }

    } catch (error: any) {
      console.error("Backend error:", error);
      setStatus(`Error: ${error.message}`);
      setScanResult("fail");
    }
  };

  return (
    <div className="flex flex-col items-center max-w-md mx-auto space-y-4 p-4">
      {/* Username Input */}
      <div className="w-full space-y-2">
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

      {/* Pass the state and refs down to the child components */}
      <BiometricWebcamArea videoRef={videoRef} status={status} setStatus={setStatus} />
      
      <BiometricTelemetry isScanning={isScanning} />
      
      <BiometricControls 
        isScanning={isScanning} 
        onEnroll={() => handleStartScan("enroll")}
        onLogin={() => handleStartScan("login")}
      />
      
      {/* Feedback component */}
      {scanResult && (
        <BiometricFeedback 
          result={scanResult} 
          message={status} 
          onClose={() => setScanResult(null)} 
        />
      )}
    </div>
  );
}