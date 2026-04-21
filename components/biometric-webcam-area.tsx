"use client";

import { useEffect, RefObject } from "react";

interface WebcamProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  status: string;
  setStatus: (status: string) => void;
}

export default function BiometricWebcamArea({ videoRef, status, setStatus }: WebcamProps) {
  
  // Turn on the camera when this component loads
  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30 } },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatus("Align face and click Start Scan");
      } catch (err) {
        setStatus("Camera access denied.");
        console.error(err);
      }
    }
    
    startCamera();

    // Cleanup: Turn off camera when user leaves the page
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [videoRef, setStatus]);

  return (
    <div className="relative w-full aspect-video bg-zinc-900 rounded-lg overflow-hidden border-2 border-zinc-800">
      {/* The actual video feed */}
      <video
        ref={videoRef as any}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover scale-x-[-1]" 
        /* scale-x-[-1] mirrors the video so it feels like a mirror */
      />
      
      {/* Dashed alignment box overlay */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-48 h-64 border-2 border-dashed border-zinc-500 rounded-full opacity-50" />
      </div>

      {/* Status Text Overlay */}
      <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none">
        <span className="bg-black/70 text-emerald-400 text-sm px-3 py-1 rounded-full font-mono">
          {status}
        </span>
      </div>
    </div>
  );
}