"use client";

import { Button } from "@/components/ui/button";
import { ScanFace, UserPlus } from "lucide-react";

interface ControlsProps {
  isScanning: boolean;
  onEnroll: () => void;
  onLogin: () => void;
}

export default function BiometricControls({ isScanning, onEnroll, onLogin }: ControlsProps) {
  return (
    <div className="w-full flex justify-center space-x-4 mt-4">
      <Button 
        onClick={onEnroll} 
        disabled={isScanning}
        className={`w-full max-w-[160px] transition-all ${
          isScanning ? "bg-zinc-700 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-500"
        }`}
      >
        <UserPlus className="mr-2 h-5 w-5" />
        {isScanning ? "Scanning..." : "1. Enroll Face"}
      </Button>

      <Button 
        onClick={onLogin} 
        disabled={isScanning}
        className={`w-full max-w-[160px] transition-all ${
          isScanning ? "bg-zinc-700 cursor-not-allowed" : "bg-emerald-600 hover:bg-emerald-500"
        }`}
      >
        <ScanFace className="mr-2 h-5 w-5" />
        {isScanning ? "Scanning..." : "2. Login"}
      </Button>
    </div>
  );
}