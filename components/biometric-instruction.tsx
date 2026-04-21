import { useEffect, useState } from 'react';

interface BiometricInstructionProps {
  isScanning: boolean;
}

const INSTRUCTION_SEQUENCE = [
  'Align Face',
  'Hold Still - Extracting rPPG (5s)',
  'Action Required: Blink Twice',
];

export function BiometricInstruction({ isScanning }: BiometricInstructionProps) {
  const [instructionIndex, setInstructionIndex] = useState(0);

  useEffect(() => {
    if (!isScanning) {
      setInstructionIndex(0);
      return;
    }

    const intervals = [1000, 3000, 2000];
    let currentIndex = 0;

    const timer = setInterval(() => {
      currentIndex++;
      if (currentIndex < INSTRUCTION_SEQUENCE.length) {
        setInstructionIndex(currentIndex);
      } else {
        clearInterval(timer);
      }
    }, intervals[currentIndex]);

    return () => clearInterval(timer);
  }, [isScanning]);

  const instruction = INSTRUCTION_SEQUENCE[instructionIndex];

  return (
    <div
      className={`absolute inset-0 flex items-center justify-center transition-opacity duration-300 ${
        isScanning ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`}
    >
      <div className="bg-black/70 px-4 py-2 rounded-lg">
        <p className="text-lg font-semibold text-accent text-center whitespace-nowrap">
          {instruction}
        </p>
      </div>
    </div>
  );
}
