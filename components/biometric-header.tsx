import { ShieldCheck } from 'lucide-react';

export function BiometricHeader() {
  return (
    <div className="flex flex-col items-center gap-2 mb-8">
      <div className="flex items-center gap-3 mb-2">
        <ShieldCheck className="w-8 h-8 text-accent" />
        <h1 className="text-3xl font-bold text-foreground">Secure Biometric Access</h1>
      </div>
      <p className="text-sm text-muted-foreground">Powered by rPPG Anti-Spoofing</p>
    </div>
  );
}
