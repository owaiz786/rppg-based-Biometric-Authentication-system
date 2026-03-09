import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle } from 'lucide-react';

interface BiometricFeedbackProps {
  state: 'idle' | 'scanning' | 'success' | 'failed';
}

export function BiometricFeedback({ state }: BiometricFeedbackProps) {
  if (state === 'success') {
    return (
      <Alert className="bg-accent/10 border-accent mb-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
        <CheckCircle2 className="h-4 w-4 text-accent" />
        <AlertDescription className="text-accent font-medium">
          Authentication Successful. Liveness verified.
        </AlertDescription>
      </Alert>
    );
  }

  if (state === 'failed') {
    return (
      <Alert className="bg-destructive/10 border-destructive mb-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
        <XCircle className="h-4 w-4 text-destructive" />
        <AlertDescription className="text-destructive font-medium">
          Access Denied. Synthetic/Replay attack detected.
        </AlertDescription>
      </Alert>
    );
  }

  return null;
}
