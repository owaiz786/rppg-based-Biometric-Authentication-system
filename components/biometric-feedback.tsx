import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, X } from 'lucide-react';

// FIX: biometric-auth.tsx passes { result, message, onClose }
// but this component expected { state } — now aligned with the caller
interface BiometricFeedbackProps {
  result: 'success' | 'fail';
  message: string;
  onClose: () => void;
}

export function BiometricFeedback({ result, message, onClose }: BiometricFeedbackProps) {
  if (result === 'success') {
    return (
      <Alert className="bg-accent/10 border-accent mb-6 animate-in fade-in slide-in-from-bottom-4 duration-300 relative">
        <CheckCircle2 className="h-4 w-4 text-accent" />
        <AlertDescription className="text-accent font-medium pr-6">
          {message}
        </AlertDescription>
        <button onClick={onClose} className="absolute top-2 right-2 text-accent hover:opacity-70">
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }

  if (result === 'fail') {
    return (
      <Alert className="bg-destructive/10 border-destructive mb-6 animate-in fade-in slide-in-from-bottom-4 duration-300 relative">
        <XCircle className="h-4 w-4 text-destructive" />
        <AlertDescription className="text-destructive font-medium pr-6">
          {message}
        </AlertDescription>
        <button onClick={onClose} className="absolute top-2 right-2 text-destructive hover:opacity-70">
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }

  return null;
}
