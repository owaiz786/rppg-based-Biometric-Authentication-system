import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, X } from 'lucide-react';

interface BiometricFeedbackProps {
  result: 'success' | 'fail';
  message: string;
  onClose: () => void;
}

export function BiometricFeedback({ result, message, onClose }: BiometricFeedbackProps) {
  if (result === 'success') {
    return (
      <Alert className="bg-emerald-950 border-emerald-700">
        <CheckCircle2 className="h-5 w-5 text-emerald-400" />
        <AlertDescription className="ml-2 text-emerald-100">
          {message}
        </AlertDescription>
        <button 
          onClick={onClose} 
          className="absolute top-2 right-2 text-emerald-400 hover:opacity-70"
        >
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }
  
  if (result === 'fail') {
    return (
      <Alert className="bg-red-950 border-red-700">
        <XCircle className="h-5 w-5 text-red-400" />
        <AlertDescription className="ml-2 text-red-100">
          {message}
        </AlertDescription>
        <button 
          onClick={onClose} 
          className="absolute top-2 right-2 text-red-400 hover:opacity-70"
        >
          <X className="h-4 w-4" />
        </button>
      </Alert>
    );
  }
  
  return null;
}