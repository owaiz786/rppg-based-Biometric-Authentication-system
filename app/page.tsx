import { BiometricAuth } from '@/components/biometric-auth';

export default function Home() {
  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-4">
      <BiometricAuth />
    </main>
  );
}
