import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Laureate Engine",
  description: "Nobel Prize-powered market regime detection & crisis anticipation dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-text font-mono antialiased">
        {children}
      </body>
    </html>
  );
}
