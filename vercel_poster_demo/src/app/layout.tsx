import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Condition-Adaptive UAV Detection | ISEF Demo",
  description: "Visual-first offline demo for multimodal UAV detection and adaptive fusion.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
