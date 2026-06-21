import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Bankrupt-AI — przewidywanie bankructwa polskich firm",
  description:
    "Model uczenia maszynowego przewidujący bankructwo polskich firm na podstawie 64 wskaźników finansowych. Projekt zaliczeniowy ML.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body>{children}</body>
    </html>
  );
}
