import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin", "latin-ext"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Bankrupt-AI — przewidywanie bankructwa polskich firm",
  description:
    "Model uczenia maszynowego przewidujący bankructwo polskich firm na podstawie 64 wskaźników finansowych. Projekt zaliczeniowy ML.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
