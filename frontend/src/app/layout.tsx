import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AMLBot Crypto Tracer",
  description: "AI-powered crypto fund tracing assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
