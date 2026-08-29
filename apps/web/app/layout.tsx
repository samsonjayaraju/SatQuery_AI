import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "SatQuery AI · Remote-Sensing Analysis",
  description: "A local-first, sensor-aware vision-language workspace for satellite image analysis.",
  openGraph: {
    title: "SatQuery AI · Remote-Sensing Analysis",
    description: "A local-first, sensor-aware vision-language workspace for satellite image analysis.",
    type: "website",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "SatQuery AI sensor-aware remote-sensing analysis" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SatQuery AI · Remote-Sensing Analysis",
    description: "A local-first, sensor-aware vision-language workspace for satellite image analysis.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col"><Providers>{children}</Providers></body>
    </html>
  );
}
