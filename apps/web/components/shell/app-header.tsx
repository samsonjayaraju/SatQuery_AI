"use client";

import { useQuery } from "@tanstack/react-query";
import { Boxes, CircleDot, Clock3, Gauge, Map, Settings2 } from "lucide-react";
import Link from "next/link";
import { fetchHealth } from "@/lib/api";

const items = [
  { href: "/", label: "Workspace", icon: Map },
  { href: "/history", label: "History", icon: Clock3 },
  { href: "/benchmarks", label: "Benchmarks", icon: Gauge },
  { href: "/models", label: "Models", icon: Boxes },
];

export function AppHeader({ active }: { active: string }) {
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  return (
    <header className="topbar">
      <Link className="brand" href="/" aria-label="SatQuery AI workspace">
        <span className="brand-mark"><CircleDot size={18} /></span>
        <span><b>SatQuery</b><i>AI</i></span>
      </Link>
      <nav className="primary-nav" aria-label="Primary navigation">
        {items.map(({ href, label, icon: Icon }) => (
          <Link className={active === label.toLowerCase() ? "active" : ""} href={href} key={href}>
            <Icon size={15} /> {label}
          </Link>
        ))}
      </nav>
      <div className={`runtime-pill ${health.isError ? "offline" : ""}`}>
        <span /> {health.data ? `LOCAL · ${health.data.device.toUpperCase()}` : health.isError ? "API OFFLINE" : "CONNECTING"}
        <Link href="/settings" aria-label="Settings"><Settings2 size={14} /></Link>
      </div>
    </header>
  );
}
