import type { ReactNode } from "react";
import { AppHeader } from "./app-header";

export function SectionPage({ active, eyebrow, title, description, children }: { active: string; eyebrow: string; title: string; description: string; children: ReactNode }) {
  return (
    <main className="section-page">
      <AppHeader active={active} />
      <header className="section-hero"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></header>
      <section className="section-content">{children}</section>
    </main>
  );
}
