"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, BookOpen, ExternalLink, Loader2 } from "lucide-react";

export default function IntegrationsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [view, setView] = useState<"swagger" | "redoc">("swagger");

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
    }
  }, [status, router]);

  if (status === "loading") {
    return (
      <main className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="flex items-center gap-2 text-gray-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          Loading...
        </div>
      </main>
    );
  }

  if (!session) {
    return null;
  }

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors">
            <span className="p-2 rounded-lg bg-gray-900 border border-gray-800">
              <ArrowLeft className="w-4 h-4" />
            </span>
            <span className="text-sm">Back to Tracer</span>
          </Link>
        </div>
        <div className="flex items-center gap-2">
          <a
            href="/docs"
            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-xs text-emerald-300 hover:text-emerald-200 hover:border-emerald-400/60 transition-colors"
          >
            <BookOpen className="w-4 h-4" />
            Developer Docs
          </a>
        </div>
      </header>

      <section className="max-w-6xl mx-auto px-6 py-8">
        <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-[#0f172a] via-[#0b1220] to-[#0a0a0a] p-6">
          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-emerald-400/80">
                Integrations
              </p>
              <h1 className="text-2xl sm:text-3xl font-semibold mt-2">
                Embed-ready API documentation
              </h1>
              <p className="text-sm text-gray-400 mt-2 max-w-2xl">
                Use Swagger UI for interactive testing or Redoc for clean, shareable
                documentation. Both are proxied through the frontend so teams can
                access them from a single origin.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <a
                href="/docs"
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-800 bg-gray-900/70 text-xs text-gray-200 hover:text-white hover:border-gray-700 transition-colors"
              >
                Swagger UI
                <ExternalLink className="w-3 h-3" />
              </a>
              <a
                href="/redoc"
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-800 bg-gray-900/70 text-xs text-gray-200 hover:text-white hover:border-gray-700 transition-colors"
              >
                Redoc
                <ExternalLink className="w-3 h-3" />
              </a>
              <a
                href="/openapi.yaml"
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-800 bg-gray-900/70 text-xs text-gray-200 hover:text-white hover:border-gray-700 transition-colors"
              >
                OpenAPI YAML
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-2">
          <button
            onClick={() => setView("swagger")}
            className={`px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
              view === "swagger"
                ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-200"
                : "bg-gray-900/60 border-gray-800 text-gray-300 hover:text-white hover:border-gray-700"
            }`}
          >
            Swagger UI
          </button>
          <button
            onClick={() => setView("redoc")}
            className={`px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
              view === "redoc"
                ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-200"
                : "bg-gray-900/60 border-gray-800 text-gray-300 hover:text-white hover:border-gray-700"
            }`}
          >
            Redoc
          </button>
          <div className="text-xs text-gray-500">
            {view === "swagger" ? "Interactive testing" : "Clean reference view"}
          </div>
        </div>

        <div className="mt-4 rounded-2xl border border-gray-800 bg-gray-950/60 overflow-hidden">
          <iframe
            key={view}
            src={view === "swagger" ? "/docs" : "/redoc"}
            title={view === "swagger" ? "Swagger UI" : "Redoc"}
            className="w-full h-[75vh] bg-black"
          />
        </div>
      </section>
    </main>
  );
}
