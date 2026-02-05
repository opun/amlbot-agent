import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const api = process.env.NEXT_PUBLIC_API_URL ?? "http://aml-agent-api:8000";

    return [
      { source: "/api/chat", destination: `${api}/api/chat` },
      { source: "/api/trace", destination: `${api}/api/trace` },
      { source: "/api/trace/:path*", destination: `${api}/api/trace/:path*` },
      { source: "/docs", destination: `${api}/docs` },
      { source: "/redoc", destination: `${api}/redoc` },
      { source: "/openapi.yaml", destination: `${api}/openapi.yaml` },
    ];
  },
};

export default nextConfig;