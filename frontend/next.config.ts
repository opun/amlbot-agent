import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      // Proxy API routes to backend, but NOT /api/auth/* (NextAuth)
      {
        source: "/api/chat",
        destination: "http://localhost:8000/api/chat",
      },
      {
        source: "/api/trace",
        destination: "http://localhost:8000/api/trace",
      },
      {
        source: "/api/trace/:path*",
        destination: "http://localhost:8000/api/trace/:path*",
      },
      {
        source: "/docs",
        destination: "http://localhost:8000/docs",
      },
      {
        source: "/redoc",
        destination: "http://localhost:8000/redoc",
      },
      {
        source: "/openapi.yaml",
        destination: "http://localhost:8000/openapi.yaml",
      },
    ];
  },
};

export default nextConfig;
