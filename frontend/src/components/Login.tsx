"use client";

import { useState, FormEvent } from "react";
import { Loader2, Lock, Mail } from "lucide-react";

interface LoginProps {
  onLoginSuccess: () => void;
}

export function Login({ onLoginSuccess }: LoginProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://api-dev.amlbot.rocks";
      const response = await fetch(`${apiUrl}/api/v2/auth/signin`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // Include cookies in request and response
        body: JSON.stringify({
          emailAddress: email,
          password: password,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Login failed: ${response.statusText}`);
      }

      const data = await response.json().catch(() => ({}));

      // Check if login was successful
      if (!data.success) {
        throw new Error("Login failed: Invalid response");
      }

      // Store user info if provided
      if (data.user) {
        localStorage.setItem("user", JSON.stringify(data.user));
      }

      // Mark as authenticated (cookies are automatically set by browser)
      localStorage.setItem("isAuthenticated", "true");

      onLoginSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during login");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-2xl">A</span>
          </div>
          <h1 className="text-2xl font-semibold text-white mb-2">AMLBot Tracer</h1>
          <p className="text-gray-400">Sign in to continue</p>
        </div>

        {/* Login Form */}
        <div className="bg-gray-900 rounded-xl p-8 border border-gray-800">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isLoading}
                  className="w-full bg-gray-800 text-white rounded-lg pl-10 pr-4 py-3 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  placeholder="Enter your email"
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={isLoading}
                  className="w-full bg-gray-800 text-white rounded-lg pl-10 pr-4 py-3 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  placeholder="Enter your password"
                />
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg p-3 text-sm">
                {error}
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !email || !password}
              className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                "Sign In"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
