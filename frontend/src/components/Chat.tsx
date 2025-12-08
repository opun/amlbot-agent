"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import { Send, Loader2, ExternalLink, RotateCcw, CheckCircle2, AlertCircle, Edit3, ArrowRight, StopCircle, Hash } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface ContinuationOption {
  address: string;
  path_id: string;
  last_amount: number;
  asset: string;
  chain: string;
  last_tx_hash: string;
  role: string;
  risk_score: number | null;
  stop_reason: string;
  description: string;
}

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  status?: "pending" | "streaming" | "complete" | "error";
  report?: TraceReport | null;
  traceUrl?: string;
  collectedInfo?: Record<string, any>;
  responseType?: string;
  continuationOptions?: ContinuationOption[];
  canContinue?: boolean;
}

interface TraceReport {
  summary_text: string;
  ascii_tree: string;
  mermaid: string;
  paths: any[];
  entities: any[];
}

const WELCOME_MESSAGE = `# 👋 Welcome to AMLBot Crypto Tracer

I can help you trace cryptocurrency fund movements across blockchains.

**To get started, describe your case:**
- Provide the **victim wallet address** or **theft transaction hash**
- Specify the **blockchain** (ETH, TRX, BTC, etc.)
- Include any additional context like approximate date or asset type

**Example:**
> "I need to trace stolen USDT from wallet 0x1234...abcd on Ethereum. The theft occurred around December 1st, 2024."

I'll collect the necessary information step by step before starting the trace.
`;

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: WELCOME_MESSAGE,
      timestamp: new Date(),
      status: "complete",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
      status: "complete",
    };

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "assistant",
      content: "",
      timestamp: new Date(),
      status: "pending",
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // Create abort controller for timeout (5 minutes for long traces)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

      // Call backend directly to avoid Next.js proxy timeout issues
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      // Check for session ID in response headers
      const newSessionId = response.headers.get("X-Session-Id");
      if (newSessionId) {
        setSessionId(newSessionId);
      }

      const contentType = response.headers.get("content-type");

      if (contentType?.includes("text/event-stream")) {
        // Handle streaming response (trace running)
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error("No reader available");

        let statusMessages: string[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n").filter((line) => line.trim());

          for (const line of lines) {
            try {
              const data = JSON.parse(line);

              if (data.type === "status") {
                statusMessages.push(data.message);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessage.id
                      ? {
                          ...m,
                          content: statusMessages.map((s) => `⏳ ${s}`).join("\n"),
                          status: "streaming",
                        }
                      : m
                  )
                );
              } else if (data.type === "trace_started") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessage.id
                      ? { ...m, traceUrl: data.trace_url }
                      : m
                  )
                );
              } else if (data.type === "result") {
                const hasOptions = data.continuation_options && data.continuation_options.length > 0;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessage.id
                      ? {
                          ...m,
                          content: formatReport(data.report, hasOptions),
                          report: data.report,
                          status: "complete",
                          responseType: hasOptions ? "continuation" : "result",
                          continuationOptions: hasOptions ? data.continuation_options : undefined,
                          canContinue: data.can_continue,
                        }
                      : m
                  )
                );
                // Reset session when trace is complete (no continuation needed)
                if (!hasOptions) {
                  setSessionId(null);
                }
              } else if (data.type === "error") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessage.id
                      ? {
                          ...m,
                          content: `❌ Error: ${data.message}`,
                          status: "error",
                        }
                      : m
                  )
                );
              }
            } catch {
              // Skip non-JSON lines
            }
          }
        }
      } else {
        // Handle JSON response (conversation flow)
        const data = await response.json();

        // Update session ID
        if (data.session_id) {
          setSessionId(data.session_id);
        }

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessage.id
              ? {
                  ...m,
                  content: data.message,
                  status: "complete",
                  responseType: data.type,
                  collectedInfo: data.collected_info,
                  continuationOptions: data.continuation_options,
                }
              : m
          )
        );
      }
    } catch (error) {
      console.error("Chat error:", error);
      let errorMessage = "Unknown error";
      if (error instanceof Error) {
        if (error.name === "AbortError") {
          errorMessage = "Request timed out. The trace may be taking longer than expected.";
        } else {
          errorMessage = error.message;
        }
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessage.id
            ? {
                ...m,
                content: `❌ Error: ${errorMessage}`,
                status: "error",
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const formatReport = (report: TraceReport, hasOptions: boolean = false): string => {
    // Only show "Paused" if there are options that need user decision (bridges, etc.)
    const title = hasOptions
      ? "## 🔄 Trace Complete - Further Options Available"
      : "## ✅ Trace Complete";
    return `${title}\n\n${report.summary_text}`;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: WELCOME_MESSAGE,
        timestamp: new Date(),
        status: "complete",
      },
    ]);
    setSessionId(null);
  };

  const handleQuickAction = (action: string) => {
    setInput(action);
    // Focus input and submit
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
            <span className="text-white font-bold text-lg">A</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">AMLBot Tracer</h1>
            <p className="text-sm text-gray-400">
              {sessionId ? "Session active" : "Crypto fund tracing assistant"}
            </p>
          </div>
        </div>
        <button
          onClick={clearChat}
          className="p-2 rounded-lg hover:bg-gray-800 transition-colors text-gray-400 hover:text-white"
          title="New chat"
        >
          <RotateCcw className="w-5 h-5" />
        </button>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onQuickAction={handleQuickAction}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-gray-800">
        <form onSubmit={handleSubmit} className="relative">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your case, provide an address, or answer questions..."
            className="w-full bg-gray-900 text-white rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500/50 placeholder-gray-500"
            rows={3}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-3 bottom-3 p-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 text-white animate-spin" />
            ) : (
              <Send className="w-5 h-5 text-white" />
            )}
          </button>
        </form>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  onQuickAction,
}: {
  message: Message;
  onQuickAction: (action: string) => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] ${
          isUser
            ? "bg-emerald-600 text-white rounded-2xl rounded-br-md px-4 py-3"
            : "bg-gray-900 text-gray-100 rounded-2xl rounded-bl-md px-4 py-3"
        }`}
      >
        {/* Status indicator */}
        {message.status === "pending" && (
          <div className="flex items-center gap-2 text-gray-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Thinking...</span>
          </div>
        )}

        {message.status === "streaming" && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 mb-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">Running trace...</span>
            </div>
            <div className="text-sm text-gray-300 whitespace-pre-wrap">
              {message.content}
            </div>
          </div>
        )}

        {(message.status === "complete" || message.status === "error") && (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Quick actions for confirming state */}
        {message.responseType === "confirming" && message.status === "complete" && (
          <div className="flex gap-2 mt-4 pt-3 border-t border-gray-700">
            <button
              onClick={() => onQuickAction("yes")}
              className="flex items-center gap-1 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm font-medium transition-colors"
            >
              <CheckCircle2 className="w-4 h-4" />
              Start Trace
            </button>
            <button
              onClick={() => onQuickAction("edit")}
              className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors"
            >
              <Edit3 className="w-4 h-4" />
              Edit
            </button>
          </div>
        )}

        {/* Continuation options - only shown for bridges or when user decision needed */}
        {message.responseType === "continuation" && message.continuationOptions && message.continuationOptions.length > 0 && (
          <div className="mt-4 pt-3 border-t border-gray-700">
            <p className="text-sm text-gray-400 mb-3">
              Optional: Continue tracing (e.g., cross-chain bridge):
            </p>
            <div className="space-y-2">
              {message.continuationOptions.map((opt, idx) => (
                <button
                  key={opt.address}
                  onClick={() => onQuickAction(`continue ${idx + 1}`)}
                  className="w-full flex items-center justify-between p-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-left transition-colors group"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-emerald-400 font-medium">{idx + 1}.</span>
                      <code className="text-xs text-gray-300">
                        {opt.address.slice(0, 10)}...{opt.address.slice(-6)}
                      </code>
                      {opt.risk_score && opt.risk_score > 0.5 && (
                        <span className="text-xs px-1.5 py-0.5 bg-red-500/20 text-red-400 rounded">
                          Risk: {(opt.risk_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {opt.last_amount.toLocaleString()} {opt.asset} • {opt.role}
                    </div>
                  </div>
                  <ArrowRight className="w-4 h-4 text-gray-500 group-hover:text-emerald-400 transition-colors" />
                </button>
              ))}
            </div>
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => onQuickAction("done")}
                className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm font-medium transition-colors"
              >
                <StopCircle className="w-4 h-4" />
                Done
              </button>
              <div className="flex-1 text-xs text-gray-500 flex items-center">
                <Hash className="w-3 h-3 mr-1" />
                Or paste a tx hash to continue from
              </div>
            </div>
          </div>
        )}

        {/* Trace URL link */}
        {message.traceUrl && (
          <a
            href={message.traceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300 mt-2"
          >
            <ExternalLink className="w-3 h-3" />
            View trace on OpenAI
          </a>
        )}

        {/* ASCII Tree for reports */}
        {message.report?.ascii_tree && (
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-emerald-400 hover:text-emerald-300">
              📊 View Trace Tree
            </summary>
            <div className="ascii-tree mt-2 text-xs">
              {message.report.ascii_tree}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
