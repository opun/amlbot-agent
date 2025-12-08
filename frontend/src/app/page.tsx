"use client";

import { useState } from "react";
import { Chat } from "@/components/Chat";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#0a0a0a]">
      <Chat />
    </main>
  );
}
