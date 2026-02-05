import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Credentials({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          const apiUrl = process.env.AMLBOT_API_URL || "https://api-dev.amlbot.rocks";
          const response = await fetch(`${apiUrl}/api/v2/auth/signin`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            credentials: "include",
            body: JSON.stringify({
              emailAddress: credentials.email,
              password: credentials.password,
            }),
          });

          if (!response.ok) {
            return null;
          }

          const data = await response.json().catch(() => null);

          if (!data?.success || !data?.user) {
            return null;
          }

          const userIdCandidates = [
            data?.user?.id,
            data?.user?.userId,
            data?.user?.user_id,
            data?.user?.uid,
            data?.userId,
            data?.user_id,
            data?.id,
          ]
            .map((value: unknown) => (typeof value === "string" ? value.trim() : ""))
            .filter(Boolean);

          let userId = userIdCandidates[0] || "";

          if (!userId) {
            const headerAny = response.headers as unknown as {
              getSetCookie?: () => string[];
              raw?: () => Record<string, string[]>;
            };

            const setCookieHeaders =
              headerAny.getSetCookie?.() ||
              headerAny.raw?.()["set-cookie"] ||
              (response.headers.get("set-cookie") ? [response.headers.get("set-cookie") as string] : []);

            if (setCookieHeaders?.length) {
              for (const cookieHeader of setCookieHeaders) {
                const userIdMatch = cookieHeader.match(/(?:^|;\s*)userId=([^;]+)/);
                if (userIdMatch?.[1]) {
                  userId = userIdMatch[1];
                  break;
                }
              }
            }
          }

          if (!userId) {
            console.warn("Auth warning: userId not found in response; falling back to email.");
            userId = data.user.emailAddress;
          }

          // Return user object with userId from cookie
          return {
            id: userId || data.user.emailAddress,
            email: data.user.emailAddress,
            name: data.user.fullName,
            userId,
            user: data.user,
          };
        } catch (error) {
          console.error("Auth error:", error);
          return null;
        }
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      // Add userId to token on first sign in
      if (user) {
        token.userId = (user as any).userId || (user as any).id || token.sub;
        token.userData = (user as any).user;
      }
      if (!token.userId && token.sub) {
        token.userId = token.sub;
      }
      return token;
    },
    async session({ session, token }) {
      // Add userId to session
      if (token) {
        (session as any).userId = token.userId || token.sub;
        (session as any).userData = token.userData;
      }
      return session;
    },
  },
  pages: {
    signIn: "/login",
  },
  session: {
    strategy: "jwt",
  },
  trustHost: true,
});
