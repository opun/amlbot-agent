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
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://api-dev.amlbot.rocks";
          const response = await fetch(`${apiUrl}/api/v2/auth/signin`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              emailAddress: credentials.email,
              password: credentials.password,
            }),
          });

          if (!response.ok) {
            return null;
          }

          const data = await response.json();

          if (!data.success || !data.user) {
            return null;
          }

          // Extract cookies from response headers
          const cookies = response.headers.get("set-cookie");
          let userId = "";

          if (cookies) {
            // Parse userId from cookies
            const userIdMatch = cookies.match(/userId=([^;]+)/);
            if (userIdMatch) {
              userId = userIdMatch[1];
            }
          }

          // Return user object with userId from cookie
          return {
            id: userId || data.user.emailAddress,
            email: data.user.emailAddress,
            name: data.user.fullName,
            userId: userId,
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
        token.userId = (user as any).userId;
        token.userData = (user as any).user;
      }
      return token;
    },
    async session({ session, token }) {
      // Add userId to session
      if (token) {
        (session as any).userId = token.userId;
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
