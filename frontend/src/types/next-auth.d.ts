import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    userId?: string;
    userData?: {
      emailAddress: string;
      fullName: string;
      [key: string]: any;
    };
  }

  interface User {
    userId?: string;
    user?: any;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    userId?: string;
    userData?: any;
  }
}
