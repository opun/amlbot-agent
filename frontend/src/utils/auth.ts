/**
 * Authentication utility functions
 */

/**
 * Get a cookie value by name
 */
const getCookie = (name: string): string | null => {
  if (typeof document === "undefined") return null;
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop()?.split(";").shift() || null;
  return null;
};

/**
 * Check if user is authenticated by verifying cookies exist
 */
export const isAuthenticated = (): boolean => {
  if (typeof window === "undefined") return false;
  // Check for userId cookie (most important for API calls)
  const userId = getCookie("userId");
  const sailsSid = getCookie("sails.sid");
  // Also check localStorage as fallback
  const localStorageAuth = localStorage.getItem("isAuthenticated") === "true";

  // User is authenticated if they have cookies OR localStorage flag
  return !!(userId || sailsSid) || localStorageAuth;
};

export const getAuthToken = (): string | null => {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("authToken");
};

/**
 * Get userId from cookie
 */
export const getUserId = (): string | null => {
  return getCookie("userId");
};

export const getUser = (): any | null => {
  if (typeof window === "undefined") return null;
  const userStr = localStorage.getItem("user");
  return userStr ? JSON.parse(userStr) : null;
};

export const logout = (): void => {
  if (typeof window === "undefined") return;
  localStorage.removeItem("isAuthenticated");
  localStorage.removeItem("authToken");
  localStorage.removeItem("user");

  // Note: Cookies will be cleared by the server on logout
  // If you have a logout endpoint, call it here to clear cookies server-side
};
