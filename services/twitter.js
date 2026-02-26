const axios = require("axios");

const BASE_URL = "https://api.twitterapi.io";

// ── Axios client ────────────────────────────────────────────────────────────
function makeClient() {
  return axios.create({
    baseURL: BASE_URL,
    headers: { "X-API-Key": process.env.TWITTER_API_KEY },
    timeout: 20000,
  });
}

// ── Rate limiter (free tier: 1 req / 5.6 sec) — queue-safe ──────────────────
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
let _nextSlot = 0;
async function throttle() {
  const now  = Date.now();
  const wait = Math.max(0, _nextSlot - now);
  _nextSlot  = Math.max(now, _nextSlot) + 5600; // pre-reserve slot before any await
  if (wait) await delay(wait);
}

// ── Helper: date string N days ago for since: filter ─────────────────────────
function daysAgo(n) {
  const d = new Date(Date.now() - n * 86400000);
  const pad = (x) => String(x).padStart(2, "0");
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}_00:00:00_UTC`;
}

// ══════════════════════════════════════════════════════════════════════════════
// TWEET ENDPOINTS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * GET /twitter/tweet/advanced_search
 * Search tweets with full Twitter query syntax.
 * Supports: from:, since:, until:, AND/OR/-, lang:, -is:retweet, min_faves:, etc.
 */
async function searchTweets(query, count = 20, queryType = "Latest", cursor = "") {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/tweet/advanced_search", {
    params: { query, queryType, ...(cursor ? { cursor } : {}) },
  });
  const tweets = res.data.tweets || [];
  return { tweets: tweets.slice(0, count), hasNext: res.data.has_next_page, nextCursor: res.data.next_cursor };
}

/**
 * GET /twitter/tweet/advanced_search — last N days variant
 * Automatically adds since: filter for fresh results.
 */
async function searchRecentTweets(query, days = 7, count = 20, queryType = "Latest") {
  const since = daysAgo(days);
  const q = `${query} since:${since} lang:en`;
  return searchTweets(q, count, queryType);
}

/**
 * GET /twitter/tweet/advanced_search — top/popular variant
 * Uses Top queryType to find most-engaged tweets.
 */
async function searchTopTweets(query, count = 10) {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/tweet/advanced_search", {
    params: { query: `${query} lang:en`, queryType: "Top" },
  });
  return (res.data.tweets || []).slice(0, count);
}

/**
 * GET /twitter/community/get_tweets_from_all_community
 * Search tweets specifically inside Twitter Communities.
 */
async function searchCommunityTweets(query, count = 20, queryType = "Latest") {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/community/get_tweets_from_all_community", {
    params: { query, queryType },
  });
  return (res.data.tweets || []).slice(0, count);
}

// ══════════════════════════════════════════════════════════════════════════════
// TRENDS ENDPOINT
// ══════════════════════════════════════════════════════════════════════════════

/**
 * GET /twitter/trends
 * Fetch real Twitter trending topics by location (WOEID).
 * Common WOEIDs: 1=Worldwide, 23424977=USA, 44418=London, 23424848=India
 */
async function getTrends(woeid = 1, count = 30) {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/trends", {
    params: { woeid, count },
  });
  return res.data.trends || [];
}

// ══════════════════════════════════════════════════════════════════════════════
// USER ENDPOINTS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * GET /twitter/user/info
 * Get full profile for a single user by username.
 */
async function getUserInfo(userName) {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/user/info", { params: { userName } });
  return res.data.data || null;
}

/**
 * GET /twitter/user/search
 * Search for users by keyword — great for finding investors/VCs.
 */
async function searchUsers(query, count = 20) {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/user/search", { params: { query } });
  return (res.data.users || []).slice(0, count);
}

/**
 * GET /twitter/user/last_tweets
 * Get recent tweets from a specific user (timeline).
 */
async function getUserTweets(userName, count = 10, includeReplies = false) {
  await throttle();
  const client = makeClient();
  const res = await client.get("/twitter/user/last_tweets", {
    params: { userName, includeReplies },
  });
  return (res.data.tweets || []).slice(0, count);
}

/**
 * GET /twitter/user/mentions
 * Get tweets mentioning a specific user.
 */
async function getUserMentions(userName, days = 7, count = 20) {
  await throttle();
  const client = makeClient();
  const sinceTime = Math.floor((Date.now() - days * 86400000) / 1000);
  const res = await client.get("/twitter/user/mentions", {
    params: { userName, sinceTime },
  });
  return (res.data.tweets || []).slice(0, count);
}

// ══════════════════════════════════════════════════════════════════════════════
// ACCOUNT INFO
// ══════════════════════════════════════════════════════════════════════════════

/**
 * GET /oapi/my/info
 * Check remaining API credits.
 */
async function getCredits() {
  try {
    await throttle();
    const client = makeClient();
    const res = await client.get("/oapi/my/info");
    return res.data.recharge_credits ?? null;
  } catch {
    return null;
  }
}

module.exports = {
  searchTweets,
  searchRecentTweets,
  searchTopTweets,
  searchCommunityTweets,
  getTrends,
  getUserInfo,
  searchUsers,
  getUserTweets,
  getUserMentions,
  getCredits,
  daysAgo,
};
