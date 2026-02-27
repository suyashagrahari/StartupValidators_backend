require("dotenv").config({ path: require("path").join(__dirname, "../.env") });

const { StateGraph, Annotation, START, END } = require("@langchain/langgraph");
const { ChatOpenAI } = require("@langchain/openai");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { tavily } = require("@tavily/core");

const {
  searchTweets, searchRecentTweets, searchTopTweets,
  searchCommunityTweets, getTrends, searchUsers,
  getUserInfo, getUserTweets, getCredits,
} = require("../services/twitter");
const { analyzeSentiment, extractTrendingTopics, filterAdopters, filterFunders } = require("../services/analyzer");

// ── Gemini client ──────────────────────────────────────────────────────────────
const genAI  = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const gemini = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// ── Tavily client ──────────────────────────────────────────────────────────────
const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

// ── OpenAI ────────────────────────────────────────────────────────────────────
function makeLLM() {
  return new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.3, apiKey: process.env.OPENAI_API_KEY });
}

// ── State ─────────────────────────────────────────────────────────────────────
const State = Annotation.Root({
  idea:          Annotation({ reducer: (_, y) => y, default: () => "" }),
  description:   Annotation({ reducer: (_, y) => y, default: () => "" }),
  queries:       Annotation({ reducer: (_, y) => y, default: () => null }),
  trendsData:    Annotation({ reducer: (_, y) => y, default: () => null }),
  demandData:    Annotation({ reducer: (_, y) => y, default: () => null }),
  adopterData:   Annotation({ reducer: (_, y) => y, default: () => null }),
  funderData:    Annotation({ reducer: (_, y) => y, default: () => null }),
  communityData: Annotation({ reducer: (_, y) => y, default: () => null }),
  tavilyData:    Annotation({ reducer: (_, y) => y, default: () => null }),
  verdict:       Annotation({ reducer: (_, y) => y, default: () => null }),
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function ts() { return new Date().toTimeString().slice(0, 8); }

function tweetSummary(tweets, max = 12) {
  return tweets.slice(0, max)
    .map(t => `@${t.author?.userName || "?"} [❤${t.likeCount || 0}|🔁${t.retweetCount || 0}]: ${t.text}`)
    .join("\n");
}

function userSummary(users, max = 8) {
  return users.slice(0, max)
    .map(u => `@${u.userName} (${u.followers || 0} followers) — ${(u.description || "").slice(0, 80)}`)
    .join("\n");
}

function fallbackQueries(idea) {
  return {
    trend_scan: `${idea} -is:retweet lang:en`,
    demand_recent: `${idea} lang:en`,
    demand_top: `${idea} lang:en`,
    adopter_pain: `${idea} (wish OR struggling OR frustrated OR "looking for" OR "need a") lang:en -is:retweet`,
    investor_search: `${idea} investor`,
    competitor_search: `${idea} alternative OR competitor lang:en`,
    community_query: idea,
  };
}

// ── Graph factory ─────────────────────────────────────────────────────────────
function createResearchGraph(emit) {
  const llm = makeLLM();
  let tavilyPrefetchPromise = null;

  function wrapNode(name, fn, fallbackFactory) {
    return async (state) => {
      try {
        return await fn(state);
      } catch (err) {
        emit({ type: "warn", ts: ts(), msg: `  ⚠ ${name} failed: ${err.message?.slice(0, 80) || "unknown error"} — continuing with partial data` });
        return fallbackFactory(state);
      }
    };
  }

  async function runSingleTavilyResearch(idea) {
    // tvly.research() is async (returns only a requestId when stream:false).
    // Instead, run parallel tvly.search() calls which return sources immediately.
    const searchQueries = [
      `${idea} market size TAM startup landscape 2025`,
      `${idea} competitors alternatives funding investors`,
      `${idea} customer pain points use cases target market`,
      `${idea} growth trends industry report`,
    ];

    let allSources = [];
    const answers = [];

    emit({ type: "info", ts: ts(), msg: `  ⚡ Running ${searchQueries.length} parallel Tavily web searches...` });

    const settled = await Promise.allSettled(
      searchQueries.map((q) =>
        Promise.race([
          tvly.search(q, { maxResults: 7, includeAnswer: true, searchDepth: "advanced" }),
          new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 25000)),
        ])
      )
    );

    for (let i = 0; i < settled.length; i++) {
      const r = settled[i];
      if (r.status === "fulfilled" && r.value) {
        const results = r.value.results || [];
        allSources.push(...results);
        if (r.value.answer) answers.push(`**${searchQueries[i]}**\n${r.value.answer}`);
        emit({ type: "info", ts: ts(), msg: `  ✓ Search ${i + 1}: ${results.length} sources` });
      } else if (r.status === "rejected") {
        emit({ type: "warn", ts: ts(), msg: `  ⚠ Search ${i + 1} failed: ${r.reason?.message?.slice(0, 50)}` });
      }
    }

    // Deduplicate by URL
    const seen = new Set();
    const uniqueSources = allSources.filter((s) => {
      if (!s.url || seen.has(s.url)) return false;
      seen.add(s.url);
      return true;
    });

    // Build summary from search answers
    const webSummary = answers.join("\n\n") || "";

    emit({ type: "info", ts: ts(), msg: `  ✓ Tavily complete: ${uniqueSources.length} unique sources from ${answers.length} answered queries` });

    return {
      searchResults: [],
      sources: uniqueSources.slice(0, 15),
      summary: webSummary,
      sourceCount: uniqueSources.length,
    };
  }

  // ── NODE 1: Plan research queries ─────────────────────────────────────────
  async function planQueries(state) {
    emit({ type: "step",     ts: ts(), msg: "▶  Planning research strategy..." });
    emit({ type: "llm_call", ts: ts(), msg: "  🤖 [GPT-4o-mini] Generating smart search queries..." });

    const credits = await getCredits();
    if (credits !== null) emit({ type: "info", ts: ts(), msg: `  💳 Twitter API credits remaining: ${credits}` });

    const resp = await llm.invoke([
      { role: "system", content: "You are a market research strategist. Generate precise queries using Twitter advanced search syntax." },
      { role: "user",   content: `Startup idea: "${state.idea}"\nDescription: "${state.description || "N/A"}"\n\nGenerate queries in JSON:\n{\n  "trend_scan": "query for trending topics (-is:retweet lang:en)",\n  "demand_recent": "7 days demand query",\n  "demand_top": "most-engaged tweets query (min_faves:)",\n  "adopter_pain": "pain signal query (wish OR struggling OR frustrated OR 'looking for')",\n  "investor_search": "keyword for investor profile search",\n  "competitor_search": "competitor mentions query",\n  "community_query": "keyword for Twitter community search"\n}\nRespond ONLY with valid JSON.` },
    ]);

    let queries;
    try {
      const m = resp.content.match(/\{[\s\S]*\}/);
      queries = JSON.parse(m ? m[0] : resp.content);
    } catch {
      const idea = state.idea;
      queries = fallbackQueries(idea);
    }

    emit({ type: "queries", ts: ts(), msg: "  ✓ 7 search queries generated:", queries });
    Object.entries(queries).forEach(([k, v]) =>
      emit({ type: "query_line", ts: ts(), msg: `    [${k}] "${String(v).slice(0, 72)}${String(v).length > 72 ? "…" : ""}"` })
    );

    // Start Tavily early so it runs in parallel with Twitter pipeline steps.
    if (!tavilyPrefetchPromise) {
      emit({ type: "info", ts: ts(), msg: "  ⚡ Starting Tavily prefetch in parallel..." });
      tavilyPrefetchPromise = runSingleTavilyResearch(state.idea);
    }
    return { queries };
  }

  // ── NODE 2: Real Twitter Trends ────────────────────────────────────────────
  async function fetchTrends(state) {
    emit({ type: "divider", ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",    step: "trends", ts: ts(), msg: "▶  STEP 1 / 6 — Real Twitter Trends" });

    // Fetch worldwide + USA trends in parallel
    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] Worldwide + USA trends — parallel fetch" });
    const [worldRes, usaRes] = await Promise.allSettled([getTrends(1, 30), getTrends(23424977, 30)]);
    const worldTrends = worldRes.status === "fulfilled" ? worldRes.value : [];
    const usaTrends = usaRes.status === "fulfilled" ? usaRes.value : [];
    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Worldwide: ${worldTrends.length}  ·  USA: ${usaTrends.length} trending topics` });

    const ideaWords = state.idea.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const allTrends = [...worldTrends, ...usaTrends];
    const matched = allTrends.filter(t =>
      ideaWords.some(w => t.name?.toLowerCase().includes(w) || t.target?.query?.toLowerCase().includes(w))
    );

    if (matched.length > 0) {
      emit({ type: "info", ts: ts(), msg: `  🔥 ${matched.length} trend(s) directly match your idea:` });
      matched.slice(0, 5).forEach(t =>
        emit({ type: "query_line", ts: ts(), msg: `    → "${t.name}" — ${t.meta_description || "trending"}` })
      );
    } else {
      emit({ type: "info", ts: ts(), msg: "  ℹ  No direct trend match — idea may be ahead of mainstream" });
    }

    // Fetch context + top tweets in parallel
    const q = state.queries?.trend_scan || `${state.idea} -is:retweet lang:en`;
    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] Context tweets + Top tweets — parallel fetch" });
    const [contextRes, topRes] = await Promise.allSettled([searchTweets(q, 25), searchTopTweets(state.idea, 8)]);
    const contextTweets = contextRes.status === "fulfilled" ? contextRes.value.tweets : [];
    const topTweets = topRes.status === "fulfilled" ? topRes.value : [];
    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Context: ${contextTweets.length}  ·  Top: ${topTweets.length} tweets` });

    emit({ type: "llm_call", ts: ts(), msg: "  🤖 [GPT-4o-mini] Analysing trend signals..." });
    const tweetBlock = tweetSummary([...contextTweets, ...topTweets]);
    const trendNames = allTrends.slice(0, 20).map(t => t.name).join(", ");

    const analysis = await llm.invoke([
      { role: "system", content: "You are a market trend analyst. Extract real signals." },
      { role: "user",   content: `Idea: "${state.idea}"\nTrends: ${trendNames}\nMatched: ${matched.map(t => t.name).join(", ") || "none"}\nTweets:\n${tweetBlock}\n\nExtract: (1) Trending area? (2) Key pain points. (3) Conversation volume. (4) Emerging topics. Max 150 words.` },
    ]);

    const extracted = extractTrendingTopics([...contextTweets, ...topTweets]);
    const trendsData = {
      worldwideCount: worldTrends.length, usaCount: usaTrends.length,
      matched: matched.slice(0, 8), allTrends: allTrends.slice(0, 30),
      hashtags: extracted.hashtags, keywords: extracted.keywords,
      contextTweets: contextTweets.slice(0, 6), topTweets: topTweets.slice(0, 4),
      insights: analysis.content,
    };

    emit({ type: "step_done", step: "trends", ts: ts(), msg: `  ✅ Trends — ${matched.length} matches, ${extracted.hashtags.length} hashtags`, data: trendsData });
    return { trendsData };
  }

  // ── NODE 3: Demand Check ───────────────────────────────────────────────────
  async function demandCheck(state) {
    emit({ type: "divider", ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",    step: "demand", ts: ts(), msg: "▶  STEP 2 / 6 — Demand Check" });

    const recentQ = state.queries?.demand_recent || `${state.idea} lang:en`;
    const topQ    = state.queries?.demand_top    || state.idea;
    const compQ   = state.queries?.competitor_search || `${state.idea} (alternative OR competitor OR "instead of") lang:en`;

    // All 4 Twitter searches in parallel
    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] 7d · 30d · Top tweets · Competitors — 4 parallel fetches" });
    const [recentRes, monthRes, topRes, compRes] = await Promise.allSettled([
      searchRecentTweets(recentQ, 7, 30),
      searchRecentTweets(recentQ, 30, 30),
      searchTopTweets(topQ, 10),
      searchTweets(compQ, 10),
    ]);
    const recentTweets = recentRes.status === "fulfilled" ? recentRes.value.tweets : [];
    const monthTweets = monthRes.status === "fulfilled" ? monthRes.value.tweets : [];
    const topTweets = topRes.status === "fulfilled" ? topRes.value : [];
    const compTweets = compRes.status === "fulfilled" ? compRes.value.tweets : [];

    emit({ type: "tool_result", ts: ts(), msg: `  ✓ 7d: ${recentTweets.length}  ·  30d: ${monthTweets.length}  ·  Top: ${topTweets.length}  ·  Comp: ${compTweets.length}` });

    const allDemandTweets = [...recentTweets, ...monthTweets, ...topTweets];
    const sentiment = analyzeSentiment(allDemandTweets);
    const avgLikes  = allDemandTweets.reduce((s, t) => s + (t.likeCount || 0), 0) / Math.max(allDemandTweets.length, 1);
    const avgRt     = allDemandTweets.reduce((s, t) => s + (t.retweetCount || 0), 0) / Math.max(allDemandTweets.length, 1);

    emit({ type: "info", ts: ts(), msg: `  📊 Sentiment: ${sentiment.positive}% pos · ${sentiment.negative}% neg · ${sentiment.neutral}% neutral` });
    emit({ type: "info", ts: ts(), msg: `  📊 Avg engagement: ${avgLikes.toFixed(1)} likes · ${avgRt.toFixed(1)} RTs` });

    emit({ type: "llm_call", ts: ts(), msg: "  🤖 [GPT-4o-mini] Evaluating demand strength..." });
    const analysis = await llm.invoke([
      { role: "system", content: "You are a market demand specialist." },
      { role: "user",   content: `Idea: "${state.idea}"\n7d tweets: ${recentTweets.length}, 30d: ${monthTweets.length}\nSentiment: ${sentiment.positive}% pos, ${sentiment.negative}% neg\nAvg likes: ${avgLikes.toFixed(1)}, Avg RTs: ${avgRt.toFixed(1)}\n\nTop tweets:\n${tweetSummary(allDemandTweets)}\n\nCompetitor tweets:\n${tweetSummary(compTweets, 5)}\n\nAssess: (1) Demand strength. (2) Problem urgency. (3) WTP signals. (4) Competition saturation. Max 150 words.` },
    ]);

    const demandScore = Math.min(100, Math.round(
      (recentTweets.length / 30) * 30 +
      sentiment.positive * 0.4 +
      Math.min(avgLikes / 10, 20) + Math.min(avgRt / 5, 10)
    ));

    const demandData = {
      recentCount: recentTweets.length, monthCount: monthTweets.length,
      topTweets: topTweets.slice(0, 5), competitorTweets: compTweets.slice(0, 3),
      sentiment, avgLikes: parseFloat(avgLikes.toFixed(1)), avgRt: parseFloat(avgRt.toFixed(1)),
      demandScore, insights: analysis.content,
    };

    emit({ type: "step_done", step: "demand", ts: ts(), msg: `  ✅ Demand score: ${demandScore}/100 — ${recentTweets.length} tweets/7d`, data: demandData });
    return { demandData };
  }

  // ── NODE 4: Find Adopters ──────────────────────────────────────────────────
  async function adopterSearch(state) {
    emit({ type: "divider", ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",    step: "adopters", ts: ts(), msg: "▶  STEP 3 / 6 — Find Early Adopters" });

    const painQ = state.queries?.adopter_pain ||
      `(${state.idea}) (wish OR struggling OR frustrated OR "looking for" OR "need a" OR "can't find") lang:en -is:retweet`;

    // Both pain searches in parallel
    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] Pain tweets + Recent pain — parallel fetch" });
    const [painRes, recentPainRes] = await Promise.allSettled([
      searchTweets(painQ, 25),
      searchRecentTweets(painQ.replace(" lang:en", ""), 14, 20),
    ]);
    const painTweets = painRes.status === "fulfilled" ? painRes.value.tweets : [];
    const recentPain = recentPainRes.status === "fulfilled" ? recentPainRes.value.tweets : [];
    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Pain: ${painTweets.length}  ·  Recent: ${recentPain.length}` });

    const allPain = [...painTweets, ...recentPain];
    const adopters = filterAdopters(allPain);
    emit({ type: "info", ts: ts(), msg: `  🎯 Pain filter: ${adopters.length} potential early adopters` });

    emit({ type: "llm_call", ts: ts(), msg: "  🤖 [GPT-4o-mini] Profiling adopter personas..." });
    const adopterBlock = adopters.slice(0, 10).map(a =>
      `@${a.username} [❤${a.likes}]: "${a.tweet.slice(0, 120)}" — signal: "${a.painSignal}"`
    ).join("\n") || tweetSummary(allPain.slice(0, 8));

    const analysis = await llm.invoke([
      { role: "system", content: "You are a customer discovery expert." },
      { role: "user",   content: `Early adopters for "${state.idea}":\n${adopterBlock}\n\nDescribe: (1) Job title/role. (2) Specific pain. (3) How to reach. (4) WTP. (5) Best channel. Max 150 words.` },
    ]);

    const adopterData = { count: adopters.length, users: adopters.slice(0, 12), allPainTweets: allPain.length, insights: analysis.content };
    emit({ type: "step_done", step: "adopters", ts: ts(), msg: `  ✅ Found ${adopters.length} early adopters from ${allPain.length} pain tweets`, data: adopterData });
    return { adopterData };
  }

  // ── NODE 5: Find Funders ───────────────────────────────────────────────────
  async function funderSearch(state) {
    emit({ type: "divider", ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",    step: "funders", ts: ts(), msg: "▶  STEP 4 / 6 — Find Funders & Investors" });

    const investorKeyword = state.queries?.investor_search || `${state.idea} investor`;
    const funderQ = `(${state.idea}) (investor OR "VC" OR "seed round" OR "angel" OR "funding") lang:en`;

    // All 3 funder searches in parallel
    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] Investors + VCs + funder tweets — parallel fetch" });
    const [investorRes, vcRes, funderTweetRes] = await Promise.allSettled([
      searchUsers(investorKeyword, 20),
      searchUsers(`venture capital ${state.idea.split(" ").slice(0, 2).join(" ")}`, 15),
      searchTweets(funderQ, 20),
    ]);
    const investorUsers = investorRes.status === "fulfilled" ? investorRes.value : [];
    const vcUsers = vcRes.status === "fulfilled" ? vcRes.value : [];
    const funderTweets = funderTweetRes.status === "fulfilled" ? funderTweetRes.value.tweets : [];
    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Investors: ${investorUsers.length}  ·  VCs: ${vcUsers.length}  ·  Tweets: ${funderTweets.length}` });

    const INVESTOR_SIGNALS = ["investor", "vc", "venture", "angel", "fund", "partner", "portfolio", "seed", "series", "backed", "investing", "capital"];
    const allUsers = [...investorUsers, ...vcUsers].filter(u =>
      INVESTOR_SIGNALS.some(s => (u.description || "").toLowerCase().includes(s))
    );
    const seen = new Set();
    const funders = allUsers.filter(u => { if (seen.has(u.userName)) return false; seen.add(u.userName); return true; });
    const funderFromTweets = filterFunders(funderTweets);

    emit({ type: "info", ts: ts(), msg: `  💰 Investor signals: ${funders.length} profiles + ${funderFromTweets.length} from tweets` });

    emit({ type: "llm_call", ts: ts(), msg: "  🤖 [GPT-4o-mini] Mapping funding landscape..." });
    const funderBlock = funders.slice(0, 8).map(u =>
      `@${u.userName} (${(u.followers / 1000).toFixed(1)}k) — ${(u.description || "").slice(0, 80)}`
    ).join("\n") || "No profiles found.";

    const analysis = await llm.invoke([
      { role: "system", content: "You are a VC analyst." },
      { role: "user",   content: `Investors for "${state.idea}":\n${funderBlock}\n\nTweets:\n${tweetSummary(funderTweets, 5)}\n\nAnalyse: (1) Fund types. (2) Stage. (3) Thesis. (4) Who to pitch first. (5) Readiness. Max 150 words.` },
    ]);

    const funderData = {
      count: funders.length + funderFromTweets.length,
      profileInvestors: funders.slice(0, 10).map(u => ({
        username: u.userName, name: u.name, bio: u.description || "",
        followers: u.followers || 0, verified: u.isBlueVerified || false,
        avatar: u.profilePicture || null,
      })),
      tweetInvestors: funderFromTweets.slice(0, 5),
      insights: analysis.content,
    };

    emit({ type: "step_done", step: "funders", ts: ts(), msg: `  ✅ Found ${funderData.count} investor signals total`, data: funderData });
    return { funderData };
  }

  // ── NODE 6: Community Research ─────────────────────────────────────────────
  async function communityResearch(state) {
    emit({ type: "divider", ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",    step: "community", ts: ts(), msg: "▶  STEP 5 / 6 — Twitter Community Research" });

    emit({ type: "tool_call", ts: ts(), msg: "  ⚡ [TWITTER API] GET /twitter/community/get_tweets_from_all_community" });
    const commTweets = await searchCommunityTweets(state.queries?.community_query || state.idea, 15).catch(() => []);
    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Community tweets: ${commTweets.length}` });

    if (commTweets.length > 0) {
      emit({ type: "info", ts: ts(), msg: `  💬 Found ${commTweets.length} community discussions` });
    } else {
      emit({ type: "info", ts: ts(), msg: "  ℹ  No community posts found — topic may be emerging" });
    }

    const communityData = { tweets: commTweets.slice(0, 8), count: commTweets.length };
    emit({ type: "step_done", step: "community", ts: ts(), msg: "  ✅ Community scan complete" });
    return { communityData };
  }

  // ── NODE 7: Tavily Research — awaited checkpoint (prefetched) ─────────────
  async function tavilyResearch(state) {
    emit({ type: "divider",  ts: ts(), msg: "─".repeat(62) });
    emit({ type: "step",     step: "web_research", ts: ts(), msg: "▶  STEP 6 / 6 — Tavily Web Intelligence" });
    emit({ type: "info", ts: ts(), msg: "  ⏳ Waiting for Tavily response before synthesis..." });
    const tavilyData = tavilyPrefetchPromise
      ? await tavilyPrefetchPromise
      : await runSingleTavilyResearch(state.idea);

    emit({ type: "tool_result", ts: ts(), msg: `  ✓ Tavily ready — ${tavilyData.sourceCount} sources` });
    emit({ type: "step_done", step: "web_research", ts: ts(), msg: `  ✅ Web intel: ${tavilyData.sourceCount} sources`, data: tavilyData });
    return { tavilyData };
  }

  // ── NODE 8: Gemini Deep Synthesis ─────────────────────────────────────────
  async function geminiSynthesis(state) {
    emit({ type: "divider",  ts: ts(), msg: "═".repeat(62) });
    emit({ type: "step",     step: "synthesis", ts: ts(), msg: "▶  DEEP SYNTHESIS — Gemini 2.0 Flash + All Data Sources" });
    emit({ type: "llm_call", ts: ts(), msg: "  🟣 [GEMINI] Deep-analysing Twitter + Web intelligence..." });

    const trendList  = state.trendsData?.allTrends?.slice(0, 15).map(t => `"${t.name}"`)?.join(", ") || "N/A";
    const matchedT   = state.trendsData?.matched?.map(t => t.name)?.join(", ") || "none";
    const hashtags   = state.trendsData?.hashtags?.map(h => h.tag)?.join(", ") || "N/A";
    const topTwBlock = tweetSummary(state.trendsData?.topTweets || []);
    const adopterStr = state.adopterData?.users?.slice(0, 6)?.map(a => `@${a.username}: "${a.tweet.slice(0, 100)}"`).join("\n") || "None";
    const funderStr  = userSummary(state.funderData?.profileInvestors || []);
    const commStr    = tweetSummary(state.communityData?.tweets || [], 5);
    const webSummary = state.tavilyData?.summary?.slice(0, 3000) || "N/A";
    const webSourcesList = (state.tavilyData?.sources || []).slice(0, 10)
      .map((s, i) => `[${i + 1}] ${s.title || "Untitled"} — ${s.url}\n    ${(s.content || "").slice(0, 200)}`)
      .join("\n") || "No sources available.";

    const bigPrompt = `You are a world-class startup analyst with deep expertise in market research, venture capital, and product strategy.

STARTUP IDEA: "${state.idea}"
DESCRIPTION: "${state.description || "N/A"}"

═══ TWITTER INTELLIGENCE ═══

1. LIVE TRENDS: ${trendList}
   Matched: ${matchedT}  |  Hashtags: ${hashtags}

2. DEMAND METRICS:
   7-day tweets: ${state.demandData?.recentCount || 0} | 30-day: ${state.demandData?.monthCount || 0}
   Demand score: ${state.demandData?.demandScore || 0}/100
   Sentiment: ${state.demandData?.sentiment?.positive || 0}% positive, ${state.demandData?.sentiment?.negative || 0}% negative
   Avg likes: ${state.demandData?.avgLikes || 0} | Avg RTs: ${state.demandData?.avgRt || 0}

3. TOP TWEETS:
${topTwBlock}

4. EARLY ADOPTERS (${state.adopterData?.count || 0} found):
${adopterStr}

5. INVESTORS (${state.funderData?.count || 0} signals):
${funderStr}

6. COMMUNITY POSTS (${state.communityData?.count || 0}):
${commStr}

═══ WEB INTELLIGENCE (TAVILY) ═══
Summary:
${webSummary}

Sources:
${webSourcesList}

═══ YOUR TASK ═══

Perform the most comprehensive, brutally honest startup validation. Cite specific data. Be concrete, not generic.

Respond in EXACTLY this JSON format (ONLY valid JSON):
{
  "score": <integer 0-100>,
  "headline": "<one sharp memorable verdict sentence>",
  "why_it_works": "<3-4 sentences citing actual data>",
  "why_it_fails": "<3-4 sentences citing actual risks>",
  "strengths": ["<data-backed strength>", "<specific>", "<specific>"],
  "weaknesses": ["<data-backed weakness>", "<specific>", "<specific>"],
  "issues": ["<critical issue 1>", "<critical issue 2>", "<critical issue 3>"],
  "best_points": ["<best selling point 1>", "<best point 2>", "<best point 3>"],
  "idea_changes": ["<how to improve/pivot the idea 1>", "<change 2>", "<change 3>"],
  "new_additions": ["<new feature or product to add 1>", "<addition 2>", "<addition 3>"],
  "target_customer": "<precise: job title, company size, specific pain, WTP>",
  "go_to_market": "<concrete first 3 steps, who to talk to first>",
  "competition_risk": "low|medium|high",
  "timing": "too_early|perfect|too_late",
  "market_size": "<estimated TAM/SAM from web research + Twitter volume>",
  "key_insight": "<the single most important insight from all research>",
  "red_flags": ["<specific red flag 1>", "<specific red flag 2>"],
  "recommendation": "build|explore|pivot|abandon",
  "investor_readiness": "not_ready|early_stage|ready",
  "next_actions": ["<specific action 1>", "<specific action 2>", "<specific action 3>"],
  "top_funders": [
    { "name": "<VC firm or fund name>", "twitter": "<@handle if known>", "focus": "<investment thesis>", "why": "<why they'd fund this specific idea>" },
    { "name": "<VC 2>", "twitter": "<@handle>", "focus": "<thesis>", "why": "<reason>" },
    { "name": "<VC 3>", "twitter": "<@handle>", "focus": "<thesis>", "why": "<reason>" }
  ],
  "web_insights": "<key findings from Tavily web research — market size data, competitor landscape, funding activity>"
}`;

    let verdict;
    try {
      const result = await gemini.generateContent(bigPrompt);
      const text = result.response.text();
      emit({ type: "info", ts: ts(), msg: `  🟣 [GEMINI] Response: ${text.length} chars` });
      const m = text.match(/\{[\s\S]*\}/);
      verdict = JSON.parse(m ? m[0] : text);
    } catch (geminiErr) {
      emit({ type: "warn", ts: ts(), msg: `  ⚠ Gemini error: ${geminiErr.message.slice(0, 60)} — GPT fallback` });
      try {
        const resp = await llm.invoke([
          { role: "system", content: "You are a startup analyst. Respond ONLY with valid JSON." },
          { role: "user",   content: `Idea: "${state.idea}". Demand: ${state.demandData?.demandScore || 50}/100. Adopters: ${state.adopterData?.count || 0}. Funders: ${state.funderData?.count || 0}. Web data: "${state.tavilyData?.summary?.slice(0, 500) || "N/A"}". Return JSON with all fields: score, headline, why_it_works, why_it_fails, strengths(array), weaknesses(array), issues(array), best_points(array), idea_changes(array), new_additions(array), target_customer, go_to_market, competition_risk, timing, market_size, key_insight, red_flags(array), recommendation, investor_readiness, next_actions(array), top_funders(array of {name,twitter,focus,why}), web_insights.` },
        ]);
        const m = resp.content.match(/\{[\s\S]*\}/);
        verdict = JSON.parse(m ? m[0] : resp.content);
      } catch {
        verdict = fallbackVerdict(state);
      }
    }

    emit({ type: "divider", ts: ts(), msg: "═".repeat(62) });
    emit({ type: "verdict", ts: ts(), msg: `  🏁 COMPLETE — Score: ${verdict.score}/100 — ${verdict.recommendation?.toUpperCase()}`, verdict });
    emit({ type: "divider", ts: ts(), msg: "═".repeat(62) });
    return { verdict };
  }

  function fallbackVerdict(state) {
    return {
      score: state.demandData?.demandScore || 50,
      headline: "Research complete — manual review recommended",
      why_it_works: "Twitter data shows market awareness.", why_it_fails: "Insufficient data for full analysis.",
      strengths: ["Market conversations exist"], weaknesses: ["Limited data"],
      issues: ["Need more data"], best_points: ["Conversations happening"],
      idea_changes: ["Gather more customer feedback"], new_additions: ["Build MVP first"],
      target_customer: "Early adopters", go_to_market: "Direct outreach",
      competition_risk: "medium", timing: "perfect", market_size: "Unknown",
      key_insight: "More research needed", red_flags: ["Limited data"],
      recommendation: "explore", investor_readiness: "not_ready",
      next_actions: ["Gather more data", "Interview potential customers"],
      top_funders: [], web_insights: "Web research unavailable",
    };
  }

  // ── Build LangGraph ────────────────────────────────────────────────────────
  const graph = new StateGraph(State)
    .addNode("plan_queries", wrapNode("Plan queries", planQueries, (state) => ({ queries: fallbackQueries(state.idea) })))
    .addNode("fetch_trends", wrapNode("Trends", fetchTrends, () => ({
      trendsData: {
        worldwideCount: 0, usaCount: 0, matched: [], allTrends: [], hashtags: [], keywords: [],
        contextTweets: [], topTweets: [], insights: "Trend data unavailable due to API/runtime failure.",
      },
    })))
    .addNode("demand_check", wrapNode("Demand", demandCheck, () => ({
      demandData: {
        recentCount: 0, monthCount: 0, topTweets: [], competitorTweets: [],
        sentiment: { positive: 0, negative: 0, neutral: 100 }, avgLikes: 0, avgRt: 0, demandScore: 0,
        insights: "Demand data unavailable due to API/runtime failure.",
      },
    })))
    .addNode("adopter_search", wrapNode("Adopters", adopterSearch, () => ({
      adopterData: { count: 0, users: [], allPainTweets: 0, insights: "Adopter data unavailable due to API/runtime failure." },
    })))
    .addNode("funder_search", wrapNode("Funders", funderSearch, () => ({
      funderData: { count: 0, profileInvestors: [], tweetInvestors: [], insights: "Funder data unavailable due to API/runtime failure." },
    })))
    .addNode("community_research", wrapNode("Community", communityResearch, () => ({
      communityData: { tweets: [], count: 0 },
    })))
    .addNode("tavily_research", wrapNode("Web intel", tavilyResearch, () => ({
      tavilyData: { searchResults: [], sources: [], summary: "", sourceCount: 0 },
    })))
    .addNode("gemini_synthesis", wrapNode("Synthesis", geminiSynthesis, (state) => ({ verdict: fallbackVerdict(state) })))
    .addEdge(START,                "plan_queries")
    .addEdge("plan_queries",       "fetch_trends")
    .addEdge("fetch_trends",       "demand_check")
    .addEdge("demand_check",       "adopter_search")
    .addEdge("adopter_search",     "funder_search")
    .addEdge("funder_search",      "community_research")
    .addEdge("community_research", "tavily_research")
    .addEdge("tavily_research",    "gemini_synthesis")
    .addEdge("gemini_synthesis",   END);

  return graph.compile();
}

module.exports = { createResearchGraph };
