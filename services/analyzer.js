const POSITIVE = ["love", "great", "amazing", "excellent", "helpful", "good", "best", "awesome", "solved", "works", "perfect", "brilliant", "useful", "fantastic", "need this", "want this", "would pay", "would use"];
const NEGATIVE = ["hate", "bad", "terrible", "awful", "broken", "failed", "sucks", "struggling", "problem", "issue", "frustrated", "annoying", "waste", "useless", "disappointing", "can't find", "wish there was"];
const PAIN_SIGNALS = ["i wish", "looking for", "struggling with", "need a tool", "anyone know", "how do i", "can't find", "is there a way", "help me", "frustrated", "problem with", "hate that", "wish someone", "does anyone", "need help", "stuck on", "can't figure"];
const INVESTOR_SIGNALS = ["investor", "vc", "venture capital", "angel", "fund", "partner at", "investing in", "portfolio", "seed", "series a", "backed by", "investing", "funding", "startup investor", "early stage"];

function analyzeSentiment(tweets) {
  let positive = 0, negative = 0, neutral = 0;
  const scored = tweets.map((t) => {
    const text = t.text.toLowerCase();
    const pos = POSITIVE.filter((w) => text.includes(w)).length;
    const neg = NEGATIVE.filter((w) => text.includes(w)).length;
    if (pos > neg) { positive++; return { ...t, sentiment: "positive" }; }
    if (neg > pos) { negative++; return { ...t, sentiment: "negative" }; }
    neutral++;
    return { ...t, sentiment: "neutral" };
  });
  const total = tweets.length || 1;
  return {
    positive: Math.round((positive / total) * 100),
    negative: Math.round((negative / total) * 100),
    neutral: Math.round((neutral / total) * 100),
    tweets: scored,
  };
}

function extractTrendingTopics(tweets) {
  const tagMap = {};
  const wordMap = {};
  tweets.forEach((t) => {
    (t.entities?.hashtags || []).forEach((h) => {
      const tag = h.text.toLowerCase();
      tagMap[tag] = (tagMap[tag] || 0) + 1;
    });
    t.text.split(/\s+/).forEach((word) => {
      const w = word.toLowerCase().replace(/[^a-z]/g, "");
      if (w.length > 4 && !["that", "this", "with", "have", "from", "they", "will", "been", "just", "your", "what", "when", "about", "there", "their", "would", "could", "should"].includes(w)) {
        wordMap[w] = (wordMap[w] || 0) + 1;
      }
    });
  });
  const hashtags = Object.entries(tagMap).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([tag, count]) => ({ tag: `#${tag}`, count }));
  const keywords = Object.entries(wordMap).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([word, count]) => ({ word, count }));
  return { hashtags, keywords };
}

function filterAdopters(tweets) {
  return tweets
    .filter((t) => {
      const text = t.text.toLowerCase();
      return PAIN_SIGNALS.some((s) => text.includes(s));
    })
    .map((t) => ({
      username: t.author?.userName || "unknown",
      name: t.author?.name || "Unknown",
      avatar: t.author?.profilePicture || null,
      tweet: t.text,
      url: t.url,
      likes: t.likeCount || 0,
      retweets: t.retweetCount || 0,
      createdAt: t.createdAt,
      painSignal: PAIN_SIGNALS.find((s) => t.text.toLowerCase().includes(s)) || "",
    }))
    .slice(0, 10);
}

function filterFunders(tweets) {
  return tweets
    .filter((t) => {
      const bio = (t.author?.description || "").toLowerCase();
      return INVESTOR_SIGNALS.some((s) => bio.includes(s));
    })
    .map((t) => ({
      username: t.author?.userName || "unknown",
      name: t.author?.name || "Unknown",
      avatar: t.author?.profilePicture || null,
      bio: t.author?.description || "",
      tweet: t.text,
      url: t.url,
      followers: t.author?.followers || 0,
      verified: t.author?.isBlueVerified || false,
      createdAt: t.createdAt,
    }))
    .slice(0, 10);
}

module.exports = { analyzeSentiment, extractTrendingTopics, filterAdopters, filterFunders };
