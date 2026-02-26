require("dotenv").config();
const express = require("express");
const http    = require("http");
const cors    = require("cors");
const { WebSocketServer } = require("ws");
const { createResearchGraph } = require("./agent/graph");

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocketServer({ server });
const PORT   = process.env.PORT || 3001;
const DB_ENABLED = Boolean(process.env.MONGODB_URI);
const bcrypt = require("bcryptjs");
const { connectDB } = require("./db");
const User = require("./models/user");
const Research = require("./models/research");
const { signToken, decodeToken, extractToken, requireAuth } = require("./auth");

// ── Terminal colours (ANSI) ────────────────────────────────────────────────────
const C = {
  reset:   "\x1b[0m",
  bright:  "\x1b[1m",
  dim:     "\x1b[2m",
  cyan:    "\x1b[36m",
  magenta: "\x1b[35m",
  yellow:  "\x1b[33m",
  green:   "\x1b[32m",
  red:     "\x1b[31m",
  gray:    "\x1b[90m",
  white:   "\x1b[37m",
};

function logToConsole(data) {
  if (!data || data.type === "ping" || data.type === "divider") return;
  const time = data.ts ? `${C.gray}[${data.ts}]${C.reset} ` : "";
  const msg  = (data.msg || "").replace(/<[^>]+>/g, ""); // strip HTML tags

  switch (data.type) {
    case "init":
      console.log(`${time}${C.dim}${msg}${C.reset}`);
      break;
    case "step":
      console.log(`\n${time}${C.cyan}${C.bright}${msg}${C.reset}`);
      break;
    case "llm_call":
      console.log(`${time}${C.magenta}${msg}${C.reset}`);
      break;
    case "tool_call":
      console.log(`${time}${C.yellow}${msg}${C.reset}`);
      break;
    case "tool_result":
      console.log(`${time}${C.green}${msg}${C.reset}`);
      break;
    case "step_done":
      console.log(`${time}${C.green}${C.bright}${msg}${C.reset}`);
      break;
    case "queries":
      console.log(`${time}${C.green}${msg}${C.reset}`);
      break;
    case "query_line":
      console.log(`${time}${C.gray}${msg}${C.reset}`);
      break;
    case "info":
      console.log(`${time}${C.white}${msg}${C.reset}`);
      break;
    case "warn":
      console.log(`${time}${C.yellow}${msg}${C.reset}`);
      break;
    case "error":
      console.log(`${time}${C.red}${msg}${C.reset}`);
      break;
    case "verdict":
      console.log(`\n${time}${C.bright}${C.green}${msg}${C.reset}\n`);
      break;
    case "complete":
      console.log(`${C.bright}${C.green}✅ Research complete — score: ${data.result?.verdict?.score ?? "?"}${C.reset}\n`);
      break;
    default:
      if (msg) console.log(`${time}${msg}`);
  }
}

app.use(cors({ origin: ["http://localhost:3000", "http://127.0.0.1:3000"] }));
app.use(express.json());

async function saveResearchForUser(userId, idea, description, result) {
  if (!DB_ENABLED || !userId) return;
  try {
    await Research.create({
      userId,
      idea: String(idea || "").trim(),
      description: String(description || "").trim(),
      result,
    });
  } catch (err) {
    console.error("[history] save error:", err.message);
  }
}

function requireDbConfigured(req, res, next) {
  if (!DB_ENABLED) return res.status(503).json({ error: "MongoDB is not configured on server" });
  next();
}

// ── Auth + History API ───────────────────────────────────────────────────────
app.post("/api/auth/signup", requireDbConfigured, async (req, res) => {
  try {
    const { name = "", email = "", password = "" } = req.body || {};
    if (String(name).trim().length < 2) return res.status(400).json({ error: "Name is too short" });
    if (!String(email).includes("@")) return res.status(400).json({ error: "Invalid email" });
    if (String(password).length < 5) return res.status(400).json({ error: "Password must be at least 5 chars" });

    const existing = await User.findOne({ email: String(email).toLowerCase().trim() }).lean();
    if (existing) return res.status(409).json({ error: "Email already exists" });

    const passwordHash = await bcrypt.hash(String(password), 10);
    const user = await User.create({
      name: String(name).trim(),
      email: String(email).toLowerCase().trim(),
      passwordHash,
    });

    const token = signToken(user);
    return res.json({ token, user: { id: String(user._id), name: user.name, email: user.email } });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Signup failed" });
  }
});

app.post("/api/auth/login", requireDbConfigured, async (req, res) => {
  try {
    const { email = "", password = "" } = req.body || {};
    const user = await User.findOne({ email: String(email).toLowerCase().trim() });
    if (!user) return res.status(401).json({ error: "Invalid credentials" });

    const ok = await bcrypt.compare(String(password), user.passwordHash);
    if (!ok) return res.status(401).json({ error: "Invalid credentials" });

    const token = signToken(user);
    return res.json({ token, user: { id: String(user._id), name: user.name, email: user.email } });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Login failed" });
  }
});

app.get("/api/auth/me", requireDbConfigured, requireAuth, async (req, res) => {
  try {
    const user = await User.findById(req.auth.userId).lean();
    if (!user) return res.status(404).json({ error: "User not found" });
    return res.json({ user: { id: String(user._id), name: user.name, email: user.email } });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Failed to fetch profile" });
  }
});

app.get("/api/history", requireDbConfigured, requireAuth, async (req, res) => {
  try {
    const docs = await Research.find({ userId: req.auth.userId }).sort({ createdAt: -1 }).limit(50).lean();
    const items = docs.map((d) => ({
      id: String(d._id),
      idea: d.idea,
      description: d.description,
      createdAt: d.createdAt,
      score: d.result?.verdict?.score ?? null,
      headline: d.result?.verdict?.headline ?? "",
      recommendation: d.result?.verdict?.recommendation ?? "",
    }));
    return res.json({ items });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Failed to fetch history" });
  }
});

app.get("/api/history/:id", requireDbConfigured, requireAuth, async (req, res) => {
  try {
    const doc = await Research.findOne({ _id: req.params.id, userId: req.auth.userId }).lean();
    if (!doc) return res.status(404).json({ error: "Not found" });
    return res.json({
      item: {
        id: String(doc._id),
        idea: doc.idea,
        description: doc.description,
        createdAt: doc.createdAt,
        result: doc.result,
      },
    });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Failed to fetch history item" });
  }
});

// ── Health ────────────────────────────────────────────────────────────────────
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    mongodb: DB_ENABLED ? "configured" : "missing",
    openai:  process.env.OPENAI_API_KEY  ? "configured" : "missing",
    gemini:  process.env.GEMINI_API_KEY  ? "configured" : "missing",
    twitter: process.env.TWITTER_API_KEY ? "configured" : "missing",
    tavily:  process.env.TAVILY_API_KEY  ? "configured" : "missing",
    version: "4.0 — LangGraph 8-node · Twitter + Tavily + Gemini",
    timestamp: new Date().toISOString(),
  });
});

// ── Shared init banner ────────────────────────────────────────────────────────
function sendBanner(emit, idea) {
  const t = () => new Date().toTimeString().slice(0, 8);
  emit({ type: "init",    ts: t(), msg: `$ startup-validator --idea "${String(idea).trim()}"` });
  emit({ type: "init",    ts: t(), msg: `  Agent:  LangGraph 8-node pipeline` });
  emit({ type: "init",    ts: t(), msg: `  LLMs:   GPT-4o-mini (steps) + Gemini 2.0 Flash (synthesis)` });
  emit({ type: "init",    ts: t(), msg: `  Data:   twitterapi.io (trends, search, users, communities)` });
  emit({ type: "init",    ts: t(), msg: `  Web:    Tavily deep research + streaming intelligence` });
  emit({ type: "divider", ts: t(), msg: "─".repeat(62) });
}

// ── SSE Research endpoint (GET /api/research?idea=...&description=...) ────────
app.get("/api/research", async (req, res) => {
  const { idea, description = "" } = req.query;
  const token = extractToken(req);
  const authPayload = token ? decodeToken(token) : null;

  if (!idea || String(idea).trim().length < 3) {
    return res.status(400).json({ error: "Provide a startup idea (min 3 chars)." });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  console.log(`\n${C.bright}${C.cyan}🔬 [SSE]  Research started: "${String(idea).trim()}"${C.reset}`);

  function send(data) {
    logToConsole(data);
    if (!res.writableEnded) res.write(`data: ${JSON.stringify(data)}\n\n`);
  }

  const ping = setInterval(() => send({ type: "ping" }), 15000);
  req.on("close", () => clearInterval(ping));

  sendBanner(send, idea);

  try {
    const graph  = createResearchGraph(send);
    const result = await graph.invoke({
      idea: String(idea).trim(),
      description: String(description).trim(),
    });
    await saveResearchForUser(authPayload?.sub, idea, description, result);
    send({ type: "complete", result });
  } catch (err) {
    console.error("[SSE research] Error:", err.message);
    send({ type: "error", ts: new Date().toTimeString().slice(0, 8), msg: `  ✗ Agent error: ${err.message}` });
  } finally {
    clearInterval(ping);
    if (!res.writableEnded) res.end();
  }
});

// ── WebSocket Research endpoint ───────────────────────────────────────────────
wss.on("connection", (ws) => {
  console.log("[WS] Client connected");

  ws.on("message", async (rawMsg) => {
    let idea, description;
    try {
      const parsed = JSON.parse(rawMsg.toString());
      idea        = parsed.idea;
      description = parsed.description || "";
      ws.userId   = parsed.token ? decodeToken(parsed.token)?.sub : null;
    } catch {
      ws.send(JSON.stringify({ type: "error", msg: "Invalid JSON message" }));
      return;
    }

    if (!idea || String(idea).trim().length < 3) {
      ws.send(JSON.stringify({ type: "error", msg: "Provide a startup idea (min 3 chars)." }));
      return;
    }

    console.log(`\n${C.bright}${C.cyan}🔬 [WS]   Research started: "${String(idea).trim()}"${C.reset}`);

    function send(data) {
      logToConsole(data);
      if (ws.readyState === 1 /* OPEN */) ws.send(JSON.stringify(data));
    }

    sendBanner(send, idea);

    try {
      const graph  = createResearchGraph(send);
      const result = await graph.invoke({
        idea: String(idea).trim(),
        description: String(description).trim(),
      });
      await saveResearchForUser(ws.userId, idea, description, result);
      send({ type: "complete", result });
    } catch (err) {
      console.error("[WS research] Error:", err.message);
      send({ type: "error", ts: new Date().toTimeString().slice(0, 8), msg: `  ✗ Agent error: ${err.message}` });
    }
  });

  ws.on("close", () => console.log("[WS] Client disconnected"));
  ws.on("error", (err) => console.error("[WS] Error:", err.message));
});

// ── Start ─────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
  console.log(`\n🚀 Startup Validator v4 — http://localhost:${PORT}`);
  console.log(`   OpenAI:  ${process.env.OPENAI_API_KEY  ? "✓" : "✗ missing"}`);
  console.log(`   Gemini:  ${process.env.GEMINI_API_KEY  ? "✓" : "✗ missing"}`);
  console.log(`   Twitter: ${process.env.TWITTER_API_KEY ? "✓" : "✗ missing"}`);
  console.log(`   Tavily:  ${process.env.TAVILY_API_KEY  ? "✓" : "✗ missing"}`);
  console.log(`   SSE:     http://localhost:${PORT}/api/research`);
  console.log(`   WS:      ws://localhost:${PORT}\n`);
});

connectDB().catch((err) => {
  console.error("[db] connection failed:", err.message);
});
