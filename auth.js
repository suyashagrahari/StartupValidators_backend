const jwt = require("jsonwebtoken");

function signToken(user) {
  const secret = process.env.JWT_SECRET || "dev-secret-change-this";
  return jwt.sign({ sub: String(user._id), email: user.email, name: user.name }, secret, { expiresIn: "7d" });
}

function decodeToken(token) {
  const secret = process.env.JWT_SECRET || "dev-secret-change-this";
  try {
    return jwt.verify(token, secret);
  } catch {
    return null;
  }
}

function extractToken(req) {
  const auth = req.headers.authorization || "";
  if (auth.startsWith("Bearer ")) return auth.slice(7);
  if (req.query?.token) return String(req.query.token);
  return null;
}

function requireAuth(req, res, next) {
  const token = extractToken(req);
  if (!token) return res.status(401).json({ error: "Unauthorized" });

  const payload = decodeToken(token);
  if (!payload?.sub) return res.status(401).json({ error: "Invalid token" });

  req.auth = { userId: payload.sub, email: payload.email, name: payload.name };
  next();
}

module.exports = { signToken, decodeToken, extractToken, requireAuth };
