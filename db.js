const mongoose = require("mongoose");

let isConnected = false;

async function connectDB() {
  if (isConnected) return;

  const uri = process.env.MONGODB_URI;
  if (!uri) {
    console.log("[db] MongoDB not configured. Auth/history endpoints will be disabled.");
    return;
  }

  await mongoose.connect(uri, {
    dbName: process.env.MONGODB_DB || "startup_validator",
  });
  isConnected = true;
  console.log("[db] Connected to MongoDB");
}

module.exports = { connectDB };
