const mongoose = require("mongoose");

const ResearchSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    idea: { type: String, required: true, trim: true },
    description: { type: String, default: "" },
    result: { type: Object, required: true },
  },
  { timestamps: true }
);

ResearchSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.models.Research || mongoose.model("Research", ResearchSchema);
