import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// base "./" so the built site works from any sub-path (e.g. GitHub Pages /<repo>/).
export default defineConfig({
  base: "./",
  plugins: [react()],
  test: { environment: "node" },
});
