import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      // Bloomberg terminal palette — mirrors .streamlit/config.toml
      colors: {
        bg: "#050505",
        surface: "#121212",
        border: "#2A2A2A",
        text: "#E0E0E0",
        dim: "#AAAAAA",
        green: "#00FFA3",
        red: "#FF3366",
        blue: "#00BFFF",
        yellow: "#FFD700",
      },
      fontFamily: {
        mono: ["Courier New", "JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
