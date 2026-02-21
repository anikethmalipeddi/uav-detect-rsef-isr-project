import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        slatebg: "#e6eaef",
        card: "#f7f9fc",
        ink: "#0f1f33",
        accent: "#0b6477",
      },
    },
  },
  plugins: [],
};

export default config;
