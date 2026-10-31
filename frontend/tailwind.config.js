/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#FFFFFF",
        canvas: "#F7F8FA",
        ink: "#20242B",
        muted: "#697386",
        border: "#E3E7ED",
        accent: {
          DEFAULT: "#315E9E",
          hover: "#28507F",
        },
        classical: "#697386",
        quantum: "#315E9E",
        success: "#1F8A5A",
        warning: "#B5750A",
        danger: "#B3261E",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["Roboto Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
