/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: {
          light: '#38bdf8',
          DEFAULT: '#0284c7',
          dark: '#0369a1',
        }
      }
    },
  },
  plugins: [],
}
