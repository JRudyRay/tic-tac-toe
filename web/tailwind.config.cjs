/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        ink: '#0f172a',
        mist: '#e2e8f0',
        accent: '#0ea5e9',
        accent2: '#a855f7',
      },
      boxShadow: {
        glass: '0 10px 50px rgba(15, 23, 42, 0.22)',
      },
    },
  },
  plugins: [],
};
