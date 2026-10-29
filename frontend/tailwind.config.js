/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0a0f1e',
          800: '#0d1428',
          700: '#111b38',
          600: '#1a2548',
        },
      },
    },
  },
  plugins: [],
}

