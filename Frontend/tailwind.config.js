/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'sans-serif'],
        display: ['Outfit', 'sans-serif'],
      },
      colors: {
        berry: {
          50: '#fdeef1',
          100: '#fbd9de',
          200: '#f4b3bd',
          300: '#ed8ca4',
          400: '#e66686',
          500: '#dd2d4a',
          600: '#c42642',
          700: '#a51f37',
          800: '#90152a',
          900: '#880d1e',
        },
        cyan: {
          50: '#f3fbfc',
          100: '#e5f6f8',
          200: '#cbeef3',
          300: '#a7dce4',
          400: '#7fc6d2',
          500: '#56adb9',
        },
        neutral: {
          50: '#f7f4f0',
          100: '#ede3d8',
          200: '#dcccc0',
          300: '#cbb9ad',
          400: '#b6a48e',
          500: '#c1b098',
          600: '#9b8c77',
          700: '#7f6f5a',
          800: '#645645',
          900: '#4b3d30',
        },
        soft: {
          bg: '#fbf7f4',
          surface: '#ffffff',
          border: '#e9ded2',
          ink: '#5a0f1a',
          inkMuted: '#8a5a52',
        },
      },
      boxShadow: {
        soft: '0 4px 20px -4px rgba(136, 13, 30, 0.08)',
        softHover: '0 10px 30px -6px rgba(136, 13, 30, 0.12)',
        berry: '0 0 0 1px rgba(221, 45, 74, 0.15)',
      },
      borderRadius: {
        xl: '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'slide-up': 'slideUp 0.6s ease-out forwards',
        'pulse-soft': 'pulseSoft 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}
