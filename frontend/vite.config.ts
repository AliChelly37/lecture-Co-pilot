import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: Vite serves the UI and proxies API + WebSocket to the Python backend.
// Prod: `npm run build` writes dist/, which the backend serves itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8765',
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
})
