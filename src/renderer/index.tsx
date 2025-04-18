import './index.css'

import { createRoot } from 'react-dom/client'
import App from './App'

const container = document.getElementById('root')
console.log(container)
const root = createRoot(container!)

root.render(<App />)
