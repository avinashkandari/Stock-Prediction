import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './App.css';

// Import Chart.js defaults (for react-chartjs-2)
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const root = createRoot(document.getElementById('root'));
root.render(<App />);