import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { FiTrendingUp, FiAlertTriangle, FiCheckCircle, FiRefreshCw, FiDollarSign, FiCalendar, FiBarChart2 } from 'react-icons/fi';
import './StockPredictor.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function StockPredictor() {
  const [ticker, setTicker] = useState('AAPL');
  const [startDate, setStartDate] = useState('2020-01-01');
  const [isTraining, setIsTraining] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState('idle');
  const [pollCount, setPollCount] = useState(0);
  const [progress, setProgress] = useState(0);
  
  const formatISTDate = (dateString) => {
    const date = new Date(dateString);
    date.setMinutes(date.getMinutes() + date.getTimezoneOffset() + 330);
    return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const trainModel = async () => {
    setIsTraining(true);
    setStatus('training');
    setError(null);
    setPredictionData(null);
    setPollCount(0);
    setProgress(0);
    
    try {
      const response = await fetch('http://localhost:5000/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          ticker, 
          start_date: startDate 
        })
      });

      if (!response.ok) {
        throw new Error(await response.text() || 'Training request failed');
      }
      
      pollForResults();
    } catch (err) {
      setError(err.message);
      setIsTraining(false);
      setStatus('error');
    }
  };

  const pollForResults = async () => {
    try {
      const response = await fetch(`http://localhost:5000/api/predict/${ticker}`);
      const data = await response.json();
      
      setPollCount(prev => prev + 1);

      if (data.status === 'ready') {
        setPredictionData({
          historical: data.historical,
          forecast: data.forecast,
          metrics: data.metrics
        });
        setIsTraining(false);
        setStatus('ready');
        return;
      }
      
      if (data.status === 'training') {
        setProgress(data.progress || 0);
        if (pollCount < 30) {
          setTimeout(pollForResults, 5000);
        } else {
          throw new Error('Training took too long. Please try again.');
        }
        return;
      }

      throw new Error(data.message || 'Unknown polling response');

    } catch (err) {
      setError(err.message);
      setIsTraining(false);
      setStatus('error');
    }
  };

  const chartData = predictionData ? {
    labels: [...predictionData.historical.dates, ...predictionData.forecast.dates],
    datasets: [
      {
        label: 'Historical Prices',
        data: [...predictionData.historical.actual, ...Array(predictionData.forecast.dates.length).fill(null)],
        borderColor: '#4361ee',
        backgroundColor: 'rgba(67, 97, 238, 0.05)',
        borderWidth: 3,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 5,
        fill: true
      },
      {
        label: 'Model Predictions',
        data: [...Array(predictionData.historical.dates.length - predictionData.historical.predicted.length).fill(null), 
               ...predictionData.historical.predicted, 
               ...predictionData.forecast.predicted],
        borderColor: '#f72585',
        backgroundColor: 'rgba(247, 37, 133, 0.05)',
        borderWidth: 3,
        borderDash: [5, 5],
        pointRadius: 0,
        pointHoverRadius: 5,
        fill: true
      }
    ]
  } : null;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <FiTrendingUp className="header-icon" />
          <h1>Stock Price Prediction with News and Sentiment Analysis
          </h1>
          <p className="header-subtitle">
          Harness the power of LSTM and real-time news sentiment to forecast market trends and make smarter investment decisions.
          </p>
        </div>
      </header>

      <main className="main-content">
        <section className="control-panel card">
          <h2>
            <FiBarChart2 className="icon-spacing" />
            Stock Selection
          </h2>
          <div className="input-group">
            <label htmlFor="ticker-input">
              <FiDollarSign className="icon-spacing" />
              Ticker Symbol
            </label>
            <input
              id="ticker-input"
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL, MSFT, GOOGL"
              disabled={isTraining}
            />
          </div>

          <div className="input-group">
            <label htmlFor="date-input">
              <FiCalendar className="icon-spacing" />
              Start Date
            </label>
            <input
              id="date-input"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              max={new Date().toISOString().split('T')[0]}
              disabled={isTraining}
            />
          </div>

          <button
            className={`predict-button ${isTraining ? 'loading' : ''}`}
            onClick={trainModel}
            disabled={isTraining}
          >
            {isTraining ? (
              <>
                <FiRefreshCw className="spin" />
                <span>Training Model ({progress}%)</span>
              </>
            ) : (
              <>
                <FiTrendingUp />
                <span>Generate Predictions</span>
              </>
            )}
          </button>
        </section>

        <div className="content-area">
          {status === 'training' && (
            <section className="status-card card">
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>
              <div className="status-content">
                <FiRefreshCw className="spin status-icon" />
                <div>
                  <h3>Training AI Model</h3>
                  <p>Processing historical data and news sentiment...</p>
                  <small>This may take a few minutes</small>
                </div>
              </div>
            </section>
          )}

          {status === 'error' && (
            <section className="error-card card">
              <div className="error-content">
                <FiAlertTriangle className="error-icon" />
                <div>
                  <h3>Error Occurred</h3>
                  <p>{error}</p>
                </div>
              </div>
              <button className="retry-button" onClick={trainModel}>
                <FiRefreshCw /> Try Again
              </button>
            </section>
          )}

          {status === 'ready' && predictionData && (
            <>
              <section className="metrics-card card">
                <h2>Model Performance Metrics</h2>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <div className="metric-value">{predictionData.metrics.rmse.toFixed(2)}</div>
                    <div className="metric-label">Root Mean Squared Error</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-value">{predictionData.metrics.mae.toFixed(2)}</div>
                    <div className="metric-label">Mean Absolute Error</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-value">
                      {new Date(predictionData.metrics.last_updated).toLocaleString('en-IN', {
                        timeZone: 'Asia/Kolkata'
                      })}
                    </div>
                    <div className="metric-label">Last Updated</div>
                  </div>
                </div>
              </section>

              <section className="chart-card card">
                <h2>Price Prediction Chart</h2>
                <div className="chart-container">
                  {chartData && (
                    <Line
                      data={chartData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'top',
                            labels: {
                              usePointStyle: true,
                              padding: 20,
                              font: {
                                size: 14
                              }
                            }
                          },
                          tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleFont: {
                              size: 16
                            },
                            bodyFont: {
                              size: 14
                            },
                            callbacks: {
                              title: (context) => formatISTDate(context[0].label)
                            }
                          }
                        },
                        scales: {
                          x: {
                            grid: {
                              display: false
                            },
                            ticks: {
                              autoSkip: true,
                              maxTicksLimit: 10,
                              font: {
                                size: 12
                              },
                              callback: (value) => {
                                const date = chartData.labels[value];
                                return formatISTDate(date);
                              }
                            }
                          },
                          y: {
                            grid: {
                              color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                              font: {
                                size: 12
                              },
                              callback: (value) => `$${value}`
                            }
                          }
                        },
                        interaction: {
                          intersect: false,
                          mode: 'index'
                        }
                      }}
                    />
                  )}
                </div>
              </section>

              <section className="forecast-card card">
                <h2>10-Day Price Forecast</h2>
                <p className="forecast-subtitle">
                  Predictions from {formatISTDate(predictionData.metrics.prediction_date)}
                </p>
                <div className="table-container">
                  <table className="forecast-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Predicted Price</th>
                        <th>Change %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionData.forecast.dates.map((date, i) => {
                        const current = predictionData.historical.actual.slice(-1)[0];
                        const predicted = predictionData.forecast.predicted[i];
                        const change = ((predicted - current) / current * 100).toFixed(2);
                        
                        return (
                          <tr key={date}>
                            <td>{formatISTDate(date)}</td>
                            <td>${predicted.toFixed(2)}</td>
                            <td className={`change ${change >= 0 ? 'positive' : 'negative'}`}>
                              {change >= 0 ? '+' : ''}{change}%
                              {i === 0 && (
                                <span className="trend-indicator">
                                  {change >= 0 ? '↑' : '↓'}
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} Stock Predictor | Powered by LSTM & News Sentiment Analysis</p>
      </footer>
    </div>
  );
}

export default StockPredictor;