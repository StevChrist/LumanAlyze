'use client'

import React from 'react';

interface StatisticsPanelProps {
  statistics: Record<string, unknown>;
  title?: string;
}

const StatisticsPanel: React.FC<StatisticsPanelProps> = ({ 
  statistics, 
  title = "Analysis Statistics" 
}) => {
  const formatValue = (value: unknown): string => {
  if (typeof value === 'number') {
    if (value % 1 === 0) {
      return value.toString();
    } else {
      return value.toFixed(4);
    }
  }
  return String(value);
};

  return (
    <div style={{
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      border: '1px solid rgba(251, 247, 199, 0.2)',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px'
    }}>
      <h3 style={{
        fontSize: '18px',
        marginBottom: '16px',
        color: '#FBF7C7',
        fontFamily: 'Lora, serif'
      }}>
        {title}
      </h3>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '16px'
      }}>
        {Object.entries(statistics).map(([key, value]) => (
          <div
            key={key}
            style={{
              backgroundColor: 'rgba(251, 247, 199, 0.1)',
              padding: '12px',
              borderRadius: '6px',
              textAlign: 'center'
            }}
          >
            <div style={{
              fontSize: '12px',
              color: 'rgba(251, 247, 199, 0.8)',
              marginBottom: '4px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              {key.replace(/_/g, ' ')}
            </div>
            <div style={{
              fontSize: '20px',
              fontWeight: '700',
              color: '#FBF7C7',
              fontFamily: 'Montserrat, sans-serif'
            }}>
              {formatValue(value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatisticsPanel;
