'use client'

import React from 'react';

interface ChartContainerProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  loading?: boolean;
}

const ChartContainer: React.FC<ChartContainerProps> = ({ 
  children, 
  title, 
  subtitle, 
  loading = false 
}) => {
  if (loading) {
    return (
      <div style={{
        backgroundColor: 'rgba(251, 247, 199, 0.05)',
        border: '1px solid rgba(251, 247, 199, 0.2)',
        borderRadius: '8px',
        padding: '24px',
        marginBottom: '24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px'
      }}>
        <div style={{
          color: '#FBF7C7',
          fontSize: '18px',
          fontFamily: 'Lora, serif'
        }}>
          Loading visualization...
        </div>
      </div>
    );
  }

  return (
    <div style={{
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      border: '1px solid rgba(251, 247, 199, 0.2)',
      borderRadius: '8px',
      padding: '24px',
      marginBottom: '24px'
    }}>
      {title && (
        <div style={{ marginBottom: '16px' }}>
          <h3 style={{
            fontSize: '20px',
            color: '#FBF7C7',
            fontFamily: 'Lora, serif',
            margin: '0 0 4px 0'
          }}>
            {title}
          </h3>
          {subtitle && (
            <p style={{
              fontSize: '14px',
              color: 'rgba(251, 247, 199, 0.8)',
              margin: 0,
              fontFamily: 'Lora, serif'
            }}>
              {subtitle}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
};

export default ChartContainer;
