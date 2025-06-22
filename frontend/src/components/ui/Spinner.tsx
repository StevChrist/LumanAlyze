'use client'

import React from 'react';

interface SpinnerProps {
  size?: 'small' | 'medium' | 'large';
  message?: string;
}

const Spinner: React.FC<SpinnerProps> = ({ 
  size = 'medium', 
  message = 'Analyzing data...' 
}) => {
  const sizeClasses = {
    small: { width: '40px', height: '40px', borderWidth: '4px' },
    medium: { width: '60px', height: '60px', borderWidth: '6px' },
    large: { width: '80px', height: '80px', borderWidth: '8px' }
  };

  const currentSize = sizeClasses[size];

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      padding: '40px 20px',
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      borderRadius: '12px',
      border: '1px solid rgba(251, 247, 199, 0.2)',
      margin: '20px 0'
    }}>
      {/* Spinner Animation */}
      <div
        style={{
          width: currentSize.width,
          height: currentSize.height,
          border: `${currentSize.borderWidth} solid rgba(251, 247, 199, 0.2)`,
          borderTop: `${currentSize.borderWidth} solid #FBF7C7`,
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          marginBottom: '16px'
        }}
      />
      
      {/* Loading Message */}
      <p style={{
        color: '#FBF7C7',
        fontFamily: 'Lora, serif',
        fontSize: '16px',
        margin: 0,
        textAlign: 'center'
      }}>
        {message}
      </p>
      
      {/* Pulsing Dots */}
      <div style={{
        display: 'flex',
        gap: '4px',
        marginTop: '12px'
      }}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              width: '8px',
              height: '8px',
              backgroundColor: '#FBF7C7',
              borderRadius: '50%',
              animation: `pulse 1.5s ease-in-out ${i * 0.2}s infinite`
            }}
          />
        ))}
      </div>

      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
          0%, 80%, 100% { 
            opacity: 0.3;
            transform: scale(0.8);
          }
          40% { 
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </div>
  );
};

export default Spinner;
