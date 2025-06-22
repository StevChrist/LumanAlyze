'use client'

import React from 'react';
import Spinner from './Spinner';

interface LoadingOverlayProps {
  isVisible: boolean;
  message?: string;
  analysisType?: string;
}

const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ 
  isVisible, 
  message, 
  analysisType = 'analysis' 
}) => {
  if (!isVisible) return null;

  const getLoadingMessage = () => {
    if (message) return message;
    
    switch (analysisType) {
      case 'prediction':
        return 'Running prediction model...';
      case 'anomaly':
        return 'Detecting anomalies...';
      case 'segmentation':
        return 'Performing data segmentation...';
      case 'preprocessing':
        return 'Preprocessing data...';
      default:
        return 'Processing your data...';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(1, 79, 57, 0.9)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 9999,
      backdropFilter: 'blur(5px)'
    }}>
      <div style={{
        backgroundColor: '#014F39',
        padding: '40px',
        borderRadius: '16px',
        border: '2px solid rgba(251, 247, 199, 0.3)',
        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)',
        maxWidth: '400px',
        width: '90%',
        textAlign: 'center'
      }}>
        <Spinner size="large" message={getLoadingMessage()} />
        
        <div style={{
          marginTop: '20px',
          padding: '16px',
          backgroundColor: 'rgba(251, 247, 199, 0.05)',
          borderRadius: '8px',
          border: '1px solid rgba(251, 247, 199, 0.1)'
        }}>
          <p style={{
            color: 'rgba(251, 247, 199, 0.8)',
            fontFamily: 'Lora, serif',
            fontSize: '14px',
            margin: 0,
            lineHeight: '1.5'
          }}>
            Please wait while we process your data. This may take a few moments depending on the size of your dataset.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoadingOverlay;
