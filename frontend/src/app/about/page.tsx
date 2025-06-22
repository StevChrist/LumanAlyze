'use client'

import React from 'react';

const AboutPage: React.FC = () => {
  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#014F39',
      color: '#FBF7C7',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <main style={{
        flex: 1,
        maxWidth: '800px',
        margin: '0 auto',
        padding: '80px 32px',
        textAlign: 'center'
      }}>
        <h1 style={{
          fontSize: '48px',
          fontWeight: '700',
          fontFamily: 'Montserrat, sans-serif',
          marginBottom: '24px'
        }}>
          About LumenALYZE
        </h1>
        
        <p style={{
          fontSize: '20px',
          fontFamily: 'Lora, serif',
          lineHeight: '1.8',
          marginBottom: '32px',
          color: 'rgba(251, 247, 199, 0.9)'
        }}>
          LumenALYZE is a machine learning platform designed to make data analysis easy and accessible to everyone. With an intuitive interface and powerful algorithms, you can analyze data without the need for in-depth programming skills.        </p>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '24px',
          marginTop: '48px'
        }}>
          <div style={{
            backgroundColor: 'rgba(251, 247, 199, 0.05)',
            padding: '24px',
            borderRadius: '12px',
            border: '1px solid rgba(251, 247, 199, 0.2)'
          }}>
            <h3 style={{
              fontSize: '20px',
              fontFamily: 'Lora, serif',
              marginBottom: '12px'
            }}>
              🤖 Machine Learning
            </h3>
            <p style={{
              fontSize: '14px',
              fontFamily: 'Lora, serif',
              color: 'rgba(251, 247, 199, 0.8)'
            }}>
              Prediction, Anomaly Detection, dan Segmentation
            </p>
          </div>

          <div style={{
            backgroundColor: 'rgba(251, 247, 199, 0.05)',
            padding: '24px',
            borderRadius: '12px',
            border: '1px solid rgba(251, 247, 199, 0.2)'
          }}>
            <h3 style={{
              fontSize: '20px',
              fontFamily: 'Lora, serif',
              marginBottom: '12px'
            }}>
              📊 Visualization
            </h3>
            <p style={{
              fontSize: '14px',
              fontFamily: 'Lora, serif',
              color: 'rgba(251, 247, 199, 0.8)'
            }}>
              Interactive charts and dashboards to easily understand data
            </p>
          </div>

          <div style={{
            backgroundColor: 'rgba(251, 247, 199, 0.05)',
            padding: '24px',
            borderRadius: '12px',
            border: '1px solid rgba(251, 247, 199, 0.2)'
          }}>
            <h3 style={{
              fontSize: '20px',
              fontFamily: 'Lora, serif',
              marginBottom: '12px'
            }}>
            ⚡ User-Friendly
            </h3>
            <p style={{
              fontSize: '14px',
              fontFamily: 'Lora, serif',
              color: 'rgba(251, 247, 199, 0.8)'
            }}>
              No coding required - upload data and get insights in minutes
            </p>
          </div>
        </div>
        <p style={{
          fontSize: '20px',
          fontFamily: 'Lora, serif',
          marginTop: '48px',
          lineHeight: '1.8',
          marginBottom: '32px',
          color: 'rgba(251, 247, 199, 0.9)'
        }}>
          This project was created by myself and assisted by LLM in the form of Perplexity with Claude 4.0 and Github Copilot with GPT-4.1.
        </p>
      </main>
    </div>
  );
};

export default AboutPage;
