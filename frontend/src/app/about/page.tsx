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
          LumenALYZE adalah platform machine learning yang dirancang untuk membuat analisis data menjadi mudah dan accessible untuk semua orang. Dengan interface yang intuitif dan algoritma yang powerful, Anda dapat menganalisis data tanpa perlu keahlian programming yang mendalam.
        </p>

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
              Prediction, Anomaly Detection, dan Segmentation dengan algoritma terdepan
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
              📊 Visualisasi
            </h3>
            <p style={{
              fontSize: '14px',
              fontFamily: 'Lora, serif',
              color: 'rgba(251, 247, 199, 0.8)'
            }}>
              Interactive charts dan dashboard untuk memahami data dengan mudah
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
              No coding required - upload data dan dapatkan insights dalam hitungan menit
            </p>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AboutPage;
