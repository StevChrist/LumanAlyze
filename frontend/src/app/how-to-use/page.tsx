'use client'

import React, { useState } from 'react';

interface StepProps {
  number: number;
  title: string;
  description: string;
  details: string[];
  tips?: string[];
}

const Step: React.FC<StepProps> = ({ number, title, description, details, tips }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div style={{
      backgroundColor: 'rgba(251, 247, 199, 0.05)',
      border: '1px solid rgba(251, 247, 199, 0.2)',
      borderRadius: '12px',
      padding: '24px',
      marginBottom: '24px'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '20px',
        marginBottom: '16px'
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          backgroundColor: '#FBF7C7',
          color: '#014F39',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '20px',
          fontWeight: '700',
          fontFamily: 'Montserrat, sans-serif',
          flexShrink: 0
        }}>
          {number}
        </div>
        <div style={{ flex: 1 }}>
          <h3 style={{
            fontSize: '20px',
            color: '#FBF7C7',
            fontFamily: 'Lora, serif',
            fontWeight: '700',
            margin: '0 0 8px 0'
          }}>
            {title}
          </h3>
          <p style={{
            fontSize: '16px',
            color: 'rgba(251, 247, 199, 0.9)',
            fontFamily: 'Lora, serif',
            margin: '0 0 16px 0',
            lineHeight: '1.6'
          }}>
            {description}
          </p>
        </div>
      </div>

      <button
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          backgroundColor: 'transparent',
          border: '1px solid rgba(251, 247, 199, 0.3)',
          color: '#FBF7C7',
          padding: '8px 16px',
          borderRadius: '6px',
          cursor: 'pointer',
          fontFamily: 'Lora, serif',
          fontSize: '14px',
          marginBottom: isExpanded ? '16px' : '0',
          transition: 'all 0.3s ease'
        }}
      >
        {isExpanded ? 'Hide Details' : 'Show Details'} {isExpanded ? '▲' : '▼'}
      </button>

      {isExpanded && (
        <div style={{
          backgroundColor: 'rgba(251, 247, 199, 0.05)',
          padding: '20px',
          borderRadius: '8px',
          border: '1px solid rgba(251, 247, 199, 0.1)'
        }}>
          <h4 style={{
            fontSize: '16px',
            color: '#FBF7C7',
            fontFamily: 'Lora, serif',
            fontWeight: '600',
            margin: '0 0 12px 0'
          }}>
            Detailed Steps:
          </h4>
          <ul style={{
            margin: '0 0 16px 0',
            paddingLeft: '20px'
          }}>
            {details.map((detail, index) => (
              <li key={index} style={{
                color: 'rgba(251, 247, 199, 0.9)',
                fontFamily: 'Lora, serif',
                fontSize: '14px',
                lineHeight: '1.6',
                marginBottom: '8px'
              }}>
                {detail}
              </li>
            ))}
          </ul>

          {tips && tips.length > 0 && (
            <>
              <h4 style={{
                fontSize: '16px',
                color: '#FBF7C7',
                fontFamily: 'Lora, serif',
                fontWeight: '600',
                margin: '16px 0 12px 0'
              }}>
                💡 Tips:
              </h4>
              <ul style={{
                margin: '0',
                paddingLeft: '20px'
              }}>
                {tips.map((tip, index) => (
                  <li key={index} style={{
                    color: '#ffc107',
                    fontFamily: 'Lora, serif',
                    fontSize: '14px',
                    lineHeight: '1.6',
                    marginBottom: '8px'
                  }}>
                    {tip}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
};

const HowToUsePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('getting-started');

  const tabs = [
    { id: 'getting-started', label: 'Getting Started' },
    { id: 'data-requirements', label: 'Data Requirements' },
    { id: 'analysis-types', label: 'Analysis Types' },
    { id: 'troubleshooting', label: 'Troubleshooting' }
  ];

  const gettingStartedSteps = [
    {
      number: 1,
      title: "Upload Your Data",
      description: "Start by uploading a CSV file containing your dataset for analysis.",
      details: [
        "Click on the upload area or drag and drop your CSV file",
        "Supported formats: .csv, .xlsx, .xls",
        "Maximum file size: 50MB",
        "Ensure your file has at least one numeric column for analysis"
      ],
      tips: [
        "Use UTF-8 encoding for best compatibility",
        "Include column headers in the first row",
        "Remove any merged cells or formatting from Excel files"
      ]
    },
    {
      number: 2,
      title: "Preview Your Data",
      description: "Review the uploaded data to ensure it was processed correctly.",
      details: [
        "Check the data preview table for accuracy",
        "Verify column names and data types",
        "Note the number of rows and columns detected",
        "Identify numeric columns available for analysis"
      ],
      tips: [
        "If data looks incorrect, check your file encoding",
        "Ensure numeric columns contain actual numbers, not text",
        "Missing values are automatically detected and can be handled in preprocessing"
      ]
    },
    {
      number: 3,
      title: "Choose Analysis Type",
      description: "Select the type of machine learning analysis you want to perform.",
      details: [
        "Prediction: Forecast values or classify data points",
        "Anomaly Detection: Identify unusual patterns or outliers",
        "Segmentation: Group similar data points into clusters"
      ],
      tips: [
        "Choose Prediction for forecasting or classification tasks",
        "Use Anomaly Detection to find unusual data points",
        "Select Segmentation to discover hidden patterns in your data"
      ]
    }
  ];

  const dataRequirements = [
    {
      number: 1,
      title: "File Format Requirements",
      description: "Ensure your data file meets the basic format requirements.",
      details: [
        "Supported formats: CSV (.csv), Excel (.xlsx, .xls)",
        "File size limit: 50MB maximum",
        "Encoding: UTF-8, Latin1, or Windows-1252",
        "Structure: Tabular data with rows and columns"
      ],
      tips: [
        "CSV files are processed faster than Excel files",
        "Avoid special characters in column names",
        "Save Excel files as CSV for better compatibility"
      ]
    },
    {
      number: 2,
      title: "Data Structure",
      description: "Organize your data in a proper tabular structure.",
      details: [
        "Each row should represent one observation/record",
        "Each column should represent one variable/feature",
        "Include column headers in the first row",
        "Minimum 2 columns, at least 1 must be numeric"
      ],
      tips: [
        "Remove any summary rows or totals at the bottom",
        "Ensure consistent data types within each column",
        "Avoid merged cells or complex formatting"
      ]
    }
  ];

  const analysisTypes = [
    {
      number: 1,
      title: "Prediction Analysis",
      description: "Forecast numerical values or classify data into categories.",
      details: [
        "Regression: Predict continuous numerical values (prices, temperatures, etc.)",
        "Classification: Predict categories or classes (spam/not spam, pass/fail)",
        "Models available: Random Forest, Neural Network (MLP)",
        "Metrics: R² Score, MSE, RMSE, Accuracy, Precision, Recall"
      ],
      tips: [
        "Use regression for numerical predictions",
        "Use classification for category predictions",
        "Random Forest is good for interpretability",
        "Neural Networks can capture complex patterns"
      ]
    },
    {
      number: 2,
      title: "Anomaly Detection",
      description: "Identify unusual patterns, outliers, or anomalous data points.",
      details: [
        "Detects data points that deviate significantly from normal patterns",
        "Uses Isolation Forest algorithm",
        "Configurable contamination rate (expected percentage of anomalies)",
        "Provides anomaly scores and visual identification"
      ],
      tips: [
        "Start with 10% contamination rate and adjust based on results",
        "Useful for fraud detection, quality control, system monitoring",
        "Review detected anomalies manually to validate results"
      ]
    }
  ];

  const troubleshooting = [
    {
      number: 1,
      title: "File Upload Issues",
      description: "Common problems when uploading data files.",
      details: [
        "File encoding error: Try saving your file with UTF-8 encoding",
        "File too large: Reduce file size or split into smaller files",
        "Invalid format: Ensure file is CSV or Excel format",
        "No numeric columns: Add at least one column with numerical data"
      ],
      tips: [
        "Use 'Save As' in Excel and select 'CSV UTF-8' format",
        "Remove unnecessary columns to reduce file size",
        "Check that numbers are not stored as text"
      ]
    }
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'getting-started':
        return gettingStartedSteps.map((step, index) => (
          <Step key={index} {...step} />
        ));
      case 'data-requirements':
        return dataRequirements.map((step, index) => (
          <Step key={index} {...step} />
        ));
      case 'analysis-types':
        return analysisTypes.map((step, index) => (
          <Step key={index} {...step} />
        ));
      case 'troubleshooting':
        return troubleshooting.map((step, index) => (
          <Step key={index} {...step} />
        ));
      default:
        return gettingStartedSteps.map((step, index) => (
          <Step key={index} {...step} />
        ));
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#014F39',
      color: '#FBF7C7',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Main Content */}
      <main style={{
        flex: 1,
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '40px 32px',
        width: '100%'
      }}>
        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <h1 style={{
            fontSize: '48px',
            fontWeight: '700',
            fontFamily: 'Montserrat, sans-serif',
            color: '#FBF7C7',
            margin: '0 0 16px 0'
          }}>
            How To Use LumenALYZE
          </h1>
          <p style={{
            fontSize: '20px',
            fontFamily: 'Lora, serif',
            color: 'rgba(251, 247, 199, 0.8)',
            maxWidth: '600px',
            margin: '0 auto',
            lineHeight: '1.6'
          }}>
            Complete guide to using LumenALYZE for machine learning data analysis
          </p>
        </div>

        {/* Tabs */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          marginBottom: '40px',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                backgroundColor: activeTab === tab.id ? '#FBF7C7' : 'transparent',
                color: activeTab === tab.id ? '#014F39' : '#FBF7C7',
                border: `1px solid ${activeTab === tab.id ? '#FBF7C7' : 'rgba(251, 247, 199, 0.3)'}`,
                padding: '12px 24px',
                borderRadius: '25px',
                cursor: 'pointer',
                fontFamily: 'Lora, serif',
                fontSize: '16px',
                fontWeight: activeTab === tab.id ? '600' : '400',
                transition: 'all 0.3s ease'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ maxWidth: '800px', margin: '0 auto' }}>
          {renderContent()}
        </div>
      </main>
    </div>
  );
};

export default HowToUsePage;
