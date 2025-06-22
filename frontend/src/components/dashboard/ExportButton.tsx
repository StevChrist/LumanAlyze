'use client'

import React, { useState } from 'react';

interface ExportButtonProps {
  data: Record<string, unknown>;
  filename?: string;
  formats?: string[];
}

const ExportButton: React.FC<ExportButtonProps> = ({ 
  data, 
  // Hapus filename jika tidak dipakai
  formats = ['csv', 'json', 'excel']
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState('csv');

  const handleExport = async () => {
    setIsExporting(true);
    
    try {
      const response = await fetch('http://localhost:8000/export-results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          export_format: selectedFormat,
          analysis_results: data
        })
      });

      if (!response.ok) {
        throw new Error('Export failed');
      }

      const result = await response.json();
      
      // Create download link
      const link = document.createElement('a');
      link.href = `data:${result.content_type};base64,${result.content}`;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
      <select
        value={selectedFormat}
        onChange={(e) => setSelectedFormat(e.target.value)}
        style={{
          padding: '8px 12px',
          borderRadius: '4px',
          border: '1px solid rgba(251, 247, 199, 0.3)',
          backgroundColor: 'rgba(251, 247, 199, 0.1)',
          color: '#FBF7C7',
          fontFamily: 'Lora, serif'
        }}
      >
        {formats.map(format => (
          <option key={format} value={format}>
            {format.toUpperCase()}
          </option>
        ))}
      </select>
      
      <button
        onClick={handleExport}
        disabled={isExporting}
        style={{
          backgroundColor: '#FBF7C7',
          color: '#120F0A',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 16px',
          fontFamily: 'Lora, serif',
          fontWeight: '700',
          cursor: isExporting ? 'wait' : 'pointer',
          opacity: isExporting ? 0.7 : 1
        }}
      >
        {isExporting ? 'Exporting...' : 'Export'}
      </button>
    </div>
  );
};

export default ExportButton;