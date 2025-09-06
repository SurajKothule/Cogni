"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../../config/api';

export default function AdminExports() {
  const [exports, setExports] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadingType, setDownloadingType] = useState(null);

  const loanTypes = [
    { key: 'education', name: 'Education', icon: '🎓', color: 'bg-blue-500' },
    { key: 'home', name: 'Home', icon: '🏠', color: 'bg-green-500' },
    { key: 'personal', name: 'Personal', icon: '👤', color: 'bg-purple-500' },
    { key: 'gold', name: 'Gold', icon: '🥇', color: 'bg-yellow-500' },
    { key: 'business', name: 'Business', icon: '💼', color: 'bg-red-500' },
    { key: 'car', name: 'Car', icon: '🚗', color: 'bg-indigo-500' }
  ];

  useEffect(() => {
    fetchExportInfo();
  }, []);

  const fetchExportInfo = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/admin/exports`);
      setExports(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching export info:', err);
      setError('Failed to load export information. Make sure the backend server is running.');
      // Mock data for demo purposes
      setExports({
        education: { 
          exists: true, 
          size: 15420, 
          lastModified: new Date().toISOString(),
          recordCount: 12
        },
        home: { 
          exists: true, 
          size: 8960, 
          lastModified: new Date(Date.now() - 86400000).toISOString(),
          recordCount: 6
        },
        personal: { 
          exists: true, 
          size: 22340, 
          lastModified: new Date(Date.now() - 172800000).toISOString(),
          recordCount: 18
        },
        gold: { 
          exists: false, 
          size: 0, 
          lastModified: null,
          recordCount: 0
        },
        business: { 
          exists: true, 
          size: 5120, 
          lastModified: new Date(Date.now() - 259200000).toISOString(),
          recordCount: 4
        },
        car: { 
          exists: true, 
          size: 12480, 
          lastModified: new Date(Date.now() - 86400000).toISOString(),
          recordCount: 9
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = async (loanType) => {
    try {
      setDownloadingType(loanType);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/admin/export/${loanType}`, {
        responseType: 'blob'
      });
      
      // Create blob link to download
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${loanType}_applications.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      console.error('Error downloading CSV:', err);
      alert('Failed to download CSV. The file might not exist or the server is not running.');
    } finally {
      setDownloadingType(null);
    }
  };

  const generateReport = async (loanType) => {
    try {
      setDownloadingType(loanType);
      await axios.post(`${API_CONFIG.BASE_URL}/admin/generate-report/${loanType}`);
      // Refresh export info after generating
      await fetchExportInfo();
      alert(`Report generated successfully for ${loanType} loans!`);
    } catch (err) {
      console.error('Error generating report:', err);
      alert('Failed to generate report. Make sure the backend server is running.');
    } finally {
      setDownloadingType(null);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{error}</p>
          <button 
            onClick={fetchExportInfo}
            className="mt-2 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Retry
          </button>
        </div>
      )}

      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">📄 Export Reports</h2>
            <p className="text-gray-600 mt-1">Download CSV reports for loan applications</p>
          </div>
          <button
            onClick={fetchExportInfo}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Export Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loanTypes.map((loanType) => {
          const exportInfo = exports[loanType.key] || {};
          const isDownloading = downloadingType === loanType.key;
          
          return (
            <div key={loanType.key} className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className={`p-3 rounded-full ${loanType.color} text-white`}>
                    <span className="text-xl">{loanType.icon}</span>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {loanType.name} Loans
                    </h3>
                    <p className="text-sm text-gray-500">CSV Export</p>
                  </div>
                </div>
                
                <div className="text-right">
                  {exportInfo.exists ? (
                    <div className="flex items-center text-green-600">
                      <span className="text-sm">✅ Available</span>
                    </div>
                  ) : (
                    <div className="flex items-center text-gray-400">
                      <span className="text-sm">❌ Not Available</span>
                    </div>
                  )}
                </div>
              </div>

              {/* File Information */}
              <div className="space-y-3 mb-4">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Records:</span>
                  <span className="text-sm font-medium text-gray-600">{exportInfo.recordCount || 0}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">File Size:</span>
                  <span className="text-sm font-medium text-gray-600">{formatFileSize(exportInfo.size || 0)}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Last Updated:</span>
                  <span className="text-sm font-medium text-gray-600">{formatDate(exportInfo.lastModified)}</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={() => downloadCSV(loanType.key)}
                  disabled={!exportInfo.exists || isDownloading}
                  className={`flex-1 px-4 py-2 rounded-md text-sm font-medium ${
                    exportInfo.exists && !isDownloading
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {isDownloading ? (
                    <span className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Downloading...
                    </span>
                  ) : (
                    '📥 Download CSV'
                  )}
                </button>
                
                <button
                  onClick={() => generateReport(loanType.key)}
                  disabled={isDownloading}
                  className={`flex-1 px-4 py-2 rounded-md text-sm font-medium ${
                    !isDownloading
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  {isDownloading ? (
                    <span className="flex items-center justify-center">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Generating...
                    </span>
                  ) : (
                    '🔄 Generate Report'
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-2">📋 Instructions</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• <strong>Download CSV:</strong> Download existing CSV reports for completed applications</li>
          <li>• <strong>Generate Report:</strong> Create a new CSV report with the latest data</li>
          <li>• <strong>File Location:</strong> Reports are stored in customer_data/[loan_type]/reports/</li>
          <li>• <strong>Data Included:</strong> Customer info, application details, and prediction results</li>
        </ul>
      </div>
    </div>
  );
}