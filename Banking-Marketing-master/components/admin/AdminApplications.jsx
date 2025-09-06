"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../../config/api';

export default function AdminApplications() {
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedLoanType, setSelectedLoanType] = useState('education');
  const [limit, setLimit] = useState(10);

  const loanTypes = [
    { key: 'education', name: 'Education', icon: '🎓' },
    { key: 'home', name: 'Home', icon: '🏠' },
    { key: 'personal', name: 'Personal', icon: '👤' },
    { key: 'gold', name: 'Gold', icon: '🥇' },
    { key: 'business', name: 'Business', icon: '💼' },
    { key: 'car', name: 'Car', icon: '🚗' }
  ];

  useEffect(() => {
    fetchApplications();
  }, [selectedLoanType, limit]);

  const fetchApplications = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/admin/applications/${selectedLoanType}?limit=${limit}`);
      setApplications(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching applications:', err);
      setError('Failed to load applications. Make sure the backend server is running.');
      // Mock data for demo purposes
      setApplications([
        {
          session_id: 'abc123def456',
          timestamp: new Date().toISOString(),
          customer_info: {
            name: 'John Doe',
            email: 'john.doe@email.com',
            phone: '9876543210'
          },
          status: 'completed',
          prediction_result: {
            result: {
              approved_amount: 250000,
              eligible_amount: 250000,
              interest_rate: 8.5,
              status: 'APPROVED'
            }
          }
        },
        {
          session_id: 'xyz789uvw012',
          timestamp: new Date(Date.now() - 86400000).toISOString(),
          customer_info: {
            name: 'Jane Smith',
            email: 'jane.smith@email.com',
            phone: '9123456789'
          },
          status: 'completed',
          prediction_result: {
            result: {
              approved_amount: 150000,
              eligible_amount: 150000,
              interest_rate: 9.2,
              status: 'PARTIAL_APPROVAL'
            }
          }
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'approved': 
      case 'approved': return 'text-green-600 bg-green-100';
      case 'partial approval':
      case 'partial_approval': return 'text-yellow-600 bg-yellow-100';
      case 'rejected': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
          <div className="flex flex-col sm:flex-row gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Loan Type
              </label>
              <select
                value={selectedLoanType}
                onChange={(e) => setSelectedLoanType(e.target.value)}
                className="border border-gray-300 text-black rounded-md px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {loanTypes.map((type) => (
                  <option key={type.key} value={type.key}>
                    {type.icon} {type.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Records
              </label>
              <select
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value))}
                className="border border-gray-300 text-black rounded-md px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>
          </div>

          <button
            onClick={fetchApplications}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
        </div>
      )}

      {/* Applications List */}
      <div className="bg-white rounded-lg shadow-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Recent {loanTypes.find(t => t.key === selectedLoanType)?.name} Loan Applications
          </h3>
        </div>

        {loading ? (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : applications.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-500">No applications found for {selectedLoanType} loans.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {applications.map((app, index) => (
              <div key={app.session_id} className="p-6 hover:bg-gray-50">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-4 mb-3">
                      <div className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                        #{index + 1}
                      </div>
                      <div className="text-sm text-gray-500">
                        ID: {app.session_id.substring(0, 8)}...
                      </div>
                      <div className="text-sm text-gray-500">
                        {formatDate(app.timestamp)}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                      <div>
                        <p className="text-sm font-medium text-gray-900">Customer Name</p>
                        <p className="text-sm text-gray-600">{app.customer_info.name}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">Email</p>
                        <p className="text-sm text-gray-600">{app.customer_info.email}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">Phone</p>
                        <p className="text-sm text-gray-600">{app.customer_info.phone}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-gray-700">Status:</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        app.status === 'completed' ? 'text-green-600 bg-green-100' : 'text-yellow-600 bg-yellow-100'
                      }`}>
                        {app.status}
                      </span>
                    </div>
                  </div>

                  {app.prediction_result && app.prediction_result.result && (
                    <div className="mt-4 lg:mt-0 lg:ml-6 lg:text-right">
                      <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                        <div>
                          <p className="text-sm font-medium text-gray-700">Eligible Amount</p>
                          <p className="text-lg font-bold text-green-600">
                            ₹{(app.prediction_result.result.eligible_amount || app.prediction_result.result.approved_amount || 0).toLocaleString()}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-700">Interest Rate</p>
                          <p className="text-sm text-gray-900">{app.prediction_result.result.interest_rate || 0}%</p>
                        </div>
                        <div>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            getStatusColor(app.prediction_result.result.status)
                          }`}>
                            {app.prediction_result.result.status || 'Unknown'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}