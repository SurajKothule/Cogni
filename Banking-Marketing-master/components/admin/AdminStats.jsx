"use client";

import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_CONFIG } from '../../config/api';

export default function AdminStats() {
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loanTypes = [
    { key: 'education', name: 'Education', icon: '🎓', color: 'bg-blue-500' },
    { key: 'home', name: 'Home', icon: '🏠', color: 'bg-green-500' },
    { key: 'personal', name: 'Personal', icon: '👤', color: 'bg-purple-500' },
    { key: 'gold', name: 'Gold', icon: '🥇', color: 'bg-yellow-500' },
    { key: 'business', name: 'Business', icon: '💼', color: 'bg-red-500' },
    { key: 'car', name: 'Car', icon: '🚗', color: 'bg-indigo-500' }
  ];

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_CONFIG.BASE_URL}/admin/stats`);
      setStats(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError('Failed to load statistics. Make sure the backend server is running.');
      // Mock data for demo purposes
      setStats({
        education: { total: 15, completed: 12, approved: 8, partial: 3, average_amount: 250000, average_interest: 8.5 },
        home: { total: 8, completed: 6, approved: 4, partial: 1, average_amount: 1500000, average_interest: 7.2 },
        personal: { total: 22, completed: 18, approved: 12, partial: 4, average_amount: 75000, average_interest: 12.5 },
        gold: { total: 10, completed: 8, approved: 6, partial: 2, average_amount: 180000, average_interest: 9.8 },
        business: { total: 5, completed: 4, approved: 3, partial: 1, average_amount: 800000, average_interest: 10.2 },
        car: { total: 12, completed: 9, approved: 7, partial: 2, average_amount: 450000, average_interest: 8.9 }
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400"></div>
      </div>
    );
  }

  const totalApplications = Object.values(stats).reduce((sum, stat) => sum + (stat.total || 0), 0);
  const totalCompleted = Object.values(stats).reduce((sum, stat) => sum + (stat.completed || 0), 0);
  const totalApproved = Object.values(stats).reduce((sum, stat) => sum + (stat.approved || 0), 0);

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{error}</p>
          <button 
            onClick={fetchStats}
            className="mt-2 bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          >
            Retry
          </button>
        </div>
      )}

      {/* Overall Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-blue-100">
              <span className="text-2xl">📊</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Applications</p>
              <p className="text-2xl font-bold text-gray-900">{totalApplications}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-green-100">
              <span className="text-2xl">✅</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Completed</p>
              <p className="text-2xl font-bold text-gray-900">{totalCompleted}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-yellow-100">
              <span className="text-2xl">🎯</span>
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Approved</p>
              <p className="text-2xl font-bold text-gray-900">{totalApproved}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Loan Type Statistics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {loanTypes.map((loanType) => {
          const loanStats = stats[loanType.key] || {};
          const approvalRate = loanStats.completed > 0 ? ((loanStats.approved / loanStats.completed) * 100).toFixed(1) : 0;
          
          return (
            <div key={loanType.key} className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className={`p-3 rounded-full ${loanType.color} text-white`}>
                    <span className="text-xl">{loanType.icon}</span>
                  </div>
                  <h3 className="ml-3 text-lg font-semibold text-gray-900">
                    {loanType.name} Loans
                  </h3>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Approval Rate</p>
                  <p className="text-lg font-bold text-green-600">{approvalRate}%</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-600">Total Applications</p>
                  <p className="text-xl font-bold text-gray-900">{loanStats.total || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Completed</p>
                  <p className="text-xl font-bold text-blue-600">{loanStats.completed || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Approved</p>
                  <p className="text-xl font-bold text-green-600">{loanStats.approved || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Partial Approval</p>
                  <p className="text-xl font-bold text-yellow-600">{loanStats.partial || 0}</p>
                </div>
              </div>

              {loanStats.completed > 0 && (
                <div className="border-t pt-4">
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Avg. Eligible Amount:</span>
                      <span className="text-sm font-medium text-gray-600">₹{(loanStats.average_amount || 0).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Avg. Interest Rate:</span>
                      <span className="text-sm font-medium text-gray-600">{(loanStats.average_interest || 0).toFixed(2)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}