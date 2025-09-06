"use client";

import { useState, useEffect } from 'react';

// Define loan types configuration
const loanTypes = {
  home: { name: "Home Loan", min: 500000, max: 10000000, baseRate: 8.5, minRate: 7, maxRate: 14 },
  car: { name: "Car Loan", min: 100000, max: 2000000, baseRate: 12, minRate: 9, maxRate: 24 },
  business: { name: "Business Loan", min: 100000, max: 5000000, baseRate: 15, minRate: 10, maxRate: 26 },
  gold: { name: "Gold Loan", min: 50000, max: 1000000, baseRate: 12, minRate: 8, maxRate: 18 },
  personal: { name: "Personal Loan", min: 50000, max: 2000000, baseRate: 12, minRate: 7, maxRate: 15 },
  education: { name: "Education Loan", min: 50000, max: 1000000, baseRate: 10, minRate: 7, maxRate: 18 }
};

export default function LoanCalculator() {
  const [loanType, setLoanType] = useState('home');
  const [loanAmount, setLoanAmount] = useState(500000);
  const [cibilScore, setCibilScore] = useState(750);
  const [tenure, setTenure] = useState(5);
  const [interestRate, setInterestRate] = useState(0);
  const [eligibleAmount, setEligibleAmount] = useState(0);
  const [monthlyPayment, setMonthlyPayment] = useState(0);
  const [totalPayment, setTotalPayment] = useState(0);
  const [totalInterest, setTotalInterest] = useState(0);

  // Calculate interest rate based on loan type and CIBIL score
  useEffect(() => {
    const loanConfig = loanTypes[loanType];
    const cibilFactor = (900 - cibilScore) / 600; // 0 (best) to 1 (worst)
    const rateRange = loanConfig.maxRate - loanConfig.minRate;
    const calculatedRate = loanConfig.minRate + (rateRange * cibilFactor);
    
    setInterestRate(parseFloat(calculatedRate.toFixed(2)));
  }, [loanType, cibilScore]);

  // Calculate eligible amount (simplified logic)
  useEffect(() => {
    const loanConfig = loanTypes[loanType];
    const cibilEligibility = cibilScore / 900; // 0 to 1 factor
    
    // Simple eligibility calculation based on CIBIL score
    let maxEligible = loanConfig.max * cibilEligibility;
    
    // If requested amount is less than eligible amount, use requested amount
    const finalAmount = Math.min(loanAmount, maxEligible);
    
    setEligibleAmount(Math.round(finalAmount));
  }, [loanAmount, cibilScore, loanType]);

  // Calculate loan details using EMI formula
  useEffect(() => {
    if (eligibleAmount > 0 && interestRate > 0 && tenure > 0) {
      const monthlyRate = interestRate / 12 / 100;
      const months = tenure * 12;
      
      // EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
      const emi = eligibleAmount * monthlyRate * Math.pow(1 + monthlyRate, months) / 
                  (Math.pow(1 + monthlyRate, months) - 1);
      
      const total = emi * months;
      const interest = total - eligibleAmount;
      
      setMonthlyPayment(Math.round(emi));
      setTotalPayment(Math.round(total));
      setTotalInterest(Math.round(interest));
    }
  }, [eligibleAmount, interestRate, tenure]);

  // Format currency in Indian numbering system
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      maximumFractionDigits: 0
    }).format(amount);
  };

  return (
    <section className="py-8 lg:py-10 bg-white">
      <div className="max-w-7xl mx-auto px-4 lg:px-6">
        {/* Section Header - Made more compact */}
        <div className="text-center mb-6 lg:mb-8">
          <h2 className="text-2xl lg:text-3xl font-bold text-gray-900 mb-2">
            Loan Calculator
          </h2>
          <p className="text-sm lg:text-base text-gray-600 max-w-2xl mx-auto">
            Calculate your EMI, interest payable and eligibility for different types of loans
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Calculator Inputs - Made more compact */}
          <div className="bg-[#000048] rounded-xl p-4 lg:p-6 shadow-lg">
            <h3 className="text-lg lg:text-xl font-bold text-white mb-4">
              Loan Details
            </h3>
            
            {/* Loan Type Selection */}
            <div className="mb-4">
              <label className="block text-white text-xs lg:text-sm font-medium mb-1">
                Loan Type
              </label>
              <select 
                className="w-full bg-white/10 border border-white/20 rounded-lg py-2 px-3 text-gray-600 focus:outline-none focus:ring-1 focus:ring-[#1E90FF] text-sm"
                value={loanType}
                onChange={(e) => setLoanType(e.target.value)}
              >
                {Object.entries(loanTypes).map(([key, value]) => (
                  <option key={key} value={key}>{value.name}</option>
                ))}
              </select>
            </div>
            
            {/* Loan Amount Slider */}
            <div className="mb-4">
              <div className="flex justify-between items-center mb-1">
                <label className="block text-white text-xs lg:text-sm font-medium">
                  Loan Amount
                </label>
                <span className="text-white font-medium text-sm">₹{formatCurrency(loanAmount)}</span>
              </div>
              <input 
                type="range" 
                min={loanTypes[loanType].min} 
                max={loanTypes[loanType].max} 
                step={10000}
                value={loanAmount}
                onChange={(e) => setLoanAmount(parseInt(e.target.value))}
                className="w-full h-1.5 bg-white/20 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#1E90FF]"
              />
              <div className="flex justify-between text-xs text-white/70 mt-1">
                <span>₹{formatCurrency(loanTypes[loanType].min)}</span>
                <span>₹{formatCurrency(loanTypes[loanType].max)}</span>
              </div>
            </div>
            
            {/* CIBIL Score Slider */}
            <div className="mb-4">
              <div className="flex justify-between items-center mb-1">
                <label className="block text-white text-xs lg:text-sm font-medium">
                  CIBIL Score
                </label>
                <span className="text-white font-medium text-sm">{cibilScore}</span>
              </div>
              <input 
                type="range" 
                min="300" 
                max="900" 
                step="10"
                value={cibilScore}
                onChange={(e) => setCibilScore(parseInt(e.target.value))}
                className="w-full h-1.5 bg-white/20 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#1E90FF]"
              />
              <div className="flex justify-between text-xs text-white/70 mt-1">
                <span>300</span>
                <span>900</span>
              </div>
            </div>
            
            {/* Tenure Slider */}
            <div className="mb-6">
              <div className="flex justify-between items-center mb-1">
                <label className="block text-white text-xs lg:text-sm font-medium">
                  Tenure (Years)
                </label>
                <span className="text-white font-medium text-sm">{tenure} Years</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="30" 
                value={tenure}
                onChange={(e) => setTenure(parseInt(e.target.value))}
                className="w-full h-1.5 bg-white/20 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#1E90FF]"
              />
              <div className="flex justify-between text-xs text-white/70 mt-1">
                <span>1 Year</span>
                <span>30 Years</span>
              </div>
            </div>
            
            {/* Eligibility Indicator */}
            <div className="bg-white/10 rounded-lg p-3 border border-white/20">
              <div className="flex justify-between items-center mb-1">
                <span className="text-white/80 text-xs">Eligible Amount</span>
                <span className="text-white font-bold text-base">₹{formatCurrency(eligibleAmount)}</span>
              </div>
              <div className="w-full bg-white/20 rounded-full h-1.5">
                <div 
                  className="bg-[#1E90FF] h-1.5 rounded-full" 
                  style={{ width: `${(eligibleAmount / loanTypes[loanType].max) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
          
          {/* Results & Graph - Made more compact */}
          <div className="bg-white rounded-xl p-4 lg:p-6 shadow-lg border border-gray-200">
            <h3 className="text-lg lg:text-xl font-bold text-gray-900 mb-4">
              Loan Breakdown
            </h3>
            
            {/* Interest Rate Display */}
            <div className="mb-6 text-center">
              <div className="inline-flex flex-col items-center justify-center bg-[#000048] rounded-xl py-2 px-6">
                <span className="text-white/80 text-xs">Interest Rate</span>
                <span className="text-white text-xl font-bold">{interestRate}%</span>
              </div>
            </div>
            
            {/* Payment Summary */}
            <div className="grid grid-cols-2 gap-3 mb-6">
              <div className="bg-gray-100 rounded-lg p-3 text-center">
                <p className="text-gray-600 text-xs mb-1">Monthly Payment</p>
                <p className="text-[#000048] text-base font-bold">₹{formatCurrency(monthlyPayment)}</p>
              </div>
              <div className="bg-gray-100 rounded-lg p-3 text-center">
                <p className="text-gray-600 text-xs mb-1">Total Interest</p>
                <p className="text-[#000048] text-base font-bold">₹{formatCurrency(totalInterest)}</p>
              </div>
              <div className="bg-gray-100 rounded-lg p-3 text-center">
                <p className="text-gray-600 text-xs mb-1">Total Payment</p>
                <p className="text-[#000048] text-base font-bold">₹{formatCurrency(totalPayment)}</p>
              </div>
              <div className="bg-gray-100 rounded-lg p-3 text-center">
                <p className="text-gray-600 text-xs mb-1">Loan Tenure</p>
                <p className="text-[#000048] text-base font-bold">{tenure} Years</p>
              </div>
            </div>
            
            {/* Pie Chart Visualization */}
            <div className="mb-6">
              <p className="text-gray-700 font-medium text-sm mb-3">Payment Breakdown</p>
              <div className="flex items-center justify-center">
                <div className="relative w-32 h-32">
                  {/* This is a simplified pie chart using CSS conic-gradient */}
                  <div 
                    className="w-full h-full rounded-full"
                    style={{
                      background: `conic-gradient(
                        #000048 0% ${(eligibleAmount / totalPayment * 100)}%,
                        #1E90FF ${(eligibleAmount / totalPayment * 100)}% 100%
                      )`
                    }}
                  ></div>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-[10px] text-gray-600">Principal</span>
                    <span className="text-sm font-bold text-[#000048]">₹{formatCurrency(eligibleAmount)}</span>
                  </div>
                </div>
              </div>
              <div className="flex justify-center gap-3 mt-3">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-[#000048] rounded mr-1"></div>
                  <span className="text-xs text-gray-600">Principal</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-[#1E90FF] rounded mr-1"></div>
                  <span className="text-xs text-gray-600">Interest</span>
                </div>
              </div>
            </div>
            
            {/* Apply Button */}
            <button className="w-full bg-[#000048] text-white py-2 px-4 rounded-lg font-semibold text-sm hover:bg-[#1E90FF] transition-colors duration-300">
              Apply for Loan
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}