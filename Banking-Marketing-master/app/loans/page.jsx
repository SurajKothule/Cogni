'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import Chatbot from '@/components/Chatbot';
import { API_CONFIG } from '@/config/api';
import axios from 'axios';

export default function Loans() {
  const [showChatbot, setShowChatbot] = useState(false);
  const [loanTypes, setLoanTypes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loanAmount, setLoanAmount] = useState(500000);
  const [interestRate, setInterestRate] = useState(10);
  const [tenure, setTenure] = useState(5);
  const [emi, setEmi] = useState(0);

  useEffect(() => {
    fetchLoanTypes();
    calculateEMI();
  }, []);

  useEffect(() => {
    calculateEMI();
  }, [loanAmount, interestRate, tenure]);

  const fetchLoanTypes = async () => {
    try {
      const response = await axios.get(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.LOAN_TYPES}`);
      const types = response.data.available_types.map(type => ({
        id: type,
        title: response.data.descriptions[type] ? 
          type.charAt(0).toUpperCase() + type.slice(1) + " Loans" : 
          type.charAt(0).toUpperCase() + type.slice(1),
        description: response.data.descriptions[type] || `Loans for ${type} purposes`,
        icon: getIconForLoanType(type),
        features: getFeaturesForLoanType(type)
      }));
      setLoanTypes(types);
    } catch (error) {
      console.error('Error fetching loan types:', error);
      // Fallback to default loan types
      setLoanTypes(getDefaultLoanTypes());
    } finally {
      setLoading(false);
    }
  };

  const getIconForLoanType = (type) => {
    const icons = {
      home: "🏠",
      business: "💼", 
      personal: "👤",
      car: "🚗",
      gold: "🥇",
      education: "🎓"
    };
    return icons[type] || "💰";
  };

  const getFeaturesForLoanType = (type) => {
    const features = {
      home: ["Up to 30-year terms", "Low interest rates", "Quick approval", "Flexible repayment"],
      business: ["High loan amounts", "Quick disbursal", "Flexible repayment", "Growth financing"],
      personal: ["No collateral needed", "Instant approval", "Flexible tenure", "Minimal documentation"],
      car: ["100% on-road funding", "Competitive rates", "Fast processing", "Easy documentation"],
      gold: ["High value per gram", "Minimal documentation", "Instant disbursal", "Secure storage"],
      education: ["Education funding", "Flexible repayment", "Competitive rates", "Quick processing"]
    };
    return features[type] || ["Competitive rates", "Quick processing", "Flexible terms", "Easy application"];
  };

  const getDefaultLoanTypes = () => [
    {
      id: "home",
      title: "Home Loans",
      description: "For purchasing, constructing, or renovating residential properties",
      features: ["Up to 30-year terms", "Low interest rates", "Quick approval", "Flexible repayment"],
      icon: "🏠"
    },
    {
      id: "business", 
      title: "Business Loans",
      description: "For business expansion, working capital, and commercial purposes",
      features: ["High loan amounts", "Quick disbursal", "Flexible repayment", "Growth financing"],
      icon: "💼"
    },
    {
      id: "personal",
      title: "Personal Loans", 
      description: "Unsecured loans for personal expenses like medical, travel, wedding, etc.",
      features: ["No collateral needed", "Instant approval", "Flexible tenure", "Minimal documentation"],
      icon: "👤"
    },
    {
      id: "car",
      title: "Car Loans",
      description: "Loans for purchasing new and used cars with flexible repayment options", 
      features: ["100% on-road funding", "Competitive rates", "Fast processing", "Easy documentation"],
      icon: "🚗"
    },
    {
      id: "gold",
      title: "Gold Loans",
      description: "Secured loans against gold jewelry and ornaments",
      features: ["High value per gram", "Minimal documentation", "Instant disbursal", "Secure storage"],
      icon: "🥇"
    },
    {
      id: "education",
      title: "Education Loans",
      description: "Loans for higher education, courses, and academic expenses",
      features: ["Education funding", "Flexible repayment", "Competitive rates", "Quick processing"],
      icon: "🎓"
    }
  ];

  const calculateEMI = () => {
    const principal = loanAmount;
    const rate = interestRate / 100 / 12;
    const time = tenure * 12;
    
    if (rate === 0) {
      setEmi(principal / time);
    } else {
      const emiValue = (principal * rate * Math.pow(1 + rate, time)) / (Math.pow(1 + rate, time) - 1);
      setEmi(emiValue);
    }
  };

  const handleApplyLoan = (loanType) => {
    setShowChatbot(true);
  };

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Hero Section */}
      <section className="bg-[#000048] text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">Loan Solutions</h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              Flexible loan options tailored to your needs. Achieve your dreams with our competitive rates and easy application process.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-16">
        {/* Loan Types Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Our Loan Options</h2>
          {loading ? (
            <div className="flex justify-center items-center h-32">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-[#000048]"></div>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {loanTypes.map((loan, index) => (
                <div key={loan.id} className="bg-white border border-gray-200 rounded-2xl p-8 hover:shadow-xl transition-shadow">
                  <div className="text-4xl mb-4">{loan.icon}</div>
                  <h3 className="text-xl font-semibold text-[#000048] mb-4">{loan.title}</h3>
                  <p className="text-gray-600 text-sm mb-4">{loan.description}</p>
                  <ul className="space-y-2 mb-6">
                    {loan.features.map((feature, idx) => (
                      <li key={idx} className="flex items-center text-gray-600">
                        <span className="w-2 h-2 bg-[#000048] rounded-full mr-3"></span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <button 
                    onClick={() => handleApplyLoan(loan.id)}
                    className="w-full bg-[#000048] text-white py-3 px-6 rounded-lg hover:bg-[#000048]/90 transition-colors"
                  >
                    Apply Now
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Loan Calculator Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Loan Calculator</h2>
          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-[#F2F4F8] p-8 rounded-2xl">
              <h3 className="text-2xl font-bold text-[#000048] mb-6">Calculate Your EMI</h3>
              <div className="space-y-6">
                <div>
                  <label className="block text-gray-700 mb-2">Loan Amount: ₹{loanAmount.toLocaleString()}</label>
                  <input 
                    type="range" 
                    min="50000" 
                    max="10000000" 
                    step="100000" 
                    value={loanAmount}
                    onChange={(e) => setLoanAmount(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-gray-600 text-sm mt-1">
                    <span>₹50,000</span>
                    <span>₹1 Cr</span>
                  </div>
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Interest Rate: {interestRate}%</label>
                  <input 
                    type="range" 
                    min="5" 
                    max="20" 
                    step="0.1" 
                    value={interestRate}
                    onChange={(e) => setInterestRate(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-gray-600 text-sm mt-1">
                    <span>5%</span>
                    <span>20%</span>
                  </div>
                </div>
                <div>
                  <label className="block text-gray-700 mb-2">Loan Tenure: {tenure} years</label>
                  <input 
                    type="range" 
                    min="1" 
                    max="30" 
                    step="1" 
                    value={tenure}
                    onChange={(e) => setTenure(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-gray-600 text-sm mt-1">
                    <span>1 year</span>
                    <span>30 years</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-[#000048] text-white p-8 rounded-2xl flex flex-col justify-center">
              <h3 className="text-2xl font-bold mb-6">Your EMI Estimate</h3>
              <div className="text-4xl font-bold mb-4">₹{Math.round(emi).toLocaleString()}</div>
              <div className="space-y-2 text-blue-100">
                <div className="flex justify-between">
                  <span>Principal amount:</span>
                  <span>₹{loanAmount.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Total interest:</span>
                  <span>₹{Math.round((emi * tenure * 12) - loanAmount).toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Total amount:</span>
                  <span>₹{Math.round(emi * tenure * 12).toLocaleString()}</span>
                </div>
              </div>
              <button 
                onClick={() => setShowChatbot(true)}
                className="w-full bg-white text-[#000048] py-3 px-6 rounded-lg mt-6 font-semibold hover:bg-gray-100 transition-colors"
              >
                Apply for Loan
              </button>
            </div>
          </div>
        </section>

        {/* Application Process */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Simple Application Process</h2>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { icon: "📝", title: "Apply Online", desc: "Fill simple form in 5 minutes" },
              { icon: "📑", title: "Submit Documents", desc: "Upload required documents" },
              { icon: "✅", title: "Get Approval", desc: "Quick verification process" },
              { icon: "💰", title: "Receive Funds", desc: "Amount disbursed to your account" }
            ].map((step, index) => (
              <div key={index} className="text-center p-6 bg-[#F2F4F8] rounded-2xl">
                <div className="text-3xl mb-4">{step.icon}</div>
                <div className="w-8 h-8 bg-[#000048] text-white rounded-full flex items-center justify-center mx-auto mb-4">
                  {index + 1}
                </div>
                <h3 className="text-lg font-semibold text-[#000048] mb-2">{step.title}</h3>
                <p className="text-gray-600 text-sm">{step.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Why Choose Our Loans */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Why Choose Our Loans?</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Competitive Rates",
                desc: "Enjoy some of the most competitive interest rates in the market",
                icon: "📉"
              },
              {
                title: "Flexible Repayment",
                desc: "Choose repayment options that suit your financial situation",
                icon: "🔄"
              },
              {
                title: "Quick Processing",
                desc: "Get approval and disbursal in minimal time with digital processing",
                icon: "⚡"
              }
            ].map((feature, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-2xl p-8 text-center">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-4">{feature.title}</h3>
                <p className="text-gray-600">{feature.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-[#000048] text-white p-12 rounded-2xl text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Apply for a Loan?</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto">
            Our financial experts are ready to help you find the perfect loan solution tailored to your specific needs.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={() => setShowChatbot(true)}
              className="bg-white text-[#000048] px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Apply Now
            </button>
            <button 
              onClick={() => setShowChatbot(true)}
              className="border border-white text-white px-8 py-3 rounded-lg hover:bg-white hover:text-[#000048] transition-colors"
            >
              Speak to an Expert
            </button>
          </div>
        </section>
      </div>

      <Footer />
      
      {/* Chatbot */}
      {showChatbot && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Chatbot onClose={() => setShowChatbot(false)} />
        </div>
      )}
    </div>
  );
}