'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { useState } from 'react';

export default function CreditCards() {
  const [selectedCard, setSelectedCard] = useState('premium');
  const [applicationStep, setApplicationStep] = useState(1);

  const cardTypes = [
    {
      id: 'premium',
      name: 'Premium Rewards',
      benefits: ['5x points on travel', '3x points on dining', '2x points on other purchases', '$300 annual travel credit', 'Priority Pass access'],
      annualFee: '$550',
      introAPR: '0% for 15 months',
      regularAPR: '16.99% - 23.99%',
      bonus: '100,000 points',
      color: 'bg-gradient-to-r from-purple-800 to-blue-900'
    },
    {
      id: 'cashback',
      name: 'Cash Back Plus',
      benefits: ['5% cash back on rotating categories', '3% cash back at grocery stores', '1.5% cash back on all other purchases', 'No annual fee', 'Cash back redemption any time'],
      annualFee: '$0',
      introAPR: '0% for 18 months',
      regularAPR: '14.99% - 22.99%',
      bonus: '$200 cash bonus',
      color: 'bg-gradient-to-r from-amber-600 to-orange-600'
    },
    {
      id: 'travel',
      name: 'Travel Elite',
      benefits: ['10x points on hotels and rental cars', '5x points on flights', '3x points on other travel', 'No foreign transaction fees', 'Complimentary travel insurance'],
      annualFee: '$95',
      introAPR: '0% for 12 months',
      regularAPR: '15.99% - 21.99%',
      bonus: '50,000 points',
      color: 'bg-gradient-to-r from-emerald-700 to-teal-600'
    },
    {
      id: 'student',
      name: 'Student Starter',
      benefits: ['1% cash back on all purchases', 'Good Grades Reward', 'No annual fee', 'Credit limit increase opportunity', 'No late fee first year'],
      annualFee: '$0',
      introAPR: '0% for 6 months',
      regularAPR: '13.99% - 19.99%',
      bonus: '$50 after first purchase',
      color: 'bg-gradient-to-r from-blue-600 to-indigo-700'
    },
    {
      id: 'business',
      name: 'Business Advantage',
      benefits: ['5x points on office supplies', '2x points on gas stations', 'Employee cards at no cost', 'Expense management tools', 'Travel insurance'],
      annualFee: '$125',
      introAPR: '0% for 12 months',
      regularAPR: '14.99% - 20.99%',
      bonus: '75,000 points',
      color: 'bg-gradient-to-r from-gray-800 to-gray-900'
    },
    {
      id: 'secured',
      name: 'Secured Builder',
      benefits: ['Build credit history', 'Credit line equal to security deposit', 'Automatic reviews for credit line increase', 'No annual fee', 'Reports to all three bureaus'],
      annualFee: '$0',
      introAPR: 'N/A',
      regularAPR: '19.99%',
      bonus: 'N/A',
      color: 'bg-gradient-to-r from-slate-700 to-slate-900'
    }
  ];

  const selectedCardData = cardTypes.find(card => card.id === selectedCard);

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Hero Section */}
      <section className="bg-[#000048] text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">CogniBank Credit Cards</h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              Find the perfect card for your lifestyle. Earn rewards, cash back, and travel benefits with our premium selection.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-16">
        {/* Card Comparison Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Compare Our Cards</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {cardTypes.map((card) => (
              <div 
                key={card.id} 
                className={`bg-white border-2 ${selectedCard === card.id ? 'border-[#000048]' : 'border-gray-200'} rounded-2xl p-6 hover:shadow-xl transition-all cursor-pointer`}
                onClick={() => setSelectedCard(card.id)}
              >
                <div className={`${card.color} text-white p-6 rounded-xl mb-6 h-44 flex items-center justify-center`}>
                  <div className="text-center">
                    <div className="text-xl font-bold mb-2">CogniBank</div>
                    <div className="text-lg">●●●● ●●●● ●●●● 1234</div>
                    <div className="flex justify-between mt-4 text-sm">
                      <span>JOHN DOE</span>
                      <span>05/28</span>
                    </div>
                  </div>
                </div>
                
                <h3 className="text-xl font-semibold text-[#000048] mb-4">{card.name}</h3>
                
                <div className="flex justify-between items-center mb-4">
                  <span className="text-gray-600">Annual Fee:</span>
                  <span className="font-bold">{card.annualFee}</span>
                </div>
                
                <div className="flex justify-between items-center mb-4">
                  <span className="text-gray-600">Welcome Bonus:</span>
                  <span className="font-bold text-[#000048]">{card.bonus}</span>
                </div>
                
                <button 
                  className={`w-full ${selectedCard === card.id ? 'bg-[#000048]' : 'bg-gray-200'} text-${selectedCard === card.id ? 'white' : 'gray-800'} py-3 px-6 rounded-lg transition-colors`}
                >
                  {selectedCard === card.id ? 'Selected' : 'Select Card'}
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Selected Card Details */}
        {selectedCardData && (
          <section className="mb-16">
            <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">{selectedCardData.name} Details</h2>
            
            <div className="grid md:grid-cols-2 gap-12">
              <div>
                <div className="bg-white border border-gray-200 rounded-2xl p-8 mb-8">
                  <h3 className="text-2xl font-bold text-[#000048] mb-6">Card Benefits</h3>
                  <ul className="space-y-4">
                    {selectedCardData.benefits.map((benefit, index) => (
                      <li key={index} className="flex items-start">
                        <span className="w-2 h-2 bg-[#000048] rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span>{benefit}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="bg-[#F2F4F8] p-8 rounded-2xl">
                  <h3 className="text-2xl font-bold text-[#000048] mb-6">Rates & Fees</h3>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Annual Fee:</span>
                      <span className="font-semibold">{selectedCardData.annualFee}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Intro APR:</span>
                      <span className="font-semibold">{selectedCardData.introAPR}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Regular APR:</span>
                      <span className="font-semibold">{selectedCardData.regularAPR}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Foreign Transaction Fee:</span>
                      <span className="font-semibold">$0</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="bg-[#000048] text-white p-8 rounded-2xl">
                <h3 className="text-2xl font-bold mb-6">Apply for {selectedCardData.name}</h3>
                
                {applicationStep === 1 ? (
                  <div>
                    <p className="mb-6">Ready to enjoy the benefits of your new CogniBank credit card? Start your application now.</p>
                    
                    <div className="space-y-4 mb-6">
                      <div className="flex items-center">
                        <div className="w-8 h-8 bg-white text-[#000048] rounded-full flex items-center justify-center mr-3">1</div>
                        <span>Check eligibility in minutes</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-8 h-8 bg-white text-[#000048] rounded-full flex items-center justify-center mr-3">2</div>
                        <span>Get a decision instantly</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-8 h-8 bg-white text-[#000048] rounded-full flex items-center justify-center mr-3">3</div>
                        <span>Start using your card right away</span>
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => setApplicationStep(2)}
                      className="w-full bg-white text-[#000048] py-3 px-6 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
                    >
                      Check Eligibility
                    </button>
                  </div>
                ) : (
                  <div>
                    <p className="mb-6">You're pre-approved for the {selectedCardData.name} card!</p>
                    
                    <form className="space-y-4">
                      <div>
                        <label className="block text-blue-100 mb-2">Full Name</label>
                        <input 
                          type="text" 
                          className="w-full bg-blue-900 text-white rounded-lg p-3 focus:ring-2 focus:ring-white focus:outline-none"
                          placeholder="John Doe"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-blue-100 mb-2">Email Address</label>
                        <input 
                          type="email" 
                          className="w-full bg-blue-900 text-white rounded-lg p-3 focus:ring-2 focus:ring-white focus:outline-none"
                          placeholder="john.doe@example.com"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-blue-100 mb-2">Annual Income</label>
                        <input 
                          type="text" 
                          className="w-full bg-blue-900 text-white rounded-lg p-3 focus:ring-2 focus:ring-white focus:outline-none"
                          placeholder="$50,000"
                        />
                      </div>
                      
                      <div className="flex items-center mb-4">
                        <input 
                          type="checkbox" 
                          id="terms-agree" 
                          className="mr-2 h-4 w-4"
                        />
                        <label htmlFor="terms-agree" className="text-blue-100 text-sm">
                          I agree to the terms and conditions
                        </label>
                      </div>
                      
                      <button 
                        type="button"
                        className="w-full bg-white text-[#000048] py-3 px-6 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
                      >
                        Submit Application
                      </button>
                    </form>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Card Features Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Why Choose CogniBank Credit Cards?</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { icon: '🔒', title: 'Zero Fraud Liability', desc: "You're not responsible for unauthorized charges" },
              { icon: '📱', title: 'Mobile Wallet', desc: 'Add your card to Apple Pay, Google Pay, or Samsung Pay' },
              { icon: '🌐', title: 'Contactless Payments', desc: 'Tap to pay for quick and secure transactions' },
              { icon: '🛡️', title: 'Purchase Protection', desc: 'Covers your new purchases for 120 days against damage or theft' },
            ].map((feature, index) => (
              <div key={index} className="text-center p-6 bg-[#F2F4F8] rounded-2xl">
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-[#000048] mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Rewards Program Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Cogni Rewards Program</h2>
          
          <div className="bg-white border border-gray-200 rounded-2xl p-8">
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl mb-4">✈️</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-4">Travel Rewards</h3>
                <p className="text-gray-600">Redeem points for flights, hotels, and rental cars with no blackout dates.</p>
              </div>
              
              <div className="text-center">
                <div className="text-4xl mb-4">💵</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-4">Cash Back</h3>
                <p className="text-gray-600">Get statement credits, direct deposits, or checks for your cash back rewards.</p>
              </div>
              
              <div className="text-center">
                <div className="text-4xl mb-4">🎁</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-4">Gift Cards</h3>
                <p className="text-gray-600">Exchange points for gift cards from hundreds of popular brands and retailers.</p>
              </div>
            </div>
            
            <div className="mt-8 text-center">
              <button className="bg-[#000048] text-white py-3 px-8 rounded-lg hover:bg-[#000048]/90 transition-colors">
                Learn More About Rewards
              </button>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-[#000048] text-white p-12 rounded-2xl text-center">
          <h2 className="text-3xl font-bold mb-6">Find Your Perfect Card</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto">
            With a CogniBank credit card, you'll enjoy premium benefits, exclusive rewards, and world-class security.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-[#000048] px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              Compare All Cards
            </button>
            <button className="border border-white text-white px-8 py-3 rounded-lg hover:bg-white hover:text-[#000048] transition-colors">
              Manage Existing Card
            </button>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}