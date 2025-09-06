'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function PersonalBanking() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Hero Section */}
      <section className="bg-[#000048] text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">Personal Banking</h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              Tailored financial solutions for your personal journey. Experience banking that understands you.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-16">
        {/* Accounts Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Bank Accounts</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Savings Account",
                features: ["4.5% interest rate", "Zero balance requirement", "Digital onboarding", "Free debit card"],
                icon: "💰",
                cta: "Open Account"
              },
              {
                title: "Salary Account",
                features: ["Premium debit card", "Zero charges", "Personal loan pre-approval", "Priority banking"],
                icon: "💳",
                cta: "Get Started"
              },
              {
                title: "Senior Citizen Account",
                features: ["Higher interest rates", "Free insurance", "Dedicated relationship manager", "Medical benefits"],
                icon: "👵",
                cta: "Learn More"
              }
            ].map((account, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-2xl p-8 hover:shadow-xl transition-shadow">
                <div className="text-4xl mb-4">{account.icon}</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-4">{account.title}</h3>
                <ul className="space-y-2 mb-6">
                  {account.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center text-gray-600">
                      <span className="w-2 h-2 bg-[#000048] rounded-full mr-3"></span>
                      {feature}
                    </li>
                  ))}
                </ul>
                <button className="w-full bg-[#000048] text-white py-3 px-6 rounded-lg hover:bg-[#000048]/90 transition-colors">
                  {account.cta}
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Digital Banking Features */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Digital Banking</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: "📱", title: "Mobile Banking", desc: "Bank anywhere, anytime" },
              { icon: "💻", title: "Internet Banking", desc: "Full-service online banking" },
              { icon: "🔄", title: "Instant Transfers", desc: "24/7 fund transfers" },
              { icon: "🔐", title: "Biometric Login", desc: "Secure fingerprint access" }
            ].map((feature, index) => (
              <div key={index} className="text-center p-6 bg-[#F2F4F8] rounded-2xl">
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-[#000048] mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Loans Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Loan Products</h2>
          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-[#000048] text-white p-8 rounded-2xl">
              <h3 className="text-2xl font-bold mb-6">Home Loans</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Interest Rate</span>
                  <span className="font-bold">8.4% - 9.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Maximum Amount</span>
                  <span className="font-bold">₹5 Crores</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Tenure</span>
                  <span className="font-bold">Up to 30 years</span>
                </div>
              </div>
              <button className="w-full bg-white text-[#000048] py-3 px-6 rounded-lg mt-6 font-semibold hover:bg-gray-100">
                Apply for Home Loan
              </button>
            </div>

            <div className="grid grid-cols-2 gap-6">
              {[
                { title: "Personal Loan", rate: "10.5%", amount: "₹25 Lakhs", icon: "👨‍💼" },
                { title: "Car Loan", rate: "8.9%", amount: "₹50 Lakhs", icon: "🚗" },
                { title: "Education Loan", rate: "8.0%", amount: "₹1 Crore", icon: "🎓" },
                { title: "Gold Loan", rate: "9.2%", amount: "₹20 Lakhs", icon: "🥇" }
              ].map((loan, index) => (
                <div key={index} className="bg-white border border-gray-200 p-6 rounded-2xl text-center">
                  <div className="text-2xl mb-2">{loan.icon}</div>
                  <h4 className="font-semibold text-[#000048] mb-2">{loan.title}</h4>
                  <p className="text-sm text-gray-600 mb-1">From {loan.rate}</p>
                  <p className="text-xs text-gray-500">Up to {loan.amount}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Investment Options */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Investment Solutions</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Fixed Deposits",
                return: "7.5% returns",
                duration: "7 days to 10 years",
                features: ["Flexible tenure", "Loan against FD", "Auto-renewal"],
                icon: "📈"
              },
              {
                title: "Mutual Funds",
                return: "Market-linked returns",
                duration: "Systematic investment",
                features: ["Diverse portfolios", "Expert guidance", "Tax benefits"],
                icon: "📊"
              },
              {
                title: "Insurance",
                return: "Life coverage + returns",
                duration: "Long-term protection",
                features: ["Life insurance", "Health plans", "Retirement solutions"],
                icon: "🛡️"
              }
            ].map((investment, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-2xl p-8 text-center">
                <div className="text-4xl mb-4">{investment.icon}</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-2">{investment.title}</h3>
                <p className="text-blue-600 font-medium mb-2">{investment.return}</p>
                <p className="text-gray-600 text-sm mb-4">{investment.duration}</p>
                <ul className="space-y-1 text-sm text-gray-600">
                  {investment.features.map((feature, idx) => (
                    <li key={idx}>• {feature}</li>
                  ))}
                </ul>
                <button className="w-full border border-[#000048] text-[#000048] py-2 px-6 rounded-lg mt-6 hover:bg-[#000048] hover:text-white transition-colors">
                  Invest Now
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Benefits Section */}
        <section className="bg-[#000048] text-white p-12 rounded-2xl">
          <h2 className="text-3xl font-bold text-center mb-12">Why Choose CogniBank?</h2>
          <div className="grid md:grid-cols-4 gap-8">
            {[
              { icon: "⚡", title: "Instant Approval", desc: "Quick processing for all services" },
              { icon: "🎯", title: "Zero Charges", desc: "No hidden fees or charges" },
              { icon: "🔒", title: "256-bit Encryption", desc: "Military-grade security" },
              { icon: "👥", title: "24/7 Support", desc: "Always here to help you" }
            ].map((benefit, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl mb-4">{benefit.icon}</div>
                <h3 className="text-lg font-semibold mb-2">{benefit.title}</h3>
                <p className="text-blue-200 text-sm">{benefit.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="text-center py-16">
          <h2 className="text-3xl font-bold text-[#000048] mb-6">Ready to Get Started?</h2>
          <p className="text-gray-600 text-lg mb-8">Join millions of satisfied customers who trust CogniBank</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-[#000048] text-white px-8 py-3 rounded-lg hover:bg-[#000048]/90 transition-colors">
              Open an Account
            </button>
            <button className="border border-[#000048] text-[#000048] px-8 py-3 rounded-lg hover:bg-[#000048] hover:text-white transition-colors">
              Download App
            </button>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}