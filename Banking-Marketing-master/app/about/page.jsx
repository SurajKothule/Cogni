'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';

export default function AboutUs() {
  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Hero Section */}
      <section className="bg-[#000048] text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">About CogniBank</h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              Your Trusted Partner in Financial Excellence. Building brighter futures since 2010.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-16">
        {/* Our Story */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold text-[#000048] mb-6">Our Story</h2>
              <div className="space-y-4 text-gray-700">
                <p className="text-lg leading-relaxed">
                  Founded in 2010, CogniBank emerged with a vision to revolutionize the banking experience. 
                  What started as a small community bank has grown into a trusted financial institution 
                  serving millions of customers nationwide.
                </p>
                <p className="text-lg leading-relaxed">
                  Our journey has been guided by innovation, integrity, and an unwavering commitment 
                  to customer satisfaction. We've consistently embraced technology to make banking 
                  simpler, faster, and more accessible for everyone.
                </p>
              </div>
            </div>
            <div className="bg-[#000048] p-8 rounded-2xl text-white">
              <div className="text-center">
                <div className="text-4xl font-bold mb-2">14+</div>
                <div className="text-lg">Years of Excellence</div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="text-center">
                  <div className="text-2xl font-bold">5M+</div>
                  <div className="text-sm">Happy Customers</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">250+</div>
                  <div className="text-sm">Branches</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">24/7</div>
                  <div className="text-sm">Customer Support</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">99.9%</div>
                  <div className="text-sm">Uptime</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Our Mission & Vision */}
        <section className="mb-16">
          <div className="grid md:grid-cols-2 gap-12">
            <div className="bg-[#F2F4F8] p-8 rounded-2xl">
              <div className="text-[#000048] mb-4">
                <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                  <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-[#000048] mb-4">Our Mission</h3>
              <p className="text-gray-700 leading-relaxed">
                To empower individuals and businesses with innovative financial solutions that drive growth, 
                foster financial literacy, and create lasting value for our communities. We believe in making 
                banking accessible, transparent, and rewarding for everyone.
              </p>
            </div>

            <div className="bg-[#000048] p-8 rounded-2xl text-white">
              <div className="text-blue-200 mb-4">
                <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05 6.981 6.981 0 003 11a7 7 0 1011.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 017 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0113 13a2.99 2.99 0 01-.879 2.121z" clipRule="evenodd"/>
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4">Our Vision</h3>
              <p className="text-blue-100 leading-relaxed">
                To be the most trusted and innovative financial partner, transforming the banking landscape 
                through cutting-edge technology, personalized service, and sustainable practices that benefit 
                both our customers and the environment.
              </p>
            </div>
          </div>
        </section>

        {/* Core Values */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Our Core Values</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Integrity",
                description: "We uphold the highest ethical standards in all our dealings, ensuring transparency and trust.",
                icon: "🤝"
              },
              {
                title: "Innovation",
                description: "We continuously evolve with technology to provide cutting-edge banking solutions.",
                icon: "💡"
              },
              {
                title: "Customer First",
                description: "Our customers are at the heart of everything we do. Their success is our success.",
                icon: "❤️"
              },
              {
                title: "Excellence",
                description: "We strive for perfection in service delivery and operational efficiency.",
                icon: "⭐"
              },
              {
                title: "Community",
                description: "We believe in giving back and supporting the communities we serve.",
                icon: "🌍"
              },
              {
                title: "Security",
                description: "We prioritize the safety and security of our customers' assets and information.",
                icon: "🔒"
              }
            ].map((value, index) => (
              <div key={index} className="text-center p-6 bg-white border border-gray-200 rounded-2xl hover:shadow-lg transition-shadow">
                <div className="text-4xl mb-4">{value.icon}</div>
                <h3 className="text-xl font-semibold text-[#000048] mb-3">{value.title}</h3>
                <p className="text-gray-600">{value.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Leadership Team */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Leadership Team</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { name: "Sarah Chen", role: "CEO & Founder", exp: "25+ years in banking" },
              { name: "Michael Rodriguez", role: "Chief Technology Officer", exp: "AI & Fintech expert" },
              { name: "Priya Sharma", role: "Chief Financial Officer", exp: "Investment banking background" },
              { name: "David Thompson", role: "Customer Experience Director", exp: "15+ years in service excellence" }
            ].map((member, index) => (
              <div key={index} className="text-center">
                <div className="w-24 h-24 bg-[#000048] rounded-full mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
                  {member.name.split(' ').map(n => n[0]).join('')}
                </div>
                <h3 className="text-lg font-semibold text-[#000048]">{member.name}</h3>
                <p className="text-blue-600 font-medium">{member.role}</p>
                <p className="text-sm text-gray-600 mt-1">{member.exp}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Awards & Recognition */}
        <section className="bg-[#000048] text-white p-12 rounded-2xl text-center">
          <h2 className="text-3xl font-bold mb-8">Awards & Recognition</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              "Best Digital Bank 2023",
              "Customer Service Excellence Award",
              "Most Innovative Fintech"
            ].map((award, index) => (
              <div key={index} className="p-6 bg-white/10 rounded-xl">
                <div className="text-2xl mb-2">🏆</div>
                <h3 className="font-semibold">{award}</h3>
              </div>
            ))}
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}