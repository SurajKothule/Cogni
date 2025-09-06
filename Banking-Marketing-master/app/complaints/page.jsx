'use client';

import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { useState } from 'react';

export default function Complaints() {
  const [complaintType, setComplaintType] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  {/*const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In a real application, you would handle form submission to your backend here
    setIsSubmitted(true);
  };*/}

  return (
    <div className="min-h-screen bg-white">
      <Header />
      
      {/* Hero Section */}
      <section className="bg-[#000048] text-white py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">Customer Support</h1>
            <p className="text-xl md:text-2xl text-blue-100 max-w-3xl mx-auto leading-relaxed">
              We're here to help. Share your concerns and we'll ensure they're addressed promptly.
            </p>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-16">
        {/* Complaint Form Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Submit a Complaint</h2>
          
          {isSubmitted ? (
            <div className="bg-green-50 border border-green-200 rounded-2xl p-8 text-center">
              <div className="text-5xl text-green-500 mb-4">✅</div>
              <h3 className="text-2xl font-bold text-[#000048] mb-4">Complaint Submitted Successfully</h3>
              <p className="text-gray-600 mb-6">
                Thank you for bringing this to our attention. Your complaint has been registered with reference ID: 
                <span className="font-semibold"> CB-{Math.floor(100000 + Math.random() * 900000)}</span>. 
                Our customer support team will contact you within 24 hours.
              </p>
              <button 
                onClick={() => setIsSubmitted(false)}
                className="bg-[#000048] text-white py-3 px-6 rounded-lg hover:bg-[#000048]/90 transition-colors"
              >
                Submit Another Complaint
              </button>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-12">
              <div className="bg-white border border-gray-200 rounded-2xl p-8">
                <h3 className="text-2xl font-bold text-[#000048] mb-6">File Your Complaint</h3>
                <form  className="space-y-6">
                  <div>
                    <label className="block text-gray-700 mb-2">Complaint Type</label>
                    <select 
                      value={complaintType}
                      onChange={(e) => setComplaintType(e.target.value)}
                      className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-[#000048] focus:border-transparent"
                      required
                    >
                      <option value="">Select complaint type</option>
                      <option value="transaction">Transaction Issue</option>
                      <option value="card">Card Related</option>
                      <option value="loan">Loan Related</option>
                      <option value="service">Poor Service</option>
                      <option value="online">Online Banking</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-gray-700 mb-2">Description</label>
                    <textarea 
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      rows={5}
                      className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-[#000048] focus:border-transparent"
                      placeholder="Please describe your issue in detail..."
                      required
                    ></textarea>
                  </div>
                  
                  <div>
                    <label className="block text-gray-700 mb-2">Attachments (Optional)</label>
                    <div className="border border-dashed border-gray-300 rounded-lg p-4 text-center">
                      <div className="text-2xl text-gray-400 mb-2">📎</div>
                      <p className="text-gray-500 text-sm">Drag & drop files here or click to browse</p>
                      <input type="file" className="hidden" id="file-upload" />
                      <label htmlFor="file-upload" className="inline-block mt-2 text-sm text-[#000048] cursor-pointer">
                        Browse files
                      </label>
                    </div>
                  </div>
                  
                  <button 
                    type="submit"
                    className="w-full bg-[#000048] text-white py-3 px-6 rounded-lg hover:bg-[#000048]/90 transition-colors"
                  >
                    Submit Complaint
                  </button>
                </form>
              </div>
              
              <div>
                <div className="bg-[#000048] text-white p-8 rounded-2xl mb-8">
                  <h3 className="text-2xl font-bold mb-6">Need Immediate Assistance?</h3>
                  <div className="space-y-4">
                    <div className="flex items-start">
                      <div className="text-xl mr-4">📞</div>
                      <div>
                        <h4 className="font-semibold">24/7 Customer Support</h4>
                        <p className="text-blue-100">1-800-COGNI-BANK (1-800-264-6422)</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="text-xl mr-4">✉️</div>
                      <div>
                        <h4 className="font-semibold">Email Support</h4>
                        <p className="text-blue-100">support@cognibank.com</p>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="text-xl mr-4">💬</div>
                      <div>
                        <h4 className="font-semibold">Live Chat</h4>
                        <p className="text-blue-100">Available 24/7 on our website and mobile app</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-[#F2F4F8] p-8 rounded-2xl">
                  <h3 className="text-xl font-bold text-[#000048] mb-4">What Happens Next?</h3>
                  <ol className="space-y-4">
                    <li className="flex items-start">
                      <span className="bg-[#000048] text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3">1</span>
                      <span>You'll receive a confirmation email with your complaint reference number</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-[#000048] text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3">2</span>
                      <span>Our support team will review your complaint within 24 hours</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-[#000048] text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3">3</span>
                      <span>We'll work to resolve your issue as quickly as possible</span>
                    </li>
                    <li className="flex items-start">
                      <span className="bg-[#000048] text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3">4</span>
                      <span>You'll receive regular updates until your issue is fully resolved</span>
                    </li>
                  </ol>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* FAQ Section */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Frequently Asked Questions</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-white border border-gray-200 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-[#000048] mb-3">How long does it take to resolve a complaint?</h3>
              <p className="text-gray-600">
                Most complaints are resolved within 3-5 business days. Complex issues may take up to 14 days. We'll keep you updated throughout the process.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-[#000048] mb-3">Can I track the status of my complaint?</h3>
              <p className="text-gray-600">
                Yes, you can track your complaint status using the reference number provided in your confirmation email through our website or mobile app.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-[#000048] mb-3">What if I'm not satisfied with the resolution?</h3>
              <p className="text-gray-600">
                If you're not satisfied, you can request escalation to a senior manager. We also provide information about regulatory bodies for further appeal.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-[#000048] mb-3">What information should I include in my complaint?</h3>
              <p className="text-gray-600">
                Please include your account number, specific details about the issue, dates, transaction IDs if applicable, and any supporting documents.
              </p>
            </div>
          </div>
        </section>

        {/* Resolution Timeline */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold text-[#000048] text-center mb-12">Our Complaint Resolution Process</h2>
          <div className="flex flex-wrap justify-center">
            {[
              { step: "Complaint Received", time: "Within 1 hour", icon: "📥" },
              { step: "Initial Response", time: "Within 24 hours", icon: "⏱️" },
              { step: "Investigation", time: "1-3 days", icon: "🔍" },
              { step: "Resolution", time: "3-5 days", icon: "✅" },
              { step: "Follow-up", time: "Within 7 days", icon: "📞" },
            ].map((item, index) => (
              <div key={index} className="w-full md:w-1/3 lg:w-1/5 p-4">
                <div className="bg-white border border-gray-200 rounded-2xl p-6 text-center h-full">
                  <div className="text-3xl mb-4">{item.icon}</div>
                  <div className="w-10 h-10 bg-[#000048] text-white rounded-full flex items-center justify-center mx-auto mb-4">
                    {index + 1}
                  </div>
                  <h3 className="font-semibold text-[#000048] mb-2">{item.step}</h3>
                  <p className="text-gray-600 text-sm">{item.time}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="bg-[#000048] text-white p-12 rounded-2xl text-center">
          <h2 className="text-3xl font-bold mb-6">We Value Your Feedback</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto">
            Your satisfaction is our priority. Help us improve our services by sharing your experience.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-[#000048] px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              Submit Feedback
            </button>
            <button className="border border-white text-white px-8 py-3 rounded-lg hover:bg-white hover:text-[#000048] transition-colors">
              Check Complaint Status
            </button>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}