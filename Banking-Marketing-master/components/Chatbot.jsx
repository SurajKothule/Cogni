"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { API_CONFIG } from "../config/api";

const Chatbot = ({ onClose }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm Cogni, your banking assistant. I can help you with loan applications. Which type of loan are you interested in?",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [currentLoanType, setCurrentLoanType] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const API_BASE_URL = API_CONFIG.BASE_URL;

  const loanTypes = [
    {
      id: "education",
      text: "Education Loan",
      description: "For higher education and courses",
    },
    {
      id: "home",
      text: "Home Loan",
      description: "For purchasing or constructing property",
    },
    {
      id: "personal",
      text: "Personal Loan",
      description: "For personal expenses",
    },
    {
      id: "gold",
      text: "Gold Loan",
      description: "Quick loan against your gold jewelry",
    },
    {
      id: "business",
      text: "Business Loan",
      description: "For business expansion and working capital",
    },
    {
      id: "car",
      text: "Car Loan",
      description: "For purchasing new and used vehicles",
    },
  ];

  const quickActions = [
    { id: 1, text: "Education Loan", action: "loan", loanType: "education" },
    { id: 2, text: "Home Loan", action: "loan", loanType: "home" },
    { id: 3, text: "Personal Loan", action: "loan", loanType: "personal" },
    { id: 4, text: "Gold Loan", action: "loan", loanType: "gold" },
    { id: 5, text: "Business Loan", action: "loan", loanType: "business" },
    { id: 6, text: "Car Loan", action: "loan", loanType: "car" },
    { id: 7, text: "Start Over", action: "restart" },
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (text, sender, isLoading = false) => {
    const newMessage = {
      id: Date.now() + Math.random(),
      text,
      sender,
      timestamp: new Date(),
      isLoading,
    };
    setMessages((prev) => [...prev, newMessage]);
    return newMessage.id;
  };

  const updateMessage = (messageId, text, isLoading = false) => {
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId ? { ...msg, text, isLoading } : msg
      )
    );
  };

  const startLoanApplication = async (loanType) => {
    setIsLoading(true);
    const loadingId = addMessage(
      "Starting your loan application...",
      "bot",
      true
    );

    try {
      const response = await axios.post(
        `${API_BASE_URL}${API_CONFIG.ENDPOINTS.CHAT_START}`,
        {
          loan_type: loanType,
        },
        {
          timeout: 150000, // 15 second timeout for starting session
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      setSessionId(response.data.session_id);
      setCurrentLoanType(loanType);
      updateMessage(loadingId, response.data.message);
    } catch (error) {
      console.error("Error starting chat:", error);
      
      let errorMessage = "Sorry, I encountered an error starting your application. Please try again.";
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = "Connection timed out. Please check your internet connection and try again.";
      } else if (error.response) {
        errorMessage = `Unable to start application: ${error.response.data?.detail || error.response.statusText}`;
      }
      
      updateMessage(loadingId, errorMessage);
    }
    setIsLoading(false);
  };

  const sendMessageToAPI = async (message) => {
    if (!sessionId) return;

    setIsLoading(true);
    const loadingId = addMessage("Processing your information...", "bot", true);

    try {
      const response = await axios.post(
        `${API_BASE_URL}${API_CONFIG.ENDPOINTS.CHAT_MESSAGE}`,
        {
          session_id: sessionId,
          message: message,
        },
        {
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );

      updateMessage(loadingId, response.data.message);

      // If prediction is available, show results
      if (response.data.prediction) {
        const result = response.data.prediction.result;
        const profile = response.data.prediction.profile;
        const requestedAmount = result.requested_amount;
        const eligibleAmount = result.eligible_amount;
        const loanTypeName =
          currentLoanType.charAt(0).toUpperCase() + currentLoanType.slice(1);

        setTimeout(() => {
          let resultMessage = "";

          if (result.status === "APPROVED") {
            // Full approval - customer gets what they asked for
            resultMessage = `
Great News! You're Pre-Approved for ${loanTypeName} Loan

YES! You are eligible for ₹${requestedAmount.toLocaleString()} at ${
              result.interest_rate
            }% per annum

What's Next:
• Your loan application is pre-approved
• Competitive interest rate of ${result.interest_rate}% per annum
• Quick processing and minimal documentation
• Our relationship manager will contact you within 24 hours

Why Choose Us:
• Fastest loan processing in the industry
• Transparent pricing with no hidden charges
• Dedicated customer support throughout the process

Ready to proceed? Our team will reach out to you soon!
            `;
          } else {
            // Partial approval - offer what they can get
            resultMessage = `
Good News! You're Eligible for ${loanTypeName} Loan

You can get up to ₹${eligibleAmount.toLocaleString()} at ${
              result.interest_rate
            }% per annum

Your Application Summary:
• Requested Amount: ₹${requestedAmount.toLocaleString()}
• Approved Amount: ₹${eligibleAmount.toLocaleString()}
• Interest Rate: ${result.interest_rate}% per annum

Special Benefits:
• Pre-approved loan offer valid for 30 days
• Flexible repayment options available
• Option to reapply for higher amount after 6 months
• Priority processing for existing customers

Want to discuss your options? Our loan specialist will call you to explore ways to maximize your loan amount.

Our team will contact you within 24 hours to proceed!
            `;
          }

          addMessage(resultMessage, "bot");
        }, 1000);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      
      let errorMessage = "Sorry, I encountered an error. Please try again.";
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = "Request timed out. Please check your connection and try again.";
      } else if (error.response) {
        // Server responded with error status
        errorMessage = `Server error: ${error.response.data?.detail || error.response.statusText}`;
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = "Unable to connect to server. Please check your connection.";
      }
      
      updateMessage(loadingId, errorMessage);
    }
    setIsLoading(false);
  };

  const handleSendMessage = () => {
    if (inputMessage.trim() && !isLoading) {
      addMessage(inputMessage, "user");

      if (sessionId) {
        sendMessageToAPI(inputMessage);
      } else {
        // Check if user is asking for a loan type
        const lowerMessage = inputMessage.toLowerCase();
        if (lowerMessage.includes("education")) {
          startLoanApplication("education");
        } else if (lowerMessage.includes("home")) {
          startLoanApplication("home");
        } else if (lowerMessage.includes("personal")) {
          startLoanApplication("personal");
        } else {
          setTimeout(() => {
            addMessage(
              "Please select a loan type from the options below, or tell me which type of loan you need (education, home, or personal).",
              "bot"
            );
          }, 500);
        }
      }

      setInputMessage("");
    }
  };

  const handleQuickAction = (action) => {
    if (isLoading) return;

    addMessage(action.text, "user");

    if (action.action === "loan") {
      startLoanApplication(action.loanType);
    } else if (action.action === "restart") {
      setSessionId(null);
      setCurrentLoanType(null);
      setMessages([
        {
          id: 1,
          text: "Hello! I'm Cogni, your banking assistant. I can help you with loan applications. Which type of loan are you interested in?",
          sender: "bot",
          timestamp: new Date(),
        },
      ]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[600px] w-[405px] bg-white rounded-lg shadow-2xl border border-gray-200 overflow-hidden font-sans">
      {/* Navbar */}
      <div className="bg-[#000048] text-white px-6 py-4 flex items-center justify-between">
        <div className="flex items-center">
          <h1 className="text-xl font-bold tracking-tight">Cogni</h1>
          <div className="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
        </div>
        <button
          onClick={onClose}
          className="text-2xl font-light hover:bg-white/10 rounded-full w-8 h-8 flex items-center justify-center transition-colors duration-200"
          aria-label="Close chatbot"
        >
          ×
        </button>
      </div>

      {/* Message Area */}
      <div className="flex-1 overflow-y-auto p-4 bg-white">
        <div className="space-y-3">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.sender === "bot"
                    ? "bg-[#F2F4F8] text-gray-800 rounded-tl-sm"
                    : "bg-[#000048] text-white rounded-tr-sm"
                }`}
              >
                {message.isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-500">Processing...</span>
                  </div>
                ) : (
                  <div className="text-sm leading-relaxed whitespace-pre-line">
                    {message.text}
                  </div>
                )}
                <p
                  className={`text-xs mt-1 ${
                    message.sender === "bot" ? "text-gray-500" : "text-blue-200"
                  }`}
                >
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Action Buttons */}
        
        <div className="px-4 py-3 bg-gray-50 border-t border-gray-200">
          <div className="flex flex-wrap gap-2">
            {quickActions.map((action) => (
              <button
                key={action.id}
                onClick={() => handleQuickAction(action)}
                disabled={isLoading}
                className="px-4 py-2 text-sm text-[#000048] border border-[#000048] rounded-full hover:bg-[#000048] hover:text-white transition-all duration-200 whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {action.text}
              </button>
            ))}
          </div>
        </div>
      

      {/* Input Area */}
      {/* Input Area */}
<div className="px-4 py-3 bg-white border-t border-gray-200">
  <div className="flex items-center space-x-3">
    <input
      type="text"
      value={inputMessage}
      onChange={(e) => setInputMessage(e.target.value)}
      onKeyPress={handleKeyPress}
      placeholder={
        sessionId
          ? "Tell me about your loan requirements..."
          : "Type your message or select a loan type..."
      }
      disabled={isLoading}
      autoFocus // This keeps the input focused
      className="flex-1 px-4 py-3 text-sm border text-gray-500 border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-[#000048]/20 focus:border-[#000048] disabled:opacity-50"
      ref={input => input && input.focus()} // Additional focus management
    />
    <button
      onClick={handleSendMessage}
      disabled={!inputMessage.trim() || isLoading}
      className="p-3 bg-[#000048] text-white rounded-full hover:bg-[#000048]/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200"
    >
      <svg
        className="w-5 h-5 transform rotate-45"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
        />
      </svg>
    </button>
  </div>
</div>
    </div>
  );
};

export default Chatbot;
