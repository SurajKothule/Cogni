'use client';

import { useState, useEffect, useRef } from 'react';
import { usePathname } from 'next/navigation'; // Import usePathname for active state

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const headerRef = useRef(null);
  const pathname = usePathname(); // Get current path for active state

  useEffect(() => {
    const setHeaderVar = () => {
      const h = headerRef.current?.offsetHeight || 0;
      document.documentElement.style.setProperty('--header-h', `${h}px`);
    };
    setHeaderVar();
    const onResize = () => setHeaderVar();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const navLinks = [
    { name: 'Home', href: '/' },
    { name: 'Personal Banking', href: '/personalbanking' },
    { name: 'Loans', href: '/loans' },
    { name: 'Credit Cards', href: '/creditcard' },
    { name: 'Complaints', href: '/complaints' },
    { name: 'About', href: '/about' }
  ];

  return (
    <header ref={headerRef} className="bg-white sticky top-0 z-50 shadow-sm w-full border-b border-[#000048]/20">
      <div className="max-w-7xl mx-auto px-8 py-2">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="flex-shrink-0">
            <h1 className="text-2xl md:text-3xl font-sans tracking-tight">
              <span className="font-normal text-gray-900">Cogni</span>
              <span className="font-bold text-[#000048]">Bank</span>
            </h1>
          </div>

          {/* Desktop Navigation - UPDATED STYLES */}
          <nav className="hidden md:flex flex-1 justify-center">
            <div className="flex items-center space-x-1 font-sans"> {/* Reduced space-x */}
              {navLinks.map((link) => {
                const isActive = pathname === link.href;
                return (
                  <a
                    key={link.name}
                    href={link.href}
                    className={`
                      relative px-4 py-2 text-base font-medium transition-all duration-200 ease-out
                      ${isActive 
                        ? 'text-[#000048] font-semibold' 
                        : 'text-gray-600 hover:text-[#000048]'
                      }
                    `}
                  >
                    {link.name}
                    {/* Active indicator bar */}
                    {isActive && (
                      <span className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-6 h-0.5 bg-[#000048] rounded-full"></span>
                    )}
                    {/* Hover effect */}
                    <span className="absolute inset-0 bg-[#000048] opacity-0 hover:opacity-5 rounded-md transition-opacity duration-200"></span>
                  </a>
                );
              })}
            </div>
          </nav>

          {/* CTA Button -> Bordered Login with dropdown - UPDATED STYLES */}
          <div className="hidden md:block">
            <button className="font-sans text-[#000048] px-5 py-2.5 rounded-md transition-all duration-200 border border-[#000048]/30 hover:border-[#000048]/60 hover:bg-[#000048]/5 flex items-center gap-2 group">
              <span>Login</span>
              <svg
                className="w-4 h-4 text-[#000048] transition-transform duration-200 group-hover:translate-y-0.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="text-gray-700 hover:text-[#000048] focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 p-2 rounded-md transition-colors duration-200"
            >
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMenuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Navigation Menu - UPDATED STYLES */}
        {isMenuOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-0 sm:px-3 bg-white border-t"> {/* Changed space-y */}
              {navLinks.map((link) => {
                const isActive = pathname === link.href;
                return (
                  <a
                    key={link.name}
                    href={link.href}
                    className={`
                      block px-4 py-3 text-base font-medium transition-colors duration-200 rounded-md
                      ${isActive 
                        ? 'text-[#000048] bg-[#000048]/5 font-semibold' 
                        : 'text-gray-700 hover:text-[#000048] hover:bg-[#000048]/3'
                      }
                    `}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    {link.name}
                  </a>
                );
              })}
              <div className="pt-4 pb-3 border-t">
                <button className="w-full font-sans text-[#000048] px-4 py-3 rounded-md transition-all duration-200 border border-[#000048]/30 hover:border-[#000048]/60 hover:bg-[#000048]/5 flex items-center justify-center gap-2">
                  <span>Login</span>
                  <svg
                    className="w-4 h-4 text-[#000048]"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}