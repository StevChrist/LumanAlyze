'use client'

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import './globals.css';

// Navbar Component
const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navItems = [
    { href: '/', label: 'Home' },
    { href: '/about', label: 'About' },
    { href: '/how-to-use', label: 'How To Use' }
  ];

  const isActive = (href: string) => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname?.startsWith(href);
  };

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 1000,
      backgroundColor: '#014F39',
      borderBottom: '1px solid rgba(251, 247, 199, 0.2)',
      width: '100%'
    }}>
      <nav style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px 32px',
        maxWidth: '100%',
        margin: '0 auto',
        height: '60px',
        boxSizing: 'border-box'
      }}>
        {/* Logo - Left Side */}
        <Link href="/" style={{
          fontSize: '28px',
          fontWeight: '700',
          fontFamily: 'Montserrat, sans-serif',
          color: '#FBF7C7',
          textDecoration: 'none',
          flexShrink: 0
        }}>
          ALYZE.
        </Link>

        {/* Desktop Navigation - Right Side */}
        <div style={{
          display: 'flex',
          gap: '32px',
          alignItems: 'center',
          height: '100%'
        }} className="desktop-nav">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              style={{
                fontFamily: 'Lora, serif',
                fontSize: '16px',
                color: isActive(item.href) ? '#FBF7C7' : 'rgba(251, 247, 199, 0.8)',
                textDecoration: 'none',
                fontWeight: isActive(item.href) ? '600' : '400',
                padding: '8px 16px',
                borderRadius: '6px',
                backgroundColor: isActive(item.href) ? 'rgba(251, 247, 199, 0.1)' : 'transparent',
                transition: 'all 0.3s ease',
                whiteSpace: 'nowrap',
                display: 'flex',
                alignItems: 'center',
                height: '40px'
              }}
              onMouseEnter={(e) => {
                if (!isActive(item.href)) {
                  e.currentTarget.style.backgroundColor = 'rgba(251, 247, 199, 0.05)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive(item.href)) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }
              }}
            >
              {item.label}
            </Link>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          style={{
            display: 'none',
            flexDirection: 'column',
            gap: '4px',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '8px',
            justifyContent: 'center',
            alignItems: 'center',
            width: '40px',
            height: '40px'
          }}
          className="mobile-menu-btn"
        >
          <span style={{
            width: '24px',
            height: '2px',
            backgroundColor: '#FBF7C7',
            transition: 'all 0.3s ease',
            transform: isMobileMenuOpen ? 'rotate(45deg) translate(5px, 5px)' : 'none'
          }}></span>
          <span style={{
            width: '24px',
            height: '2px',
            backgroundColor: '#FBF7C7',
            transition: 'all 0.3s ease',
            opacity: isMobileMenuOpen ? 0 : 1
          }}></span>
          <span style={{
            width: '24px',
            height: '2px',
            backgroundColor: '#FBF7C7',
            transition: 'all 0.3s ease',
            transform: isMobileMenuOpen ? 'rotate(-45deg) translate(7px, -6px)' : 'none'
          }}></span>
        </button>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            backgroundColor: '#014F39',
            border: '1px solid rgba(251, 247, 199, 0.2)',
            borderTop: 'none',
            padding: '16px 32px',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            zIndex: 999
          }} className="mobile-nav">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                style={{
                  fontFamily: 'Lora, serif',
                  fontSize: '16px',
                  color: isActive(item.href) ? '#FBF7C7' : 'rgba(251, 247, 199, 0.8)',
                  textDecoration: 'none',
                  fontWeight: isActive(item.href) ? '600' : '400',
                  padding: '12px 16px',
                  borderRadius: '6px',
                  backgroundColor: isActive(item.href) ? 'rgba(251, 247, 199, 0.1)' : 'transparent',
                  transition: 'all 0.3s ease'
                }}
              >
                {item.label}
              </Link>
            ))}
          </div>
        )}
      </nav>

      <style jsx>{`
        @media (max-width: 768px) {
          .desktop-nav {
            display: none !important;
          }
          .mobile-menu-btn {
            display: flex !important;
          }
        }
        @media (min-width: 769px) {
          .mobile-nav {
            display: none !important;
          }
        }
      `}</style>
    </header>
  );
};

// Footer Component
const Footer: React.FC = () => {
  return (
    <footer style={{
      backgroundColor: '#014F39',
      color: '#FBF7C7',
      padding: '24px 32px',
      textAlign: 'center',
      borderTop: '1px solid rgba(251, 247, 199, 0.2)',
      marginTop: 'auto',
      width: '100%'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '20px',
        flexWrap: 'wrap',
        fontFamily: 'Lora, serif',
        fontSize: '14px'
      }}>
        <span style={{
          color: '#FBF7C7',
          opacity: 0.9
        }}>
          Copyright © 2025 by Steven | All Rights Reserved
        </span>
        
        <div style={{
          display: 'flex',
          gap: '16px',
          alignItems: 'center'
        }}>
          {/* Instagram Icon */}
          <a 
            href="https://www.instagram.com/_stev.chris/" 
            target="_blank" 
            rel="noopener noreferrer" 
            style={{ 
              color: '#FBF7C7',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '8px',
              borderRadius: '50%',
              backgroundColor: 'transparent'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(251, 247, 199, 0.1)';
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.transform = 'scale(1)';
            }}
            aria-label="Instagram"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              fill="currentColor" 
              viewBox="0 0 24 24" 
              width="22" 
              height="22"
            >
              <path d="M7.75 2h8.5A5.75 5.75 0 0122 7.75v8.5A5.75 5.75 0 0116.25 22h-8.5A5.75 5.75 0 012 16.25v-8.5A5.75 5.75 0 017.75 2zm0 1.5A4.25 4.25 0 003.5 7.75v8.5A4.25 4.25 0 007.75 20.5h8.5a4.25 4.25 0 004.25-4.25v-8.5A4.25 4.25 0 0016.25 3.5h-8.5zM12 7a5 5 0 110 10 5 5 0 010-10zm0 1.5a3.5 3.5 0 100 7 3.5 3.5 0 000-7zm4.75-.75a1.25 1.25 0 110 2.5 1.25 1.25 0 010-2.5z"/>
            </svg>
          </a>

          {/* Email Icon */}
          <a 
            href="mailto:stevenchristiano333@gmail.com" 
            style={{ 
              color: '#FBF7C7',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '8px',
              borderRadius: '50%',
              backgroundColor: 'transparent'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(251, 247, 199, 0.1)';
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.transform = 'scale(1)';
            }}
            aria-label="Email"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              fill="currentColor" 
              viewBox="0 0 24 24" 
              width="22" 
              height="22"
            >
              <path d="M20 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2zm0 2v.01L12 13 4 6.01V6h16zM4 18V8l8 5 8-5v10H4z"/>
            </svg>
          </a>
        </div>
      </div>

      {/* Mobile responsive copyright */}
      <style jsx>{`
        @media (max-width: 768px) {
          footer > div {
            flex-direction: column !important;
            gap: 12px !important;
          }
          footer span {
            font-size: 12px !important;
          }
        }
      `}</style>
    </footer>
  );
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{
        margin: 0,
        padding: 0,
        fontFamily: 'Lora, serif',
        backgroundColor: '#014F39',
        color: '#FBF7C7',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <Navbar />
        <main style={{ flex: 1 }}>
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
