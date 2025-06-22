import type { NextConfig } from "next";
import type { Configuration } from "webpack";

const nextConfig: NextConfig = {
  webpack: (config: Configuration, { isServer }) => {
    if (!isServer) {
      // Plotly.js optimization untuk client-side
      config.resolve = config.resolve || {};
      config.resolve.fallback = {
        ...(config.resolve.fallback || {}),
        fs: false,
        path: false,
        os: false,
      };
    }
    
    // Optimize plotly.js bundle
    const plotlyExternal = { 'plotly.js-dist': 'plotly.js-dist' };
    
    if (Array.isArray(config.externals)) {
      config.externals.push(plotlyExternal);
    } else if (config.externals) {
      config.externals = [config.externals, plotlyExternal];
    } else {
      config.externals = [plotlyExternal];
    }

    return config;
  },
  
  // Transpile plotly.js untuk compatibility
  transpilePackages: ['plotly.js-dist-min'],
  
  // Headers untuk CORS jika diperlukan
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, PUT, DELETE, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ];
  },
};

export default nextConfig;
