/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      '@tensorflow/tfjs-node': false,
    };
    return config;
  },
};

module.exports = nextConfig;
