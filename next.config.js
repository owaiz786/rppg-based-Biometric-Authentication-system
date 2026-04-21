/** @type {import('next').NextConfig} */
const nextConfig = {
  allowedDevOrigins: [
    'mud-xml-parker-tigers.trycloudflare.com',  // Your current tunnel
    'romance-dist-pursue-eclipse.trycloudflare.com', // Your previous tunnel
    '*.trycloudflare.com',  // Allow ALL trycloudflare subdomains
    'localhost',
    '127.0.0.1'
  ],
}

module.exports = nextConfig