module.exports = {
  apps: [
    {
      name: "cx-voice-engine",
      script: "python3",
      args: "-m app.main",
      cwd: "./",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
      },
      autorestart: true,
      watch: false,
      max_memory_restart: "2G"
    },
    {
      name: "cx-enterprise",
      script: "src/index.js",
      cwd: "./enterprise/backend",
      env: {
        NODE_ENV: "production",
        PORT: 4500
      },
      autorestart: true,
      watch: false,
      max_memory_restart: "1G"
    }
  ]
};
