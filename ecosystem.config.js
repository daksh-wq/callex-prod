module.exports = {
  apps: [
    {
      name: "cx-voice-engine",
      script: "venv/bin/python3",
      args: "-m app.main",
      cwd: "./",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
        PYTHONPATH: ".",
      },
      error_file: "logs/voice-error.log",
      out_file: "logs/voice-out.log",
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
        PORT: 4000
      },
      error_file: "logs/enterprise-error.log",
      out_file: "logs/enterprise-out.log",
      autorestart: true,
      watch: false,
      max_memory_restart: "1G"
    }
  ]
};
