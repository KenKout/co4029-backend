module.exports = {
  apps: [
    {
      name: "abridgeai-backend",
      cwd: "/root/co4029/backend",
      script: "/root/co4029/backend/.venv/bin/uvicorn",
      args: "abridgeai.api:create_app --factory --host 0.0.0.0 --port 8000",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      kill_timeout: 5000,
      max_memory_restart: "1500M",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      out_file: "/root/.pm2/logs/abridgeai-backend-out.log",
      error_file: "/root/.pm2/logs/abridgeai-backend-err.log",
      merge_logs: true,
    },
    {
      name: "abridgeai-worker",
      cwd: "/root/co4029/backend",
      script: "/root/co4029/backend/.venv/bin/arq",
      args: "abridgeai.workers.arq_app.WorkerSettings",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      kill_timeout: 10000,
      max_memory_restart: "2000M",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      out_file: "/root/.pm2/logs/abridgeai-worker-out.log",
      error_file: "/root/.pm2/logs/abridgeai-worker-err.log",
      merge_logs: true,
    },
    {
      // LiveKit voice-interview agent worker (Phase 3). Connects out to
      // LiveKit Cloud and is dispatched per session by the join token's
      // room-config. Requires the `interview-agent` extra installed in the
      // venv (uv sync --extra interview-agent) and the LIVEKIT_* env set.
      // Only run this process when INTERVIEW_VOICE_ENABLED=true.
      name: "abridgeai-interview-agent",
      cwd: "/root/co4029/backend",
      script: "/root/co4029/backend/.venv/bin/python",
      args: "-m abridgeai.features.interviews.realtime.agent start",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      kill_timeout: 10000,
      max_memory_restart: "2000M",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      out_file: "/root/.pm2/logs/abridgeai-interview-agent-out.log",
      error_file: "/root/.pm2/logs/abridgeai-interview-agent-err.log",
      merge_logs: true,
    },
  ],
};
