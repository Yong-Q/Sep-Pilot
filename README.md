# BiMemAgent SDK

Sanitized release snapshot: **v3.4.63**.

Multi-agent scientific workflow backend and React frontend.

This is a sanitized source distribution. Production users, conversations, jobs, results, credentials, server configuration, scientific binaries and local datasets are intentionally excluded.

Install the backend with `pip install -e .`; install frontend dependencies with `cd frontend && npm ci`. Configure your model credentials through environment variables and configure your own scheduler, scientific software and datasets before running calculations. `/home/user` paths are placeholders.

Start the API with `python -m uvicorn api:app --host 127.0.0.1 --port 8000`.
