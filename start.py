import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import asyncio
from main import run_bot  # Make sure run_bot is an async def

# Dummy web server for Render health checks
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_healthcheck_server():
    server = HTTPServer(("", 10000), HealthCheckHandler)
    server.serve_forever()

if __name__ == "__main__":
    # Run HTTP healthcheck server in a thread
    threading.Thread(target=run_healthcheck_server, daemon=True).start()

    # Run the bot in the main thread
    print("Starting bot...")
    asyncio.run(run_bot())
