import threading
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer

def run_bot():
    from main import run_bot as start_polling
    asyncio.run(start_polling())

# Dummy web server for Render
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

def run_server():
    server = HTTPServer(("", 10000), HealthCheckHandler)
    server.serve_forever()

# Run both bot and server
if __name__ == "__main__":
    threading.Thread(target=run_bot).start()
    run_server()
