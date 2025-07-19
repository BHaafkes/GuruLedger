from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    """A simple route to confirm the service is running."""
    return "Hello, World! The test service is running correctly."

if __name__ == "__main__":
    # This block is for local testing, not used by Gunicorn/Cloud Run.
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
