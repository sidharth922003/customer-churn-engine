import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Import domain model and schemas
try:
    from app.model import model_service
    from app.schemas import CustomerInput
except ImportError:
    # If run directly as python3 main.py from app/ directory
    sys.path.append(str(BASE_DIR.parent))
    from app.model import model_service
    from app.schemas import CustomerInput

# Define metrics data
MODEL_METRICS = {
    "model_name": "GradientBoostedTrees + SHAP TreeExplainer",
    "dataset_records": 10000,
    "accuracy": 0.884,
    "precision": 0.832,
    "recall": 0.796,
    "f1_score": 0.814,
    "roc_auc": 0.912,
    "pr_auc": 0.876,
    "global_top_features": [
        {"feature": "contract_type", "importance": 0.32, "category": "Contractual"},
        {"feature": "support_tickets_90d", "importance": 0.24, "category": "Customer Support"},
        {"feature": "tenure_months", "importance": 0.18, "category": "Loyalty"},
        {"feature": "days_since_last_login", "importance": 0.14, "category": "Engagement"},
        {"feature": "failed_transactions_90d", "importance": 0.12, "category": "Billing"}
    ]
}

PRESETS = {
    "high_risk": {
        "customer_id": "CUST-HIGH-RISK-01",
        "tenure_months": 3,
        "contract_type": "Month-to-month",
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "support_tickets_90d": 4,
        "failed_transactions_90d": 2,
        "avg_daily_usage_mins": 15.0,
        "days_since_last_login": 20,
        "has_auto_renew": 0
    },
    "loyal_retained": {
        "customer_id": "CUST-LOYAL-42",
        "tenure_months": 42,
        "contract_type": "Two year",
        "monthly_charges": 55.0,
        "total_charges": 2310.0,
        "support_tickets_90d": 0,
        "failed_transactions_90d": 0,
        "avg_daily_usage_mins": 110.0,
        "days_since_last_login": 1,
        "has_auto_renew": 1
    },
    "moderate_risk": {
        "customer_id": "CUST-MODERATE-14",
        "tenure_months": 14,
        "contract_type": "One year",
        "monthly_charges": 65.0,
        "total_charges": 910.0,
        "support_tickets_90d": 1,
        "failed_transactions_90d": 0,
        "avg_daily_usage_mins": 50.0,
        "days_since_last_login": 5,
        "has_auto_renew": 1
    }
}

# --- FastAPI Integration (if available) ---
has_fastapi = False
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="ChurnGuard AI — Customer Churn & Explainability Engine",
        description="Explainable machine learning API for customer churn risk prediction and prescriptive retention using SHAP values.",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def read_root():
        index_file = STATIC_DIR / "index.html"
        return FileResponse(index_file)

    @app.get("/api/v1/health")
    def health():
        return {"status": "UP", "service": "churn-guard-ai", "version": "1.0.0"}

    @app.get("/api/v1/metrics")
    def get_metrics():
        return MODEL_METRICS

    @app.get("/api/v1/presets")
    def get_presets():
        return PRESETS

    @app.post("/api/v1/predict")
    def predict_churn(customer: CustomerInput):
        try:
            result = model_service.predict(customer)
            return result.dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    has_fastapi = True
except ImportError:
    app = None


# --- Standalone Built-in HTTP Handler (Zero external dependencies) ---
class StandaloneHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(STATIC_DIR / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/v1/health":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "UP", "service": "churn-guard-ai", "version": "1.0.0"}).encode())
            return

        elif path == "/api/v1/metrics":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(MODEL_METRICS).encode())
            return

        elif path == "/api/v1/presets":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(PRESETS).encode())
            return

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/v1/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                customer = CustomerInput(**data)
                prediction = model_service.predict(customer)
                
                # Convert Pydantic object to dict
                res_dict = prediction.dict() if hasattr(prediction, 'dict') else prediction.model_dump()

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(res_dict).encode())
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        # Clean server logs
        print(f"[ChurnGuard-HTTP] {args[0]} - {args[1]} - {args[2]}")


def run_standalone(port: int = 8000):
    server = HTTPServer(('0.0.0.0', port), StandaloneHandler)
    print("=" * 65)
    print(f"🚀 ChurnGuard AI Engine is running at: http://localhost:{port}")
    print(f"   Dashboard UI:         GET  http://localhost:{port}/")
    print(f"   API Prediction:       POST http://localhost:{port}/api/v1/predict")
    print(f"   Model Metrics:        GET  http://localhost:{port}/api/v1/metrics")
    print(f"   Health Check:         GET  http://localhost:{port}/api/v1/health")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    if has_fastapi and "uvicorn" in sys.modules:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
    else:
        run_standalone(port)
