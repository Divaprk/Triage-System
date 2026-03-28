"""
laptop_receiver.py  —  run this on your Windows laptop

Live dashboard with WebSocket push — updates instantly when Pi sends data.

Install: pip install flask flask-socketio
Run:     python laptop_receiver.py
"""

from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO
from datetime import datetime

app = Flask(__name__)
app.config["SECRET_KEY"] = "triage"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

latest = {
    "triage_label":   "Waiting...",
    "triage_level":   "-",
    "confidence_pct": 0,
    "timestamp":      None,
    "heart_rate_bpm": None,
    "spo2_percent":   None,
    "temperature_c":  None,
    "ear":            None,
    "chest_pain":     0,
    "breathless":     0,
}

LEVEL_COLOUR_ANSI = {
    "CRITICAL": "\033[91m",
    "URGENT":   "\033[93m",
    "STABLE":   "\033[92m",
}
RESET = "\033[0m"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Pi Triage Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d0d0d; color: #e0e0e0; padding: 40px; }
    h1 { color: #aaa; font-size: 18px; margin-bottom: 30px; }
    .level { font-size: 64px; font-weight: bold; margin: 10px 0; transition: color 0.2s; }
    .CRITICAL { color: #ff4444; }
    .URGENT   { color: #ffaa00; }
    .STABLE   { color: #44ff88; }
    .Waiting  { color: #888; }
    .conf { font-size: 20px; color: #888; margin-bottom: 30px; }
    .grid { display: grid; grid-template-columns: repeat(3, 200px); gap: 16px; margin-top: 20px; }
    .card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px; transition: all 0.2s; }
    .card .label { font-size: 11px; color: #666; text-transform: uppercase; margin-bottom: 6px; }
    .card .value { font-size: 28px; color: #e0e0e0; }
    .card .unit  { font-size: 12px; color: #666; }
    .yes { color: #ff4444; font-weight: bold; }
    .no  { color: #44ff88; }
    .time { color: #555; font-size: 13px; margin-top: 30px; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #44ff88; margin-right: 8px; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .flash { animation: flash 0.3s; }
    @keyframes flash { 0%{background:#2a2a2a} 100%{background:#1a1a1a} }
  </style>
</head>
<body>
  <h1><span class="dot"></span>Pi 5 Triage System — Live Dashboard</h1>

  <div class="level Waiting" id="level">Waiting...</div>
  <div class="conf" id="conf">—</div>

  <div class="grid">
    <div class="card" id="card-hr">
      <div class="label">Heart Rate</div>
      <div class="value" id="hr">—</div>
      <div class="unit">BPM</div>
    </div>
    <div class="card" id="card-spo2">
      <div class="label">SpO2</div>
      <div class="value" id="spo2">—</div>
      <div class="unit">%</div>
    </div>
    <div class="card" id="card-temp">
      <div class="label">Temperature</div>
      <div class="value" id="temp">—</div>
      <div class="unit">°C</div>
    </div>
    <div class="card" id="card-ear">
      <div class="label">EAR (Alertness)</div>
      <div class="value" id="ear">—</div>
      <div class="unit">ratio</div>
    </div>
    <div class="card" id="card-cp">
      <div class="label">Chest Pain</div>
      <div class="value no" id="cp">No</div>
    </div>
    <div class="card" id="card-br">
      <div class="label">Breathlessness</div>
      <div class="value no" id="br">No</div>
    </div>
  </div>

  <div class="time" id="time">Last update: —</div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
  <script>
    const socket = io();

    function flash(id) {
      const el = document.getElementById(id);
      el.classList.remove('flash');
      void el.offsetWidth;
      el.classList.add('flash');
    }

    socket.on('update', function(d) {
      // Triage level
      const levelEl = document.getElementById('level');
      levelEl.textContent = d.triage_label;
      levelEl.className = 'level ' + (d.triage_label || 'Waiting').split(' ')[0];

      document.getElementById('conf').textContent = d.confidence_pct.toFixed(1) + '% confidence';

      // Vitals
      document.getElementById('hr').textContent   = d.heart_rate_bpm  != null ? d.heart_rate_bpm  : '—';
      document.getElementById('spo2').textContent = d.spo2_percent     != null ? d.spo2_percent    : '—';
      document.getElementById('temp').textContent = d.temperature_c    != null ? d.temperature_c.toFixed(1) : '—';
      document.getElementById('ear').textContent  = d.ear              != null ? d.ear.toFixed(2)  : '—';

      // Symptoms
      const cpEl = document.getElementById('cp');
      cpEl.textContent  = d.chest_pain ? 'YES' : 'No';
      cpEl.className    = 'value ' + (d.chest_pain ? 'yes' : 'no');

      const brEl = document.getElementById('br');
      brEl.textContent  = d.breathless ? 'YES' : 'No';
      brEl.className    = 'value ' + (d.breathless ? 'yes' : 'no');

      // Timestamp
      if (d.timestamp) {
        const t = new Date(d.timestamp * 1000);
        document.getElementById('time').textContent =
          'Last update: ' + t.toLocaleTimeString();
      }

      // Flash updated cards
      ['card-hr','card-spo2','card-temp','card-ear','card-cp','card-br'].forEach(flash);
    });
  </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route("/update_vitals", methods=["POST"])
def update_vitals():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no JSON body"}), 400

    latest.update(data)

    # Push to all connected dashboard browsers instantly
    socketio.emit("update", latest)

    label    = data.get("triage_label", "UNKNOWN")
    level    = data.get("triage_level", "?")
    conf     = data.get("confidence_pct", 0)
    ts       = data.get("timestamp", 0)
    hr       = data.get("heart_rate_bpm")
    spo2     = data.get("spo2_percent")
    temp     = data.get("temperature_c")
    ear      = data.get("ear")
    cp       = data.get("chest_pain", 0)
    br       = data.get("breathless", 0)
    colour   = LEVEL_COLOUR_ANSI.get(label, RESET)
    time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"

    print()
    print("=" * 48)
    print(f"  {colour}TRIAGE LEVEL {level} - {label}{RESET}   ({conf:.1f}%)")
    print(f"  Time        : {time_str}")
    print(f"  Heart rate  : {hr} BPM")
    print(f"  SpO2        : {spo2}%")
    print(f"  Temperature : {temp} C")
    print(f"  EAR         : {f'{ear:.3f}' if ear is not None else 'N/A'}")
    print(f"  Chest pain  : {'YES' if cp else 'no'}")
    print(f"  Breathless  : {'YES' if br else 'no'}")
    print("=" * 48)

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    print("=" * 48)
    print("  Pi 5 Triage Receiver - listening on :5001")
    print("  Dashboard: http://127.0.0.1:5000")
    print("  WebSocket live push enabled")
    print("=" * 48)
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)