import mmap
import ctypes
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import threading

ETS2_PLUGIN_MMF_NAME = "Local\\SimTelemetryETS2"
ETS2_PLUGIN_MMF_SIZE = 16 * 1024

# Read float from memory
def get_float(raw, offset):
    return ctypes.c_float.from_buffer_copy(raw[offset:offset + 4]).value

def get_uint32(raw, offset):
    return ctypes.c_uint32.from_buffer_copy(raw[offset:offset + 4]).value

def get_int32(raw, offset):
    return ctypes.c_int32.from_buffer_copy(raw[offset:offset + 4]).value

def get_char_array(raw, offset, length):
    return bytes(raw[offset:offset+length]).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')

# Shared data dictionary
telemetry_data = {
    "speed": 0.0,
    "engine_rpm": 0.0,
    "fuel": 0.0,
    "truck_model": "",
    "cruise_speed": 0.0,
    "odometer": 0.0,
}

def read_telemetry():
    try:
        shared_mem = mmap.mmap(-1, ETS2_PLUGIN_MMF_SIZE, ETS2_PLUGIN_MMF_NAME, access=mmap.ACCESS_READ)
        while True:
            shared_mem.seek(0)
            raw = shared_mem.read(ETS2_PLUGIN_MMF_SIZE)

            telemetry_data["speed"] = get_float(raw, 24) * 3.6  # m/s to km/h
            telemetry_data["engine_rpm"] = get_float(raw, 80)
            telemetry_data["fuel"] = get_float(raw, 88)
            telemetry_data["truck_model"] = get_char_array(raw, 804, 64)
            telemetry_data["cruise_speed"] = get_float(raw, 672) * 3.6
            telemetry_data["odometer"] = get_float(raw, 668)
            time.sleep(0.001)
    except FileNotFoundError:
        print("MMF not found. Start ETS2 with SDK plugin enabled.")
    except Exception as e:
        print("Error reading MMF:", e)

# Start telemetry reading in background
threading.Thread(target=read_telemetry, daemon=True).start()

# Dash app
app = dash.Dash(__name__)
app.title = "ETS2 Real-Time Dashboard"

app.layout = html.Div([
    html.H2("ETS2 Real-Time Telemetry Dashboard"),
    html.Div(id='truck-info', style={'fontSize': '20px', 'marginBottom': '20px'}),
    dcc.Interval(id='update', interval=2000, n_intervals=0),
    html.Div([
        dcc.Graph(id='speed-gauge', style={'display': 'inline-block', 'width': '30%'}),
        dcc.Graph(id='rpm-gauge', style={'display': 'inline-block', 'width': '30%'}),
        dcc.Graph(id='fuel-gauge', style={'display': 'inline-block', 'width': '30%'}),
    ]),
    dcc.Graph(id='odometer-line', style={'width': '100%'})
])

# Odometer history (last 20 points)
odometer_history = []

@app.callback(
    [Output('truck-info', 'children'),
     Output('speed-gauge', 'figure'),
     Output('rpm-gauge', 'figure'),
     Output('fuel-gauge', 'figure'),
     Output('odometer-line', 'figure')],
    [Input('update', 'n_intervals')]
)
def update_dashboard(n):
    speed = telemetry_data["speed"]
    rpm = telemetry_data["engine_rpm"]
    fuel = telemetry_data["fuel"]
    truck_model = telemetry_data["truck_model"]
    cruise = telemetry_data["cruise_speed"]
    odometer = telemetry_data["odometer"]

    odometer_history.append(odometer)
    if len(odometer_history) > 20:
        odometer_history.pop(0)

    return (
        f"Truck Model: {truck_model} | Cruise Speed: {cruise:.1f} km/h",
        go.Figure(go.Indicator(mode="gauge+number", value=speed,
                               title={'text': "Speed (km/h)"},
                               gauge={'axis': {'range': [0, 160]}})),
        go.Figure(go.Indicator(mode="gauge+number", value=rpm,
                               title={'text': "Engine RPM"},
                               gauge={'axis': {'range': [0, 3000]}})),
        go.Figure(go.Indicator(mode="gauge+number", value=fuel,
                               title={'text': "Fuel (liters)"},
                               gauge={'axis': {'range': [0, 1000]}})),
        go.Figure(data=[go.Scatter(y=odometer_history, mode='lines+markers')])
        .update_layout(title="Truck Odometer (km)", xaxis_title="Time", yaxis_title="Odometer")
    )

if __name__ == '__main__':
    app.run(debug=True)
