import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import deque
import threading
import time
import queue

st.set_page_config(
    page_title="Flow Meter Reader",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("pivot_x",      642)
_init("pivot_y",      425)
_init("angle_min",    200.0)
_init("angle_max",    80.0)
_init("flow_min",     48.0)
_init("flow_max",     480.0)
_init("valid_min",    48.0)
_init("valid_max",    380.0)
_init("gauge_x1",     215)
_init("gauge_y1",     140)
_init("gauge_x2",     740)
_init("gauge_y2",     515)
_init("smooth_win",   7)
_init("running",      False)
_init("flow_log",     [])
_init("last_frame",   None)
_init("last_flow",    None)
_init("last_angle",   None)
_init("frame_q",      queue.Queue(maxsize=2))
_init("cap_ref",      [None])
_init("debug_mask",   False)
_init("show_calib",   False)
_init("calib_frame",  None)
_init("click_coords", None)
_init("arc_lo",       70.0)
_init("arc_hi",       230.0)

# ── NEW: shake-resistance state ──────────────────────────────
_init("angle_history",  deque(maxlen=12))   # temporal smoothing buffer
_init("prev_gray",      None)               # for frame differencing
_init("last_good_angle", None)              # last accepted angle
_init("lock_radius",    35.0)               # max jump allowed per frame (°)

S = st.session_state

# ─────────────────────────────────────────────────────────────
#  ANGLE → FLOW
# ─────────────────────────────────────────────────────────────
def angle_to_flow(angle_deg):
    a   = angle_deg % 360
    lo  = S.angle_min % 360
    hi  = S.angle_max % 360
    dist_a  = (lo - a)  % 360
    dist_hi = (lo - hi) % 360
    ratio = dist_a / dist_hi if dist_hi > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return round(S.flow_min + ratio * (S.flow_max - S.flow_min), 1)

# ─────────────────────────────────────────────────────────────
#  ANGULAR DISTANCE  (handles 0/360 wraparound)
# ─────────────────────────────────────────────────────────────
def angular_dist(a, b):
    """Shortest angular distance between two angles in degrees."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

# ─────────────────────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────────────────────
def _put(img, text, y, color=(0,255,0), small=False):
    cv2.putText(img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60 if small else 0.80,
                color, 2)

def draw_calib_overlay(frame):
    img = frame.copy()
    px, py = S.pivot_x, S.pivot_y
    cv2.rectangle(img,
                  (S.gauge_x1, S.gauge_y1),
                  (S.gauge_x2, S.gauge_y2),
                  (0, 200, 255), 2)
    cv2.circle(img, (px, py), 12, (0, 255, 255), 2)
    cv2.circle(img, (px, py),  4, (0, 255, 255), -1)
    cv2.putText(img, f"Pivot ({px},{py})", (px+15, py-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    for angle, label, color in [
        (S.angle_min, f"MIN {S.flow_min:.0f}", (0, 80, 255)),
        (S.angle_max, f"MAX {S.flow_max:.0f}", (0, 255, 80)),
    ]:
        rad = np.radians(angle)
        ex  = int(px + 180 * np.cos(rad))
        ey  = int(py - 180 * np.sin(rad))
        cv2.line(img, (px, py), (ex, ey), color, 2)
        cv2.putText(img, label, (ex+4, ey+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
    cv2.putText(img, "CALIBRATION VIEW — adjust sliders in sidebar",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
    return img

# ─────────────────────────────────────────────────────────────
#  MOTION BLUR DETECTION  — skip blurry frames
# ─────────────────────────────────────────────────────────────
def is_too_blurry(gray, threshold=60.0):
    """Returns True if the frame is motion-blurred (Laplacian variance)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

# ─────────────────────────────────────────────────────────────
#  SHAKE-STABILISED NEEDLE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_needle(frame, return_mask=False):
    frame  = cv2.resize(frame, (960, 540))
    output = frame.copy()
    px, py = S.pivot_x, S.pivot_y

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 1. SKIP BLURRY FRAMES (hand shake = motion blur) ────
    #    Return the last known good angle instead of a bad read
    if is_too_blurry(gray, threshold=55.0):
        _put(output, "Motion blur — skipped", 40, (0, 165, 255))
        if S.last_good_angle is not None:
            # Re-draw the last known needle position in orange
            rad = np.radians(S.last_good_angle)
            ex = int(px + 200 * np.cos(rad))
            ey = int(py - 200 * np.sin(rad))
            cv2.line(output, (px, py), (ex, ey), (0, 165, 255), 2)
            _put(output, f"Last angle: {S.last_good_angle:.1f} deg (held)", 75,
                 (0, 165, 255), small=True)
        return output, None, (gray if return_mask else None)

    # ── 2. CLAHE — boosts contrast even in bad lighting ─────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ── 3. DENOISE before thresholding ──────────────────────
    blur = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # ── 4. DUAL-THRESHOLD: combine adaptive + Otsu ──────────
    adapt = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=8
    )
    _, otsu = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_mask = cv2.bitwise_or(adapt, otsu)

    # ── 5. ROI mask ─────────────────────────────────────────
    gauge_mask = np.zeros_like(dark_mask)
    cv2.rectangle(gauge_mask,
                  (S.gauge_x1, S.gauge_y1),
                  (S.gauge_x2, S.gauge_y2), 255, -1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gauge_mask = cv2.erode(gauge_mask, erode_k, iterations=4)

    needle_region = cv2.bitwise_and(dark_mask, gauge_mask)
    needle_region[py:, px:] = 0  # remove counterweight quadrant

    # ── 6. MORPHOLOGY: close gaps caused by shake ───────────
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # wider kernel
    needle_region = cv2.morphologyEx(needle_region, cv2.MORPH_CLOSE,
                                     close_k, iterations=3)      # more iterations

    edges = cv2.Canny(needle_region, 20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=20,          # lower threshold = more tolerant
                             minLineLength=50,      # accept slightly shorter lines
                             maxLineGap=25)

    mask_out = needle_region if return_mask else None

    if lines is None:
        _put(output, "Needle NOT detected", 40, (0, 0, 255))
        # Show last good reading in orange so the display doesn't go blank
        if S.last_good_angle is not None:
            rad = np.radians(S.last_good_angle)
            ex = int(px + 200 * np.cos(rad))
            ey = int(py - 200 * np.sin(rad))
            cv2.line(output, (px, py), (ex, ey), (0, 165, 255), 2)
        return output, None, mask_out

    def endpoint_angle(ex, ey):
        dx = ex - px
        dy = py - ey
        return np.degrees(np.arctan2(dy, dx)) % 360

    arc_lo, arc_hi = S.arc_lo, S.arc_hi

    candidates = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        length = np.hypot(x2-x1, y2-y1)
        if length < 45:
            continue
        dx, dy = x2-x1, y2-y1
        perp = abs(dy*px - dx*py + x2*y1 - y2*x1) / (length + 1e-9)
        if perp > 55:
            continue

        d1 = np.hypot(x1-px, y1-py)
        d2 = np.hypot(x2-px, y2-py)
        tip_x, tip_y = (x1, y1) if d1 >= d2 else (x2, y2)

        tip_a  = endpoint_angle(tip_x, tip_y)
        in_arc = arc_lo <= tip_a <= arc_hi

        # ── TEMPORAL BIAS: prefer angles close to last known ──
        temporal_penalty = 0
        if S.last_good_angle is not None:
            dist = angular_dist(tip_a, S.last_good_angle)
            # Add a graduated penalty — small jump = low penalty
            temporal_penalty = min(dist * 8, 5000)

        score = (0 if in_arc else 10000) + perp + temporal_penalty
        candidates.append((score, perp, tip_x, tip_y, tip_a, length))

    if not candidates:
        _put(output, "No line near pivot", 40, (0, 0, 255))
        return output, None, mask_out

    candidates.sort(key=lambda c: c[0])
    _, best_perp, tip_x, tip_y, raw_angle, seg_len = candidates[0]

    # ── 7. JUMP FILTER: reject implausible sudden jumps ─────
    if S.last_good_angle is not None:
        jump = angular_dist(raw_angle, S.last_good_angle)
        if jump > S.lock_radius:
            # Accept only a partial step toward the new angle
            # (rubber-band effect — smoothly follows even big moves)
            direction = 1 if ((raw_angle - S.last_good_angle) % 360) < 180 else -1
            raw_angle = (S.last_good_angle + direction * min(jump, S.lock_radius)) % 360
            _put(output, f"Jump damped ({jump:.1f}°)", 175, (0, 165, 255), small=True)

    # ── 8. TEMPORAL SMOOTHING: weighted circular mean ────────
    S.angle_history.append(raw_angle)

    # Circular mean (handles 0/360 wraparound correctly)
    sins = np.mean([np.sin(np.radians(a)) for a in S.angle_history])
    coss = np.mean([np.cos(np.radians(a)) for a in S.angle_history])
    angle_deg = np.degrees(np.arctan2(sins, coss)) % 360

    S.last_good_angle = angle_deg

    flow = angle_to_flow(angle_deg)

    # ── Draw diagnostics ─────────────────────────────────────
    # Green = smoothed needle line
    cv2.line(output, (px, py), (tip_x, tip_y), (0, 255, 0), 3)
    cv2.circle(output, (px, py), 9, (0,   0, 255), -1)
    cv2.circle(output, (tip_x, tip_y), 6, (255, 0, 0), -1)
    cv2.rectangle(output,
                  (S.gauge_x1, S.gauge_y1),
                  (S.gauge_x2, S.gauge_y2),
                  (0, 200, 255), 1)

    _put(output, f"Angle  : {angle_deg:.1f} deg", 40,  (0, 200, 255))
    _put(output, f"Raw    : {flow:.1f} Nm3/h",    80,  (0, 255,   0))
    _put(output, f"Pivot  : ({px},{py})",          120, (200,200,  0), small=True)
    _put(output, f"PerpDist: {best_perp:.1f}px",  145, (200,200,  0), small=True)
    _put(output, f"Seg len : {seg_len:.0f}px",    165, (200,200,  0), small=True)

    valid = S.valid_min <= flow <= S.valid_max
    if not valid:
        _put(output, f"REJECTED  valid={S.valid_min:.0f}-{S.valid_max:.0f}",
             190, (0, 0, 255))
        return output, None, mask_out

    return output, flow, mask_out

# ─────────────────────────────────────────────────────────────
#  THREADED CAMERA READER
# ─────────────────────────────────────────────────────────────
def _camera_thread(url, frame_q, cap_ref, stop_event):
    cap = cv2.VideoCapture(url)
    cap_ref[0] = cap

    if not cap.isOpened():
        frame_q.put(("error", "Cannot connect to camera"))
        return

    frame_q.put(("ok", None))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(("frame", frame))

    cap.release()

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    camera_url = st.text_input(
        "Mobile Camera URL",
        "http://192.168.1.5:8080/video",
        help="IP Webcam app → Start server → copy URL shown on screen"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True,
                     disabled=S.running):
            S.frame_q        = queue.Queue(maxsize=2)
            S.cap_ref        = [None]
            S.angle_history  = deque(maxlen=12)
            S.last_good_angle = None
            stop_event       = threading.Event()
            S["_stop"]       = stop_event
            t = threading.Thread(
                target=_camera_thread,
                args=(camera_url, S.frame_q, S.cap_ref, stop_event),
                daemon=True
            )
            t.start()
            S.running  = True
            S.flow_log = []

    with col2:
        if st.button("⏹ Stop", use_container_width=True,
                     disabled=not S.running):
            if "_stop" in S:
                S["_stop"].set()
            S.running = False

    st.divider()

    st.subheader("📍 Pivot Position")
    st.caption("Set to the needle's rotation centre (hub)")
    S.pivot_x = st.slider("Pivot X", 0, 960, S.pivot_x)
    S.pivot_y = st.slider("Pivot Y", 0, 540, S.pivot_y)

    st.subheader("📐 Angle Calibration")
    st.caption("Use Calibration View to read angles at MIN/MAX flow")
    S.angle_min = st.slider("Angle at MIN flow (°)", 0.0, 360.0, S.angle_min, 1.0)
    S.angle_max = st.slider("Angle at MAX flow (°)", 0.0, 360.0, S.angle_max, 1.0)
    S.flow_min  = st.number_input("MIN flow (Nm³/h)", value=S.flow_min)
    S.flow_max  = st.number_input("MAX flow (Nm³/h)", value=S.flow_max)

    st.subheader("✅ Valid Range")
    S.valid_min = st.number_input("Valid MIN", value=S.valid_min)
    S.valid_max = st.number_input("Valid MAX", value=S.valid_max)

    st.subheader("🔲 Gauge ROI (960×540)")
    st.caption("Crop to just the gauge face")
    S.gauge_x1 = st.slider("ROI X1", 0, 960, S.gauge_x1)
    S.gauge_y1 = st.slider("ROI Y1", 0, 540, S.gauge_y1)
    S.gauge_x2 = st.slider("ROI X2", 0, 960, S.gauge_x2)
    S.gauge_y2 = st.slider("ROI Y2", 0, 540, S.gauge_y2)

    st.subheader("🎯 Arc Filter")
    st.caption("Needle tip must fall within this angular range")
    S.arc_lo = st.slider("Arc LO (°)", 0.0, 360.0, S.arc_lo, 1.0)
    S.arc_hi = st.slider("Arc HI (°)", 0.0, 360.0, S.arc_hi, 1.0)

    st.subheader("🔧 Shake Resistance")
    S.lock_radius = st.slider(
        "Max angle jump per frame (°)", 5.0, 90.0, S.lock_radius, 1.0,
        help="Lower = more stable but slower to follow real needle movement"
    )
    S.smooth_win = st.slider("Temporal smoothing window", 1, 20, S.smooth_win)
    S.debug_mask = st.checkbox("Show detection mask", S.debug_mask)
    S.show_calib = st.checkbox("Show calibration overlay", S.show_calib)

    if S.flow_log:
        df_dl = pd.DataFrame({"flow_nm3h": S.flow_log})
        st.download_button(
            "⬇ Download CSV",
            df_dl.to_csv(index=False),
            "flow_log.csv", "text/csv",
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────────────────────
st.title("🔵 Real-Time Flow Meter Reader")
st.caption("Connect your mobile phone camera → adjust pivot/ROI in sidebar → hit Start")

status_col, metric_col, angle_col = st.columns([3, 1, 1])
status_box  = status_col.empty()
metric_box  = metric_col.empty()
angle_box   = angle_col.empty()

vid_col, mask_col = st.columns([3, 1])
frame_placeholder = vid_col.empty()
mask_placeholder  = mask_col.empty()

chart_col, table_col = st.columns([2, 1])
chart_placeholder = chart_col.empty()
table_placeholder = table_col.empty()

# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
flow_buffer = deque(maxlen=S.smooth_win)

if S.running:
    status_box.info("🔗 Connecting to camera…")

    try:
        first = S.frame_q.get(timeout=8)
    except queue.Empty:
        first = ("error", "Timeout — no response from camera")

    if first[0] == "error":
        status_box.error(f"❌ {first[1]}")
        S.running = False
        st.stop()

    status_box.success("✅ Camera connected — reading frames")

    while S.running:
        try:
            msg_type, payload = S.frame_q.get(timeout=2)
        except queue.Empty:
            status_box.warning("⚠️ No frame received — check camera")
            continue

        if msg_type != "frame":
            break

        frame = payload

        if S.show_calib:
            calib_img = draw_calib_overlay(cv2.resize(frame, (960, 540)))
            frame_placeholder.image(
                cv2.cvtColor(calib_img, cv2.COLOR_BGR2RGB),
                channels="RGB", use_container_width=True
            )
            status_box.info("📐 Calibration view — detection paused")
            time.sleep(0.04)
            continue

        output, flow, mask = detect_needle(frame, return_mask=S.debug_mask)

        frame_placeholder.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True
        )

        if S.debug_mask and mask is not None:
            mask_placeholder.image(mask, clamp=True,
                                   caption="Detection mask",
                                   use_container_width=True)

        if flow is not None:
            flow_buffer = deque(flow_buffer, maxlen=S.smooth_win)
            flow_buffer.append(flow)
            smoothed = round(float(np.median(flow_buffer)), 1)
            S.flow_log.append(smoothed)
            S.last_flow  = smoothed
            S.last_angle = round(S.last_good_angle, 1) if S.last_good_angle else None

            metric_box.metric("Flow (Nm³/h)", f"{smoothed}")
            angle_box.metric("Angle (°)", f"{S.last_angle or '—'}")

            if len(S.flow_log) > 1:
                df = pd.DataFrame({"Flow (Nm³/h)": S.flow_log[-200:]})
                chart_placeholder.line_chart(df, height=220)

            if len(S.flow_log) >= 5:
                recent = S.flow_log[-10:][::-1]
                table_placeholder.dataframe(
                    pd.DataFrame({"Recent readings": recent}),
                    hide_index=True, height=200
                )
        else:
            # Show last known value in metric even when detection fails
            metric_box.metric(
                "Flow (Nm³/h)",
                f"{S.last_flow or '—'}",
                delta="held" if S.last_flow else None
            )

        time.sleep(0.03)

    status_box.info("⏹ Stopped.")

else:
    status_box.info("Press **▶ Start** in the sidebar to begin")
    if S.flow_log:
        st.subheader("Last session data")
        df = pd.DataFrame({"Flow (Nm³/h)": S.flow_log})
        st.line_chart(df, height=200)