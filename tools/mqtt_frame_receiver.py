#!/usr/bin/env python3
"""
ESP32-CAM MQTT event receiver with HTTP frame pull or video recording,
with optional YOLO-based detection filtering.

Listens for motion events over MQTT, then either:
  - (default) fetches a single JPEG via GET /control?still=1
  - (--video)  records a video from the MJPEG stream at /sustain?stream=1,
               stopping --video-timeout seconds after the last motion event.

YOLO filtering (--yolo):
  Images: saved only if a target class is detected; boxes are drawn.
          Discarded frames are reported in stdout and optionally written
          to {outdir}/discarded/ with --save-discarded.
  Videos: decoded into per-frame JPEGs in {video}_frames/; only frames
          with detections are kept. Discarded frames go to
          {video}_frames/discarded/ when --save-discarded is set.
          The original .avi is always kept.
  Inference runs in a separate process — MQTT is never blocked.
  Default model: yolo11n.pt (auto-downloaded by ultralytics on first run).
  Default target classes: 0=person, 15=cat.

Motion log:
  motion_log_YYYYMMDD.json is written in outdir (JSON Lines format).
  Contains on/off motion events, detections, discards, and fetch errors.

Also supports LDR readings:
  Subscribes to {prefix}sensor/{hostname}/ldr; use --ldr-trigger to
  request a reading on connect (requires --hostname).

Usage:
    pip install paho-mqtt requests ultralytics opencv-python

    python3 mqtt_frame_receiver.py --broker 192.168.1.x --camera 192.168.1.y
    python3 mqtt_frame_receiver.py ... --yolo
    python3 mqtt_frame_receiver.py ... --video --yolo [--save-discarded]
    python3 mqtt_frame_receiver.py ... --hostname esp32cam --ldr-trigger
"""

import argparse
import fcntl
import json
import multiprocessing
import os
import sys
import time
import traceback
import threading
import requests
import paho.mqtt.client as mqtt

# ---------------------------------------------------------------------------
DEFAULT_BROKER        = "localhost"
DEFAULT_PORT          = 1883
DEFAULT_TOPIC_PREFIX  = "homeassistant/"
DEFAULT_OUTPUT_DIR    = "./motion_frames"
DEFAULT_VIDEO_TIMEOUT = 300   # seconds
HTTP_TIMEOUT          = 5     # seconds for still-image requests
YOLO_QUEUE_MAXSIZE    = 100   # drop items rather than grow unbounded
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
# LDR food-level estimation parameters (adjust to match your cat's intake)
_FLAT_ZONE_MAX_DIFF   = 1160   # differential above this → reliable low-food zone (< ~600 g)
_FLAT_ZONE_MIN_DIFF   = 950    # differential below this → reliable high-food zone (> ~1100 g)
_REFILL_DROP          = 800    # minimum drop in differential to detect a refill event
_CONSUMPTION_GPD      = 35.0   # grams per day nominal consumption
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="ESP32-CAM MQTT event receiver + HTTP frame/video puller")
    p.add_argument("--broker",         default=DEFAULT_BROKER,
                   help="MQTT broker IP/hostname")
    p.add_argument("--port",           default=DEFAULT_PORT, type=int,
                   help="MQTT broker port (default 1883)")
    p.add_argument("--prefix",         default=DEFAULT_TOPIC_PREFIX,
                   help="mqtt_topic_prefix set in ESP32 config")
    p.add_argument("--hostname",       default="",
                   help="ESP32 hostname (omit to accept any device)")
    p.add_argument("--camera",         required=True,
                   help="ESP32 camera IP for HTTP requests")
    p.add_argument("--outdir",         default=DEFAULT_OUTPUT_DIR,
                   help="Directory to save files")
    p.add_argument("--user",           default="",
                   help="MQTT broker username (if required)")
    p.add_argument("--password",       default="",
                   help="MQTT broker password (if required)")
    p.add_argument("--video",          action="store_true",
                   help="Record video instead of single frame")
    p.add_argument("--video-timeout",  default=DEFAULT_VIDEO_TIMEOUT, type=int,
                   metavar="SECONDS",
                   help=f"Stop recording N seconds after last motion event "
                        f"(default {DEFAULT_VIDEO_TIMEOUT})")
    p.add_argument("--ldr-trigger",    action="store_true",
                   help="Send LDR measurement command on connect (requires --hostname)")
    # YOLO options
    p.add_argument("--yolo",           action="store_true",
                   help="Enable YOLO-based filtering (requires ultralytics)")
    p.add_argument("--yolo-model",     default="yolo11n.pt", metavar="MODEL",
                   help="YOLO model file (default: yolo11n.pt, auto-downloaded)")
    p.add_argument("--yolo-conf",      default=0.25, type=float, metavar="CONF",
                   help="Minimum detection confidence (default: 0.25)")
    p.add_argument("--yolo-classes",   default=[0, 15], type=int, nargs="+",
                   metavar="ID",
                   help="COCO class IDs to keep (default: 0=person 15=cat)")
    p.add_argument("--save-discarded", action="store_true",
                   help="Write frames with no detections to {outdir}/discarded/ "
                        "(or {video}_frames/discarded/ for videos). "
                        "Warning: can be many frames for long videos.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Motion log — safe for concurrent writes from main + worker processes
# ---------------------------------------------------------------------------

def _log_event(out_dir, entry):
    """Append one JSON line to today's motion log in out_dir."""
    date_str = time.strftime("%Y%m%d")
    log_path = os.path.join(out_dir, f"motion_log_{date_str}.json")
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    line = json.dumps(entry) + "\n"
    try:
        with open(log_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"[!] motion log write failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Feeder log — LDR readings with optional food-level calibration
# ---------------------------------------------------------------------------

def _load_calibration():
    """Load calibration.txt from the script directory.

    Supports two sections separated by a [high] marker:
      - Default / [low] section: 0-600 g range (high differential → low food)
      - [high] section:          600-1400 g range (flat zone + upper range)

    Each line: differential  food_grams  (whitespace separated; # = comment)

    Returns (cal_low, cal_high), each a list of (diff, food) sorted by diff
    ascending, or None if that section is absent/empty.
    """
    cal_path = os.path.join(SCRIPT_DIR, "calibration.txt")
    if not os.path.exists(cal_path):
        return None, None
    low_pts, high_pts = [], []
    section = low_pts
    try:
        with open(cal_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower() == "[high]":
                    section = high_pts
                    continue
                if line.lower() == "[low]":
                    section = low_pts
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        section.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
        low_pts.sort(key=lambda p: p[0])
        high_pts.sort(key=lambda p: p[0])
    except Exception as e:
        print(f"[!] calibration.txt read error: {e}", file=sys.stderr)
    return (low_pts or None, high_pts or None)


def _interpolate_food_level(differential, calibration):
    """Return food level string interpolated from calibration, or 'unknown'."""
    if not calibration:
        return "unknown"
    if len(calibration) == 1:
        return f"{calibration[0][1]:.1f}"
    if differential <= calibration[0][0]:
        return f"{calibration[0][1]:.1f}"
    if differential >= calibration[-1][0]:
        return f"{calibration[-1][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(calibration, calibration[1:]):
        if x0 <= differential <= x1:
            t = (differential - x0) / (x1 - x0)
            return f"{y0 + t * (y1 - y0):.1f}"
    return "unknown"


def _find_feeder_log(log_dir):
    """Return path to feeder_log.txt, preferring log_dir then SCRIPT_DIR."""
    for d in [log_dir, SCRIPT_DIR]:
        if d:
            p = os.path.join(d, "feeder_log.txt")
            if os.path.exists(p):
                return p
    return None


def _find_last_refill(log_path, cal_high):
    """Scan feeder_log.txt for the most recent refill event.

    A refill is detected when differential drops by > _REFILL_DROP and lands
    below _FLAT_ZONE_MIN_DIFF (food clearly in the upper range after refill).
    The first log entry with low differential also counts as an initial state.

    Returns (unix_timestamp, food_grams_at_refill) or None.
    """
    if not log_path:
        return None
    entries = []
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("timestamp") or line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                try:
                    ts   = time.mktime(time.strptime(parts[0], "%Y-%m-%d %H:%M:%S"))
                    diff = float(parts[3])
                    entries.append((ts, diff))
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        print(f"[!] feeder_log refill scan error: {e}", file=sys.stderr)
        return None

    if not entries:
        return None

    def food_at(d):
        if cal_high:
            try:
                return float(_interpolate_food_level(d, cal_high))
            except (ValueError, TypeError):
                pass
        return 1200.0  # fallback if no high calibration

    last_refill = None

    # First entry with low differential → feeder was full at log start
    if entries[0][1] < _FLAT_ZONE_MIN_DIFF:
        last_refill = (entries[0][0], food_at(entries[0][1]))

    for i in range(1, len(entries)):
        prev_diff        = entries[i - 1][1]
        curr_ts, curr_diff = entries[i]
        if prev_diff - curr_diff > _REFILL_DROP and curr_diff < _FLAT_ZONE_MIN_DIFF:
            last_refill = (curr_ts, food_at(curr_diff))

    return last_refill


def _estimate_food_level(differential, cal_low, cal_high, log_dir):
    """Estimate food level from differential + calibration + log history.

    Returns (food_level_str, source_str).

    source values:
      'calibrated'  — direct calibration lookup, reliable
      'log_estimated' — time-based estimate from last refill event in log
      'approx'      — flat-zone calibration fallback, low reliability
      'unknown'     — no data available
    """
    # Unambiguous low-food zone
    if differential > _FLAT_ZONE_MAX_DIFF:
        if cal_low:
            return _interpolate_food_level(differential, cal_low), "calibrated"
        return "unknown", "unknown"

    # Unambiguous high-food zone
    if differential < _FLAT_ZONE_MIN_DIFF:
        if cal_high:
            return _interpolate_food_level(differential, cal_high), "calibrated"
        return "unknown", "unknown"

    # Flat / ambiguous zone — prefer log-based time estimate
    log_path = _find_feeder_log(log_dir)
    refill   = _find_last_refill(log_path, cal_high) if log_path else None

    if refill:
        refill_ts, refill_food = refill
        elapsed_h = (time.time() - refill_ts) / 3600
        est = refill_food - elapsed_h * (_CONSUMPTION_GPD / 24)
        return f"{max(0, min(1400, est)):.0f}", "log_estimated"

    # No log history: rough calibration fallback
    for cal in [cal_high, cal_low]:
        if cal:
            return _interpolate_food_level(differential, cal), "approx"
    return "unknown", "unknown"


_FEEDER_LOG_HEADER = "timestamp\tambient\tilluminated\tdifferential\tpercent\tfood_level\tsource\n"


def _log_feeder(ldr_data, out_dir):
    """Append one LDR reading to feeder_log.txt.

    Location: next to the script while no calibration exists (calibration
    phase), then in out_dir once calibration.txt is present.
    Calibration and log history are re-read on every call.
    """
    cal_low, cal_high = _load_calibration()
    log_dir  = out_dir if (cal_low is not None or cal_high is not None) else SCRIPT_DIR
    log_path = os.path.join(log_dir, "feeder_log.txt")

    differential        = ldr_data.get("differential", 0)
    food_level, source  = _estimate_food_level(differential, cal_low, cal_high, log_dir)

    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = (f"{ts}\t"
            f"{ldr_data.get('ambient', '')}\t"
            f"{ldr_data.get('illuminated', '')}\t"
            f"{differential}\t"
            f"{ldr_data.get('percent', '')}\t"
            f"{food_level}\t"
            f"{source}\n")
    try:
        write_header = not os.path.exists(log_path)
        with open(log_path, "a") as f:
            if write_header:
                f.write(_FEEDER_LOG_HEADER)
            f.write(line)
    except Exception as e:
        print(f"[!] feeder_log.txt write error: {e}", file=sys.stderr)
    return food_level


# ---------------------------------------------------------------------------
# YOLO worker — runs in a separate process
# ---------------------------------------------------------------------------

def yolo_worker(queue, model_path, conf_threshold, target_classes, save_discarded):
    """Pull work items from queue and run YOLO inference.

    Item formats:
      ("image", jpeg_bytes, out_dir, ts, fetch_elapsed_s)
      ("video", video_path, out_dir)
      None  → shutdown sentinel
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[yolo] ultralytics not installed. Run: pip install ultralytics",
              file=sys.stderr)
        return

    print(f"[yolo] Loading {model_path} ...", flush=True)
    model = YOLO(model_path)
    names = model.names
    print(f"[yolo] Ready. Target classes: "
          f"{[names.get(c, str(c)) for c in target_classes]}  conf≥{conf_threshold}",
          flush=True)

    while True:
        item = queue.get()
        if item is None:
            print("[yolo] Shutting down.", flush=True)
            break
        kind = item[0]
        try:
            if kind == "image":
                _, jpeg_bytes, out_dir, ts, fetch_elapsed = item
                _yolo_image(model, jpeg_bytes, out_dir, ts,
                            fetch_elapsed, conf_threshold, target_classes,
                            save_discarded)
            elif kind == "video":
                _, video_path, out_dir = item
                _yolo_video(model, video_path, out_dir,
                            conf_threshold, target_classes, save_discarded)
        except Exception:
            print(f"[yolo] Error processing {kind}:", file=sys.stderr)
            traceback.print_exc()


def _draw_boxes(frame, hits, names):
    import cv2
    for b in hits:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        label = f"{names[int(b.cls)]} {float(b.conf):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def _yolo_image(model, jpeg_bytes, out_dir, ts,
                fetch_elapsed, conf, target_classes, save_discarded):
    """Run inference on one JPEG; save annotated file only if target detected."""
    import cv2
    import numpy as np

    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("[yolo] Could not decode image", file=sys.stderr)
        return

    t_infer = time.time()
    results  = model(frame, conf=conf, verbose=False)[0]
    detect_elapsed = time.time() - t_infer

    hits = [b for b in results.boxes if int(b.cls) in target_classes]

    fetch_ms  = int(fetch_elapsed * 1000)
    detect_ms = int(detect_elapsed * 1000)

    if hits:
        best_conf    = max(float(b.conf) for b in hits)
        hit_classes  = [model.names[int(b.cls)] for b in hits]
        hit_confs    = [round(float(b.conf), 3) for b in hits]
        _draw_boxes(frame, hits, model.names)
        filename = os.path.join(out_dir, f"cat_{ts}_{best_conf:.2f}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[yolo] DETECTED  {os.path.basename(filename)}  "
              f"classes={hit_classes}  fetch={fetch_ms}ms  detect={detect_ms}ms",
              flush=True)
        _log_event(out_dir, {
            "event":      "detected",
            "file":       os.path.basename(filename),
            "classes":    hit_classes,
            "confs":      hit_confs,
            "fetch_ms":   fetch_ms,
            "detect_ms":  detect_ms,
        })
    else:
        print(f"[yolo] discarded  motion_{ts}  "
              f"(no target)  fetch={fetch_ms}ms  detect={detect_ms}ms",
              flush=True)
        _log_event(out_dir, {
            "event":     "discarded",
            "file":      f"motion_{ts}.jpg",
            "fetch_ms":  fetch_ms,
            "detect_ms": detect_ms,
        })
        if save_discarded:
            discard_dir = os.path.join(out_dir, "discarded")
            os.makedirs(discard_dir, exist_ok=True)
            cv2.imwrite(os.path.join(discard_dir, f"motion_{ts}.jpg"), frame)


def _yolo_video(model, video_path, out_dir, conf, target_classes, save_discarded):
    """Process a recorded video: save only frames with target detections.

    Detected frames  → {out_dir}/{base}_frames/frame_NNNNN.jpg
    Discarded frames → {out_dir}/{base}_frames/discarded/frame_NNNNN.jpg
                       (only when save_discarded=True)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[yolo] Cannot open {video_path}", file=sys.stderr)
        return

    base       = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(out_dir, f"{base}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    if save_discarded:
        discard_dir = os.path.join(frames_dir, "discarded")
        os.makedirs(discard_dir, exist_ok=True)

    total = saved = 0
    t0    = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1

            results = model(frame, conf=conf, verbose=False)[0]
            hits    = [b for b in results.boxes if int(b.cls) in target_classes]

            if hits:
                _draw_boxes(frame, hits, model.names)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{total:05d}.jpg"), frame)
                saved += 1
            elif save_discarded:
                cv2.imwrite(os.path.join(discard_dir, f"frame_{total:05d}.jpg"), frame)
    finally:
        cap.release()

    detect_ms = int((time.time() - t0) * 1000)

    if saved == 0:
        os.rmdir(frames_dir)
        if save_discarded:
            # remove the now-empty discarded subdir too
            try:
                os.rmdir(discard_dir)
            except OSError:
                pass
        print(f"[yolo] discarded  {os.path.basename(video_path)}  "
              f"no detections in {total} frames  detect={detect_ms}ms",
              flush=True)
        _log_event(out_dir, {
            "event":          "video_discarded",
            "file":           os.path.basename(video_path),
            "frames_total":   total,
            "detect_ms":      detect_ms,
        })
    else:
        print(f"[yolo] DETECTED  {os.path.basename(video_path)}  "
              f"{saved}/{total} frames → {frames_dir}/  detect={detect_ms}ms",
              flush=True)
        _log_event(out_dir, {
            "event":           "video_detected",
            "file":            os.path.basename(video_path),
            "frames_total":    total,
            "frames_detected": saved,
            "frames_dir":      os.path.basename(frames_dir),
            "detect_ms":       detect_ms,
        })


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def fetch_frame(camera_ip, out_dir, yolo_queue=None):
    """Fetch a single JPEG from /control?still=1 (one attempt only).

    With yolo_queue: bytes are queued for YOLO filtering (raw file not saved).
    Without yolo_queue: frame saved immediately as motion_TIMESTAMP.jpg.
    Failures are logged but not retried — the ESP32 can't handle concurrent
    requests and the cat will likely trigger motion again if still present.
    """
    url = f"http://{camera_ip}/control?still=1"
    ts  = time.strftime("%Y%m%d_%H%M%S")
    t0  = time.time()
    try:
        resp    = requests.get(url, timeout=HTTP_TIMEOUT)
        elapsed = time.time() - t0
        if resp.status_code == 200 and resp.content:
            if yolo_queue is not None:
                try:
                    yolo_queue.put_nowait(
                        ("image", resp.content, out_dir, ts, elapsed))
                    print(f"[+] Frame queued for YOLO  "
                          f"({len(resp.content)} bytes  fetch={int(elapsed*1000)}ms)",
                          flush=True)
                except Exception:
                    print("[!] YOLO queue full — frame dropped", file=sys.stderr)
                    _log_event(out_dir, {"event": "queue_full", "ts_frame": ts})
            else:
                filename = os.path.join(out_dir, f"motion_{ts}.jpg")
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"[+] Frame saved: {filename}  "
                      f"({len(resp.content)} bytes  fetch={int(elapsed*1000)}ms)")
        else:
            elapsed = time.time() - t0
            print(f"[!] HTTP {resp.status_code} from {url}  ({int(elapsed*1000)}ms)",
                  flush=True)
            _log_event(out_dir, {"event": "fetch_failed", "ts_frame": ts,
                                  "reason": f"HTTP {resp.status_code}"})
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"[!] Timeout fetching frame  ({int(elapsed*1000)}ms)", flush=True)
        _log_event(out_dir, {"event": "fetch_failed", "ts_frame": ts,
                              "reason": "timeout"})
    except Exception as e:
        print(f"[!] Fetch error: {e}", flush=True)
        _log_event(out_dir, {"event": "fetch_failed", "ts_frame": ts,
                              "reason": str(e)})


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------

def _parse_mjpeg_frames(resp):
    """Yield raw JPEG bytes from an MJPEG multipart response using Content-Length."""
    buf = b""
    for chunk in resp.iter_content(chunk_size=8192):
        buf += chunk
        while True:
            cl_idx = buf.find(b"Content-Length:")
            if cl_idx < 0:
                break
            eol = buf.find(b"\r\n", cl_idx)
            if eol < 0:
                break
            try:
                frame_len = int(buf[cl_idx + 15 : eol].strip())
            except ValueError:
                buf = buf[cl_idx + 15:]
                continue
            hdr_end = buf.find(b"\r\n\r\n", cl_idx)
            if hdr_end < 0:
                break
            data_start = hdr_end + 4
            data_end   = data_start + frame_len
            if len(buf) < data_end:
                break
            yield buf[data_start : data_end]
            buf = buf[data_end:]


def record_video(camera_ip, out_dir, stop_event, max_duration, userdata,
                 yolo_queue=None):
    """Stream MJPEG from /sustain?stream=1 and write to a video file.

    Deadline extends on every motion "on" event stored in
    userdata["last_motion_time"].  Reconnects on stream stall.
    On completion, queues the video file for YOLO processing if yolo_queue set.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[!] opencv-python required for video mode.  "
              "Run: pip install opencv-python", file=sys.stderr)
        return

    url      = f"http://{camera_ip}/sustain?stream=1"
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_dir, f"video_{ts}.avi")
    print(f"[*] Recording video → {filename}  "
          f"(idle-stop after {max_duration}s of no motion)")

    writer      = None
    frame_count = 0
    t0          = time.time()
    last_report = t0

    def deadline_reached():
        last = userdata.get("last_motion_time") or t0
        return time.time() - last >= max_duration

    try:
        while not stop_event.is_set() and not deadline_reached():
            resp = None
            try:
                resp = requests.get(url, stream=True, timeout=(HTTP_TIMEOUT, 10))
                userdata["current_response"] = resp
                if resp.status_code != 200:
                    print(f"[!] Stream rejected: HTTP {resp.status_code}  ({url})")
                    break
                ct = resp.headers.get("Content-Type", "")
                if "multipart" not in ct:
                    print(f"[!] Unexpected Content-Type: {ct!r}  ({url})")
                    break
                for jpg in _parse_mjpeg_frames(resp):
                    if stop_event.is_set() or deadline_reached():
                        break
                    arr   = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    if writer is None:
                        h, w = frame.shape[:2]
                        print(f"[*] Stream: {w}x{h}", flush=True)
                        writer = cv2.VideoWriter(
                            filename, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
                    writer.write(frame)
                    frame_count += 1
                    now = time.time()
                    if now - last_report >= 5:
                        last      = userdata.get("last_motion_time") or t0
                        remaining = max(0, max_duration - (now - last))
                        fps       = frame_count / (now - t0)
                        print(f"[*] {frame_count} frames  {fps:.1f} fps  "
                              f"deadline in {remaining:.0f}s", flush=True)
                        last_report = now
            except requests.exceptions.ReadTimeout:
                if stop_event.is_set() or deadline_reached():
                    break
                print(f"[*] Stream stalled at {frame_count} frames, "
                      f"reconnecting...", flush=True)
                time.sleep(0.5)
                continue
            except Exception as e:
                if stop_event.is_set() or deadline_reached():
                    break
                print(f"[!] Stream error: {e}", flush=True)
                time.sleep(0.5)
                continue
            finally:
                userdata["current_response"] = None
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            if not stop_event.is_set() and not deadline_reached():
                time.sleep(0.2)
    except Exception:
        print("[!] Error during video recording:", file=sys.stderr)
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
        elapsed = time.time() - t0
        if frame_count > 0:
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[+] Video saved: {filename}  "
                  f"({frame_count} frames  {elapsed:.1f}s  {fps:.1f} fps)")
            if yolo_queue is not None:
                try:
                    yolo_queue.put_nowait(("video", filename, out_dir))
                    print(f"[*] Video queued for YOLO processing", flush=True)
                except Exception:
                    print("[!] YOLO queue full — video not queued", file=sys.stderr)
                    _log_event(out_dir, {"event": "queue_full",
                                         "file": os.path.basename(filename)})
        else:
            print("[!] No frames captured — nothing saved")


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def hostname_from_topic(topic):
    parts = topic.split("/")
    try:
        return parts[parts.index("sensor") + 1]
    except (ValueError, IndexError):
        return "unknown"


def on_connect(client, userdata, flags, rc, *args):
    if rc == 0:
        motion_topic = userdata["motion_topic"]
        ldr_topic    = userdata["ldr_topic"]
        client.subscribe(motion_topic)
        client.subscribe(ldr_topic)
        client.subscribe(userdata["lwt_topic"])
        mode = "video" if userdata["video_mode"] else "image"
        yolo = " + YOLO" if userdata["yolo_queue"] is not None else ""
        print(f"[+] Connected ({mode}{yolo} mode). Subscribed to {motion_topic}")
        print(f"[+] Subscribed to LDR: {ldr_topic}  LWT: {userdata['lwt_topic']}")
        if userdata.get("ldr_trigger") and userdata.get("cmd_topic"):
            client.publish(userdata["cmd_topic"], "ldr")
            print(f"[*] LDR trigger sent → {userdata['cmd_topic']}")
    else:
        print(f"[!] Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    try:
        hostname = hostname_from_topic(msg.topic)
        payload  = msg.payload.decode("utf-8", errors="replace").strip()

        # Camera up/down via LWT
        if msg.topic == userdata["lwt_topic"] or msg.topic.endswith("/lwt"):
            status = payload.lower()
            if status == "online":
                print(f"[+] Camera {hostname} is ONLINE")
                _log_event(userdata["out_dir"],
                           {"event": "camera_online", "host": hostname})
            elif status == "offline":
                print(f"[!] Camera {hostname} is OFFLINE")
                _log_event(userdata["out_dir"],
                           {"event": "camera_offline", "host": hostname})
            else:
                print(f"[lwt/{hostname}] {payload}")
            return

        # LDR reading — collect 5 samples, log the median
        if msg.topic == userdata["ldr_topic"] or msg.topic.endswith("/ldr"):
            try:
                data = json.loads(payload)
                buf  = userdata["ldr_buffer"]

                # Discard stale buffer from a previous burst (> 60 s old)
                if buf and time.time() - userdata["ldr_buffer_ts"] > 60:
                    print(f"[ldr/{hostname}] stale buffer ({len(buf)} readings discarded)")
                    buf.clear()

                buf.append(data)
                userdata["ldr_buffer_ts"] = time.time()
                sample = data.get("sample", len(buf))
                print(f"[ldr/{hostname}] sample {sample}/5 — "
                      f"ambient={data.get('ambient','?')} "
                      f"illuminated={data.get('illuminated','?')} "
                      f"differential={data.get('differential','?')}")

                if len(buf) >= 5:
                    # Median of 5: sort by differential, take the middle value
                    sorted_buf  = sorted(buf, key=lambda x: x.get("differential", 0))
                    median_data = sorted_buf[2]
                    food_level  = _log_feeder(median_data, userdata["out_dir"])
                    print(f"[ldr/{hostname}] median → "
                          f"ambient={median_data.get('ambient','?')} "
                          f"illuminated={median_data.get('illuminated','?')} "
                          f"differential={median_data.get('differential','?')} "
                          f"({median_data.get('percent','?')}%)  food={food_level}")
                    buf.clear()
                    ldr_done = userdata.get("ldr_done")
                    if ldr_done is not None:
                        ldr_done.set()
            except json.JSONDecodeError:
                print(f"[ldr/{hostname}] {payload}")
            return

        print(f"[motion/{hostname}] {payload}")

        if payload.lower() == "on":
            _log_event(userdata["out_dir"], {
                "event": "motion_on", "host": hostname})
            yolo_queue = userdata.get("yolo_queue")
            if userdata["video_mode"]:
                userdata["last_motion_time"] = time.time()
                if (userdata["recording_thread"] is None
                        or not userdata["recording_thread"].is_alive()):
                    stop_event = threading.Event()
                    userdata["stop_event"] = stop_event
                    t = threading.Thread(
                        target=record_video,
                        args=(userdata["camera_ip"], userdata["out_dir"],
                              stop_event, userdata["video_timeout"],
                              userdata, yolo_queue),
                        daemon=True,
                    )
                    userdata["recording_thread"] = t
                    t.start()
                else:
                    print(f"[*] Motion extended recording deadline "
                          f"(+{userdata['video_timeout']}s)")
            else:
                threading.Thread(
                    target=fetch_frame,
                    args=(userdata["camera_ip"], userdata["out_dir"], yolo_queue),
                    daemon=True,
                ).start()

        elif payload.lower() == "off":
            _log_event(userdata["out_dir"], {
                "event": "motion_off", "host": hostname})

    except Exception:
        print("[!] Exception in on_message:", file=sys.stderr)
        traceback.print_exc()


def on_callback_exception(client, userdata, callback, exception):
    print(f"[!] Exception in paho callback {callback.__name__}: {exception}",
          file=sys.stderr)
    traceback.print_exception(type(exception), exception, exception.__traceback__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device       = args.hostname if args.hostname else "+"
    motion_topic = f"{args.prefix}sensor/{device}/motion"
    ldr_topic    = f"{args.prefix}sensor/{device}/ldr"
    lwt_topic    = f"{args.prefix}sensor/{device}/lwt"
    cmd_topic    = (f"{args.prefix}sensor/{args.hostname}/cmd"
                    if args.hostname else None)

    if args.ldr_trigger and not args.hostname:
        print("[!] --ldr-trigger requires --hostname", file=sys.stderr)
        sys.exit(1)

    # Start YOLO worker process if requested
    yolo_queue   = None
    yolo_process = None
    if args.yolo:
        yolo_queue = multiprocessing.Queue(maxsize=YOLO_QUEUE_MAXSIZE)
        yolo_process = multiprocessing.Process(
            target=yolo_worker,
            args=(yolo_queue, args.yolo_model, args.yolo_conf,
                  args.yolo_classes, args.save_discarded),
            daemon=True,
        )
        yolo_process.start()
        print(f"[*] YOLO worker started  model={args.yolo_model}  "
              f"conf={args.yolo_conf}  classes={args.yolo_classes}"
              + ("  save-discarded=on" if args.save_discarded else ""))

    ldr_done = threading.Event() if args.ldr_trigger else None

    userdata = {
        "motion_topic":     motion_topic,
        "ldr_topic":        ldr_topic,
        "lwt_topic":        lwt_topic,
        "cmd_topic":        cmd_topic,
        "ldr_trigger":      args.ldr_trigger,
        "ldr_done":         ldr_done,
        "camera_ip":        args.camera,
        "out_dir":          args.outdir,
        "video_mode":       args.video,
        "video_timeout":    args.video_timeout,
        "yolo_queue":       yolo_queue,
        "recording_thread": None,
        "stop_event":       None,
        "current_response": None,
        "last_motion_time": None,
        "ldr_buffer":       [],
        "ldr_buffer_ts":    0.0,
    }

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, userdata=userdata)
    else:
        client = mqtt.Client(userdata=userdata)

    client.on_connect = on_connect
    client.on_message = on_message
    if hasattr(client, "on_callback_exception"):
        client.on_callback_exception = on_callback_exception

    if args.user:
        client.username_pw_set(args.user, args.password)

    print(f"[*] Connecting to MQTT broker at {args.broker}:{args.port} ...")
    print(f"[*] Camera: http://{args.camera}/"
          f"{'sustain?stream=1' if args.video else 'control?still=1'}")
    if args.video:
        print(f"[*] Video timeout: {args.video_timeout}s")
    if args.ldr_trigger:
        print(f"[*] LDR trigger on connect → {cmd_topic}")
    client.connect(args.broker, args.port, keepalive=60)

    if ldr_done is not None:
        # Single-shot mode: connect, send trigger, wait for reading or timeout
        client.loop_start()
        received = ldr_done.wait(timeout=30)
        client.loop_stop()
        client.disconnect()
        if not received:
            print("[!] No LDR reading received within 30s", file=sys.stderr)
            sys.exit(1)
        return

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped.")
        if userdata["stop_event"]:
            userdata["stop_event"].set()
        client.disconnect()
    finally:
        if yolo_queue is not None:
            yolo_queue.put(None)        # shutdown sentinel
        if yolo_process is not None:
            yolo_process.join(timeout=60)
            if yolo_process.is_alive():
                yolo_process.terminate()


if __name__ == "__main__":
    main()
