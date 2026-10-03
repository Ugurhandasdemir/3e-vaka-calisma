"""tests/test_step2.py - Comprehensive verification for STEP 2.

1. Starts app.py on port 7861 as a subprocess, records exact PID.
2. Uses gradio_client to:
   - Call /analyze with samples/cable/test_good/000.png
   - Call /sam_segment with one box
   - Call /save_inspection with the result and operator labels
3. Verifies:
   - operator_labels table contains a row for the new inspection
   - mask PNG exists under DATA_DIR/masks/
   - storage.export_yolo_zip() contains labels_segment/*.txt
4. Kills ONLY the PID that was started.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import storage


def main():
    port = 7861
    server_url = f"http://127.0.0.1:{port}/"

    app_py = ROOT / "app.py"
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Launching app.py on port {port}...")
    proc = subprocess.Popen(
        [sys.executable, str(app_py)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    started_pid = proc.pid
    print(f"Started app.py with PID {started_pid}")

    try:
        # Wait for server to become responsive
        import urllib.request

        ready = False
        for attempt in range(40):
            try:
                with urllib.request.urlopen(server_url, timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1)

        if not ready:
            raise RuntimeError(f"Server on port {port} did not start within 40 seconds")

        print(f"Server is live on {server_url}")

        from gradio_client import Client, handle_file

        client = Client(server_url)
        print("Gradio Client connected successfully.")

        # 1. Call analyze endpoint
        sample_img_path = str(ROOT / "samples" / "cable" / "test_good" / "000.png")
        print(f"Calling /analyze on {sample_img_path}...")
        analyze_res = client.predict(
            image=handle_file(sample_img_path),
            product_group="Gözetleme ünitesi",
            serial_no="SN-STEP2-TEST",
            inspector="QC-Tester",
            api_name="/analyze",
        )
        print("Analyze completed successfully.")

        # 2. Call SAM endpoint with one box
        print("Calling /sam_segment with one box...")
        box_coords = [100, 100, 220, 220]
        sam_payload = {
            "image": handle_file(sample_img_path),
            "boxes": [
                {
                    "xmin": box_coords[0],
                    "ymin": box_coords[1],
                    "xmax": box_coords[2],
                    "ymax": box_coords[3],
                    "label": "Yüzey çiziği",
                }
            ],
            "orientation": 0,
        }

        sam_res = client.predict(
            annotations=sam_payload,
            fallback_image=handle_file(sample_img_path),
            api_name="/sam_segment",
        )
        print("SAM segment completed.")
        # sam_res returns: (sam_overlay_view, sam_status_view)
        print("SAM Status result:", sam_res[1])

        # 3. Call save_inspection endpoint
        print("Calling /save_inspection...")
        save_res = client.predict(
            image=handle_file(sample_img_path),
            product_group="Gözetleme ünitesi",
            serial_no="SN-STEP2-TEST",
            inspector="QC-Tester",
            user_decision="Ret",
            selected_defect_type="Yüzey çiziği",
            note="STEP 2 test kaydı: operatör SAM ile çizik ekledi",
            annotations=sam_payload,
            api_name="/save_inspection",
        )
        conf_msg = save_res[0]
        print("Save confirmation message:", conf_msg)

        # 4. Verify DB and Filesystem
        inspections = storage.list_inspections(limit=5)
        latest = inspections[0]
        rec_id = latest["id"]
        print(f"Latest inspection record ID: {rec_id}")

        op_labels = storage.get_operator_labels(rec_id)
        print(f"Operator labels from DB for record {rec_id}: {op_labels}")
        assert len(op_labels) >= 1, "No operator_labels row found for record"
        op_row = op_labels[0]
        assert op_row["label"] == "Yüzey çiziği"
        assert op_row["source"] == "operator"

        mask_path = op_row["mask_path"]
        print(f"Mask path in DB: {mask_path}")
        assert mask_path and Path(mask_path).exists(), f"Mask file does not exist at {mask_path}"
        print(f"Mask file verified on disk: {mask_path} (size: {Path(mask_path).stat().st_size} bytes)")

        # 5. Verify export_yolo_zip()
        print("Exporting active-learning YOLO zip...")
        zip_path = storage.export_yolo_zip()
        print(f"YOLO zip exported to: {zip_path}")
        assert Path(zip_path).exists()

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            print("Zip contents:", namelist)
            # Must contain labels_segment/*.txt
            segment_files = [n for n in namelist if n.startswith("labels_segment/") and n.endswith(".txt")]
            detect_files = [n for n in namelist if n.startswith("labels_detect/") and n.endswith(".txt")]
            print("Found segment label files:", segment_files)
            print("Found detect label files:", detect_files)
            assert len(segment_files) > 0, "No labels_segment/*.txt files found in zip!"
            assert len(detect_files) > 0, "No labels_detect/*.txt files found in zip!"

            target_seg_file = f"labels_segment/{rec_id}.txt"
            assert target_seg_file in namelist, f"Expected {target_seg_file} in zip"
            seg_content = zf.read(target_seg_file).decode("utf-8").strip()
            print(f"Content of {target_seg_file} (first 120 chars):\n  {seg_content[:120]}")
            parts = seg_content.split()
            cls_id = parts[0]
            pts = parts[1:]
            print(f"Class ID: {cls_id}, Number of polygon coordinates: {len(pts)}")
            assert len(pts) >= 6, "Expected at least 3 (x, y) coordinate pairs for segment polygon"

        # 6. Verify dashboard metric in storage.stats()
        stats = storage.stats()
        print("Dashboard stats missed_by_ai:", stats.get("missed_by_ai"))
        assert stats.get("missed_by_ai", 0) >= 1, "missed_by_ai should be >= 1"

        print("\n==========================================")
        print("ALL STEP 2 TEST VERIFICATIONS PASSED!")
        print("==========================================\n")

    finally:
        # Kill ONLY the PID we started
        print(f"Terminating started PID {started_pid}...")
        try:
            os.kill(started_pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.kill(started_pid, signal.SIGKILL)
            except Exception:
                pass
        print(f"Process PID {started_pid} terminated.")


if __name__ == "__main__":
    main()
