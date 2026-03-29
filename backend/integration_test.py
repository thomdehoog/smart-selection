"""
Integration test — end-to-end smoke test for the backend API.

Calls every endpoint in the correct order and validates response shapes.
Works against both the real backend (with models) and the mock server.

Usage:
    # Against mock server:
    python mock_server.py &
    python integration_test.py

    # Against real backend with BBBC021 data:
    python app.py &
    python integration_test.py /path/to/bbbc021_raw/Week1_22123/
"""

import sys
import time
import json
import requests

BASE = "http://localhost:5050/api"


def check(label, response, expected_keys=None):
    """Validate a response and print pass/fail."""
    data = response.json()
    ok = response.status_code == 200
    if expected_keys:
        for k in expected_keys:
            if k not in data:
                print(f"  FAIL {label}: missing key '{k}' in response")
                ok = False
    if ok:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}: status={response.status_code}")
        print(f"        {json.dumps(data, indent=2)[:200]}")
    return data, ok


def main():
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "/mock/path"
    all_pass = True

    print("\n=== Integration Test ===\n")

    # 1. Upload
    print("Step 1: Upload")
    r = requests.post(f"{BASE}/upload_bbbc021", json={"image_dir": image_dir, "n": 5})
    data, ok = check("upload", r, ["dataset_id", "num_images", "channel_names"])
    all_pass &= ok
    num_images = data.get("num_images", 0)
    print(f"         {num_images} images, channels: {data.get('channel_names')}")

    # 2. Segment & embed
    print("\nStep 2: Segment & Embed")
    r = requests.post(f"{BASE}/segment_and_embed")
    data, ok = check("start pipeline", r, ["status"])
    all_pass &= ok

    # 3. Poll until complete (max 5 minutes)
    print("\nStep 3: Poll Status")
    deadline = time.time() + 300
    while time.time() < deadline:
        r = requests.get(f"{BASE}/status")
        data = r.json()
        status = data.get("status")
        progress = data.get("progress", 0)
        msg = data.get("message", "")
        print(f"         [{progress:.0%}] {msg}")

        if status == "complete":
            print(f"  PASS pipeline complete — {data.get('num_objects', 0)} objects")
            break
        elif status == "error":
            print(f"  FAIL pipeline error: {data.get('error')}")
            all_pass = False
            break
        time.sleep(2)
    else:
        print("  FAIL pipeline timed out after 5 minutes")
        all_pass = False

    num_objects = data.get("num_objects", 0)
    if num_objects == 0:
        print("\n  ABORT: No objects detected, cannot test search endpoints")
        sys.exit(1)

    # 4. Get image
    print("\nStep 4: Get Image")
    r = requests.get(f"{BASE}/image/0")
    data, ok = check("get image 0", r, ["thumbnail_base64", "width", "height"])
    all_pass &= ok
    has_b64 = data.get("thumbnail_base64", "").startswith("data:image/png;base64,")
    print(f"         valid base64 thumbnail: {has_b64}")
    all_pass &= has_b64

    # 5. Get objects
    print("\nStep 5: Get Objects")
    r = requests.get(f"{BASE}/objects?image_index=0")
    data, ok = check("get objects", r, ["objects", "num_objects"])
    all_pass &= ok
    objects = data.get("objects", [])
    print(f"         {len(objects)} objects in image 0")
    if objects:
        obj = objects[0]
        for key in ["object_id", "centroid", "bbox", "area"]:
            if key not in obj:
                print(f"  FAIL object missing key: {key}")
                all_pass = False

    # 6. Get crops for first 3 objects
    print("\nStep 6: Get Crops")
    sample_ids = [o["object_id"] for o in objects[:3]]
    r = requests.post(f"{BASE}/crops", json={"object_ids": sample_ids})
    data, ok = check("get crops", r, ["crops"])
    all_pass &= ok
    crops = data.get("crops", [])
    print(f"         {len(crops)} crops returned for {len(sample_ids)} requested")
    all_pass &= len(crops) == len(sample_ids)
    if crops:
        has_thumb = crops[0].get("thumbnail_base64", "").startswith("data:image/png;base64,")
        print(f"         valid crop thumbnail: {has_thumb}")
        all_pass &= has_thumb

    # 7. Search with positives only
    print("\nStep 7: Search (positives only)")
    r = requests.post(f"{BASE}/search", json={
        "positive_ids": sample_ids,
        "negative_ids": [],
        "top_k": 10,
        "negative_alpha": 0.4,
    })
    data, ok = check("search", r, ["results"])
    all_pass &= ok
    results = data.get("results", [])
    print(f"         {len(results)} results returned")

    # Verify no positive IDs in results
    result_ids = {res["object_id"] for res in results}
    leaked = result_ids & set(sample_ids)
    if leaked:
        print(f"  FAIL positive IDs leaked into results: {leaked}")
        all_pass = False
    else:
        print(f"  PASS positive IDs correctly excluded from results")

    # Verify scores are valid
    if results:
        scores = [res["similarity_score"] for res in results]
        valid_scores = all(-1.01 <= s <= 1.01 for s in scores)
        sorted_desc = all(scores[i] >= scores[i + 1] - 0.001 for i in range(len(scores) - 1))
        print(f"         scores valid: {valid_scores}, sorted descending: {sorted_desc}")
        all_pass &= valid_scores and sorted_desc

    # 8. Search with negatives (reject the top result)
    print("\nStep 8: Search (with rejection)")
    if results:
        rejected_id = results[0]["object_id"]
        r = requests.post(f"{BASE}/search", json={
            "positive_ids": sample_ids,
            "negative_ids": [rejected_id],
            "top_k": 10,
            "negative_alpha": 0.4,
        })
        data, ok = check("search with negative", r, ["results"])
        all_pass &= ok
        new_results = data.get("results", [])
        neg_leaked = rejected_id in {res["object_id"] for res in new_results}
        if neg_leaked:
            print(f"  FAIL rejected ID {rejected_id} appeared in results")
            all_pass = False
        else:
            print(f"  PASS rejected ID {rejected_id} correctly excluded")

    # 9. Export
    print("\nStep 9: Export")
    export_ids = sample_ids + [results[0]["object_id"]] if results else sample_ids
    r = requests.post(f"{BASE}/export", json={"accepted_ids": export_ids})
    data, ok = check("export", r, ["exported_objects"])
    all_pass &= ok
    print(f"         {len(data.get('exported_objects', []))} objects exported")

    # Summary
    print("\n" + "=" * 40)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — see above")
    print("=" * 40)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
