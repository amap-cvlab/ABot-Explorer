import os
import json
import glob
import csv

root_dir = "YOUR_ROOT_DIR"
exp_date = "YOUR_EXP_DATE"
scene_id_file = "YOUR_SCENE_ID_FILE"

with open(scene_id_file, "r") as f:
    valid_scenes = {line.strip() for line in f if line.strip()}

json_files = glob.glob(os.path.join(root_dir, exp_date, "*", "*", "result.json"))

scene_results = []
for file_path in json_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        scene_name = data.get("scene_name")
        if not scene_name or scene_name not in valid_scenes:
            continue

        fcr = data.get("occ_cr")
        fncr = data.get("node_cr")
        path_len = data.get("pl")
        explored_area = data.get("occ_area")
        matched_nodes = data.get("node_count")

        if None in (fcr, fncr, path_len, explored_area, matched_nodes) or path_len <= 0:
            continue

        node_eff = matched_nodes / path_len
        occ_eff = explored_area / path_len

        scene_results.append(
            {
                "scene_id": scene_name,
                "occ_cr": round(fcr, 3),
                "node_cr": round(fncr, 3),
                "pl": round(path_len, 3),
                "node_eff": round(node_eff, 3),
                "occ_eff": round(occ_eff, 3),
            }
        )
    except Exception:
        pass

scene_results.sort(key=lambda x: x["scene_id"])

csv_path = os.path.join(root_dir, exp_date, "summary_table.csv")
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "scene_id",
        "occ_cr",
        "node_cr",
        "pl",
        "node_eff",
        "occ_eff",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(scene_results)

if scene_results:
    n = len(scene_results)
    mean_fcr = sum(r["occ_cr"] for r in scene_results) / n
    mean_fncr = sum(r["node_cr"] for r in scene_results) / n
    mean_path = sum(r["pl"] for r in scene_results) / n
    mean_node_per_len = sum(r["node_eff"] for r in scene_results) / n
    mean_area_per_len = sum(r["occ_eff"] for r in scene_results) / n

    print(f"Saved: {csv_path}")
    print(f"Scenes: {n}")
    print(f"Mean Occupancy Coverage Rate: {mean_fcr:.3f}")
    print(f"Mean Node Coverage Rate:      {mean_fncr:.3f}")
    print(f"Mean Path Length (m):         {mean_path:.3f}")
    print(f"Mean Nodes per Meter:         {mean_node_per_len:.3f}")
    print(f"Mean Area per Meter (m^2/m):  {mean_area_per_len:.3f}")

else:
    print("No valid scenes processed.")
