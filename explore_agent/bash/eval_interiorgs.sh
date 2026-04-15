SCENE_FILE="YOUE_EVAL_SCENE_PATH"
RENDER_DIR="YOUR_RENDER_DIR"
INFER_DIR="YOUR_EXPLORE_DIR"
PLY_ROOT_BASE="YOUR_3DGS_ASSETS_DIR"
RENDER_ENV_PATH="YOUR_RENDER_ENV"
INFER_ENV_PATH="YOUR_INFER_ENV"
INFER_SCRIPT="scripts/infer_close_3dgs_singal.py"
PID_FILE="/tmp/render_sim.pid"

eval "$(conda shell.bash hook)"
cd "$RENDER_DIR" || { echo "Work dir not found"; exit 1; }

kill_current() {
    if [ -f "$PID_FILE" ]; then
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Killing render_sim (PID: $old_pid)"
            kill "$old_pid"
            wait "$old_pid" 2>/dev/null
        fi
        rm -f "$PID_FILE"
    fi
    pkill -f "render_sim.py" 2>/dev/null
    sleep 2
}

while IFS= read -r scene_id; do
    [[ -z "$scene_id" ]] && continue
    echo ">>> Processing scene: $scene_id"
    kill_current

    echo "Starting render service for $scene_id..."
    cd "$RENDER_DIR" || { echo "Work dir not found"; exit 1; }
    nohup conda run -p "$RENDER_ENV_PATH" python render_sim.py \
        --load_ply_root "$PLY_ROOT_BASE/$scene_id/gs/" \
        --gpus 0 \
        &
    NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"
    echo "Render service started (PID: $NEW_PID)"
    sleep 8 

    echo "Running inference for $scene_id..."
    cd "$INFER_DIR" || { echo "Work dir not found"; exit 1; }
    conda run -p "$INFER_ENV_PATH" python "$INFER_SCRIPT" -s "$scene_id" -c config/infer_interiorgs.yaml
    kill_current
    sleep 2

done < "$SCENE_FILE"

echo "✅ All scenes processed."
