# Local stack — cropping MCP server + MinIO

Self-contained dev/demo replacement for the private `crop_image_tool_crop_image_post`
service. After the steps below, the main repo's cropping flow runs entirely on
your machine.

**Dev/demo only.** Default credentials are `minioadmin/minioadmin`. Do not
expose ports 9000/9001/8000 to the internet.

## Quickstart

1. **Prereqs**: Docker (or compatible runtime), [uv](https://docs.astral.sh/uv/), Python 3.12+, macOS or Linux.

2. **Bring up MinIO + bucket init** (from `tools/local-stack/`):
   ```bash
   docker compose up -d
   docker compose ps          # wait until minio is "healthy" and minio-init exits 0
   ```
   If MinIO exits with `FATAL Unable to use the drive /data: drive not found`,
   create the host-side bind path manually and retry: `mkdir -p .data/minio && docker compose up -d`.

3. **Install server deps** (creates `tools/local-stack/.venv`):
   ```bash
   uv sync
   ```

4. **Upload a sample image** — prints the URL you'll feed to the smoke test:
   ```bash
   uv run python upload_sample.py
   # http://localhost:9000/cv-agent/samples/sample.jpg
   ```

5. **Start the cropping MCP server** (leave running in this terminal):
   ```bash
   uv run python server.py
   ```
   It binds `0.0.0.0:8000` and serves at `/mcp`.

6. **Run the end-to-end smoke test** — from the **main repo root**, using the
   main repo's venv (which has `cv_agent` editable-installed):
   ```bash
   cd ../../              # back to toolgate/
   uv run python tools/local-stack/smoke_test.py
   ```
   Expected output (last 3 lines):
   ```
   data={'status': 'success', 'output_image': 'http://localhost:9000/cv-agent/crops/<hex>.jpg'}
   GET http://localhost:9000/cv-agent/crops/<hex>.jpg -> 200 image/jpeg
   OK
   ```

7. **(Optional) Wire it into the agent.** In the main repo's `.env`:
   ```bash
   CV_AGENT_CROPPING_MCP_URL=http://localhost:8000/mcp
   ```
   Then run a benchmark config that uses cropping (`configs/boed-full.yaml` etc.).

## Configuration

All settings are env vars; defaults match the bundled compose file.

| Variable | Default | Notes |
|---|---|---|
| `MINIO_ENDPOINT` | `http://localhost:9000` | Where the server PUTs and pulls. |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / `minioadmin` | Match `docker-compose.yml`. |
| `MINIO_BUCKET` | `cv-agent` | Auto-created with anonymous-download policy. |
| `MINIO_PUBLIC_BASE_URL` | = `MINIO_ENDPOINT` | Override when downstream consumers (VLM, other tools) need a different host than the server itself uses. |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8000` | Server bind. |
| `CROP_DOWNLOAD_TIMEOUT` | `30` (seconds) | HTTP timeout when the server fetches the source image before cropping. Bump for slow upstreams. |

## Architecture note: why the server runs on the host, not in docker

The URL the server returns in `output_image` must be reachable from BOTH the
agent process AND any downstream tool/VLM. If the server runs in docker, MinIO
is at `minio:9000` from inside the network but `localhost:9000` from the host —
returning either makes the URL useless on the other side.

To dockerize anyway, set `MINIO_PUBLIC_BASE_URL=http://host.docker.internal:9000`
(macOS) or a host-reachable IP (Linux), and put the server in the same compose
network so its outbound PUT can reach `minio:9000` while it advertises the
host-reachable URL.

## VLM caveat

The default `output_image` URL is `http://localhost:9000/...`. This works only
when your VLM also runs on the host (or in a container with `--network host`).
For a remote VLM endpoint, set `MINIO_PUBLIC_BASE_URL` to a publicly-reachable
host (e.g. via cloudflared tunnel or a real S3) so the VLM can fetch the crop.

## Known limitations

> BOED selection-coordinate echo (`original_coordinates` /
> `mcmc_selected_coordinates` / `look_ahead_selected_coordinates` /
> `boed_selected_coordinates`) is not implemented in this minimal server.
> `_post_process_tool_output` (`cv_agent_nodes.py:549-556`) reads them directly
> off the tool's returned data and for cropping no upstream code mutates `data`
> between tool return and post-process — so BOED-mode telemetry will silently
> drop those fields. Functionally fine (all reads are guarded by
> `if "X" in data`), but use the production server if you need BOED telemetry.

## Teardown

```bash
docker compose down -v          # also wipes the bucket; remove `-v` to keep data
rm -rf .data .venv               # full reset
```


---

## Other tool servers (bring your own — model-backed)

This local-stack provides **MinIO + the `cropping` tool** (model-free). The
agent also calls model-backed tools that you serve yourself as MCP endpoints
(no model weights are shipped). Point the agent at them via `server_url` in
`configs/*.yaml` — defaults: detection/segmentation/depth/crop/bbox on
`http://localhost:10009/mcp`, OCR on `http://localhost:10011/mcp`.

Every tool takes a publicly-fetchable `image_url` (e.g. a MinIO URL); image ops
write results back to MinIO and return an `output_image` URL. The **MCP tool
name registered on your server must match** the names below.

| MCP tool name | extra params | returns | typical model |
|---|---|---|---|
| `detection` | `targets: list[str]` | `[{label, bbox:[x1,y1,x2,y2], score}]` | open-vocab detector (e.g. GroundingDINO) |
| `detection_small_object` | `targets: list[str]` | same as detection | detector + tiling/hi-res |
| `ocr` | — | extracted text / blocks | OCR (e.g. MinerU, PaddleOCR) |
| `segmentation_segmentation_post` | `text_prompt: str` | `output_image` (mask overlay) | text-prompted segmenter (e.g. GroundingDINO + SAM) |
| `depth_estimation_depth_estimation_post` | — | `output_image` (depth map) | monocular depth (e.g. Depth-Anything) |
| `crop_image_tool_crop_image_post` | `coordinates: [x1,y1,x2,y2]` | `output_image` (crop) | **provided here** (`server.py`) |
| `bbox_drawing_tool_bbox_drawing_post` | `boxes`, `line_thickness` | `output_image` | pure PIL (no model) |
| `bbox_center_labeling_tool_bbox_center_labeling_post` | `bboxes`, `font_size` | `output_image` | pure PIL (no model) |

Serve them with any MCP framework (`fastmcp`, as in `server.py`). The two
`bbox_*` tools are pure-PIL like cropping and can be added to `server.py`
trivially if you want them local too; `detection`/`ocr`/`segmentation`/`depth`
need real models, so wrap your own and expose them at the URLs above.
