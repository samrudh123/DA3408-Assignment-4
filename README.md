# DA3408-Assignment-4

CPU-only image captioning app built with Streamlit and instrumented with `prometheus_client`.

## Deliverables

| Deliverable | Location |
| --- | --- |
| Streamlit app | `app.py`, `captioner.py`, `metrics.py`, `run.py` |
| Prometheus config | `prometheus.yml`, `rules/recording_rules.yml` |
| Grafana dashboard (JSON export) | `grafana/dashboards/caption-app.json` |
| Dashboard video (59.9 s, 5 s refresh) | `A4_dashboard_video.mp4`, https://drive.google.com/file/d/1Rd4reHYfa6PQ7GM2Kwxzi0nhFg0r_5pN/view?usp=sharing |
| Metric verification report (hand calculations) | [`A4_Report.pdf`](A4_Report.pdf) |
| AI usage disclosure | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) |

## Features

- **Single mode**: upload one image and get a caption.
- **Bulk mode**: upload a ZIP of images; every image is captioned with a progress bar, results are shown with thumbnails and can be downloaded as CSV. Hidden files, `__MACOSX` entries, non-images, encrypted and oversized entries are skipped and listed.
- **Models**: `Salesforce/blip-image-captioning-base` (default) or `nlpconnect/vit-gpt2-image-captioning`, selectable in the sidebar along with max tokens and beam width.
- **CPU safety**: one shared inference lock (images queue instead of oversubscribing the CPU), images downscaled to 1024 px before inference, per-image pixel limit, per-ZIP image and size limits, and corrupt files reported instead of crashing the app.
- **Metrics**: exposed on a separate port at `http://localhost:18000/metrics`.

## Repository layout

```
.
├── app.py                     Streamlit UI (single + bulk ZIP captioning)
├── captioner.py               Model loading and CPU inference
├── metrics.py                 All prometheus_client metrics and the /metrics server
├── run.py                     Launcher: starts /metrics, preloads the model, then Streamlit
├── requirements.txt           Pinned Python dependencies (CPU-only torch)
├── .streamlit/config.toml     Streamlit server settings
├── prometheus.yml             Prometheus scrape config
├── rules/recording_rules.yml  Recording rules used by the dashboard
├── docker-compose.yml         Grafana service
├── grafana/
│   ├── provisioning/          Datasource + dashboard provisioning
│   └── dashboards/caption-app.json   Dashboard (JSON export)
├── A4_Report.pdf              Metric verification report (hand calculations vs Prometheus/Grafana)
├── AI_DISCLOSURE.md           AI usage disclosure and summary of the metric checks
└── README.md
```

## Reproducing the results

Tested on Ubuntu 26.04 (x86_64) with Python 3.12, Docker 29 with Compose v2, Prometheus 3.15.0, node_exporter 1.12.1 and Grafana 13.2.4. All commands are run from the repository root unless stated otherwise.

### 1. Prerequisites

| Requirement | Version tested | Check with |
| --- | --- | --- |
| Linux x86_64 | Ubuntu 26.04 | `uname -m` |
| Python | 3.12 (3.10+ should work) | `python3 --version` |
| Docker Engine + Compose v2 | 29.1 / 2.40 | `docker compose version` |
| Prometheus + promtool | 3.15.0 | `prometheus --version` (step 4 installs it) |
| node_exporter | 1.12.1 | `node_exporter --version` (step 4 installs it) |
| git, curl, zip | any | |

You'll also need:
- About 6 GB of free disk: Python packages ~1.5 GB, the captioning model ~1 GB, the Grafana image ~2 GB.
- At least 4 GB of RAM.
- These ports free: **8501, 18000, 9090, 9100, 3000**. Check with `ss -ltnp | grep -E ':(8501|18000|9090|9100|3000)\b'`, which should print nothing.

### 2. Get the code

```bash
git clone https://github.com/samrudh123/DA3408-Assignment-4.git
cd DA3408-Assignment-4
```

### 3. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pulls the CPU-only build of torch from the PyTorch index, so no GPU or CUDA is needed. A conda environment (`conda create -n a4 python=3.12`, then the same `pip install` line) works too.

### 4. Install Prometheus and node_exporter

Skip this step if `prometheus`, `promtool` and `node_exporter` are already on your `PATH`. Otherwise, this installs them into `~/.local/bin` without `sudo`:

```bash
mkdir -p ~/.local/bin /tmp/a4-bin && cd /tmp/a4-bin
curl -sSLO https://github.com/prometheus/prometheus/releases/download/v3.15.0/prometheus-3.15.0.linux-amd64.tar.gz
curl -sSLO https://github.com/prometheus/node_exporter/releases/download/v1.12.1/node_exporter-1.12.1.linux-amd64.tar.gz
tar xzf prometheus-3.15.0.linux-amd64.tar.gz
tar xzf node_exporter-1.12.1.linux-amd64.tar.gz
cp prometheus-3.15.0.linux-amd64/prometheus prometheus-3.15.0.linux-amd64/promtool \
   node_exporter-1.12.1.linux-amd64/node_exporter ~/.local/bin/
cd - && export PATH="$HOME/.local/bin:$PATH"
prometheus --version | head -1 && node_exporter --version | head -1
```

Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` (or `~/.zshrc`) so new terminals find the binaries.

### 5. Pull the Grafana image and validate the configs

```bash
docker compose pull grafana
promtool check config prometheus.yml
```

Expected output: `SUCCESS: prometheus.yml is valid prometheus config file syntax` and `SUCCESS: 21 rules found`.

### 6. Create the sample data

This makes 50 COCO validation images, plus one non-image file and one truncated JPEG so that skipped and failed files show up in the metrics:

```bash
mkdir -p sample_data/imgs && cd sample_data/imgs
for id in 000000039769 000000000139 000000000285 000000000632 000000000724 \
          000000000776 000000000785 000000000802 000000000872 000000000885; do
  curl -sfLO http://images.cocodataset.org/val2017/$id.jpg
done
for f in 0*.jpg; do for k in 1 2 3 4; do cp "$f" "copy${k}_$f"; done; done
echo "not an image" > notes.txt
head -c 2000 000000000139.jpg > broken.jpg
zip -q ../demo_bulk.zip *
cd ../..
```

The result is `sample_data/demo_bulk.zip` (50 good images, 1 broken image, 1 text file). `sample_data/` is git-ignored.

### 7. Start the stack (four terminals)

Open four terminals, `cd` into the repository in each, and start one component per terminal **in this order**. Leave each one running.

**Terminal 1: node_exporter** (host CPU, RAM, disk and network metrics)

```bash
node_exporter --web.listen-address=127.0.0.1:9100 --collector.processes --collector.cpu.info
```

Ready when the log shows `msg="Listening on" address=127.0.0.1:9100`.

**Terminal 2: the caption app**

```bash
source .venv/bin/activate
python run.py
```

Ready when you see `metrics exposed on http://0.0.0.0:18000/metrics`, `loaded Salesforce/blip-image-captioning-base`, and Streamlit's `Local URL: http://localhost:8501`. The first run downloads the model (~1 GB) into `~/.cache/huggingface`. A warning about unauthenticated requests to the HF Hub is harmless.

**Terminal 3: Prometheus**

```bash
prometheus --config.file=prometheus.yml --storage.tsdb.path=data/prometheus \
  --web.listen-address=127.0.0.1:9090 --web.enable-lifecycle
```

Ready when the log shows `Server is ready to receive web requests.` Data goes to `data/` (git-ignored).

**Terminal 4: Grafana**

```bash
docker compose up grafana
```

Ready when the log shows `msg="HTTP Server Listen" address=127.0.0.1:3000`. Grafana uses host networking so it can reach Prometheus on `localhost:9090`.

### 8. Check that everything is connected

In a fifth terminal (repository root):

```bash
curl -s localhost:9090/api/v1/targets | python3 -c 'import json,sys; [print(t["labels"]["job"], t["health"]) for t in json.load(sys.stdin)["data"]["activeTargets"]]'
curl -s localhost:18000/metrics | grep -c '^caption_'
curl -s localhost:3000/api/health
```

Expected:
- The first command prints `caption_app up`, `node up` and `prometheus up`.
- The second prints a count of well over 100 `caption_` sample lines.
- The third prints JSON with `"database": "ok"` and `"version": "13.2.4"`.

| URL | What you should see |
| --- | --- |
| http://localhost:8501 | The captioning app with **Single image** and **Bulk (ZIP)** tabs |
| http://localhost:18000/metrics | Raw application metrics (`caption_*`, `process_*`) |
| http://localhost:9100/metrics | Raw host metrics (`node_*`) |
| http://localhost:9090/targets | All three scrape targets **UP** |
| http://localhost:3000 | The Grafana dashboard *CPU Image Captioning · Service and Host Monitoring* (no login needed; `admin` / `admin` to edit) |

### 9. Generate the workload and watch the dashboard

1. Open Grafana at http://localhost:3000/d/caption-app?orgId=1&from=now-5m&to=now&refresh=5s. It refreshes every 5 s.
2. In the app at http://localhost:8501, open **Single image**, upload `sample_data/imgs/000000039769.jpg` and click **Generate caption**. You should get a caption like "two cats sleeping on a couch".
3. Open **Bulk (ZIP)**, upload `sample_data/demo_bulk.zip` and click **Caption all images**. The job takes about a minute on a modern laptop CPU (longer on fewer cores).

What to look for in Grafana while the job runs:

| Row | Expected behaviour |
| --- | --- |
| 1 · Service health | App and node_exporter **▲ UP**, model **● loaded**, **Requests in flight** = 1, **Images captioned** rising |
| 2 · Traffic | A `bulk` plateau in **Throughput**; `bulk images pending` counting down from 51 |
| 3 · Latency | p50/p95/p99 of roughly 1–3 s per image (depends on the CPU) |
| 4 · Correlation | **Host CPU busy** and the **caption app process** share rising and falling together inside a grey "bulk job running" band |
| 5 · Host resources | User CPU share and load per core rise; memory stays flat |
| 6 · Model and inputs | **Skipped ZIP entries**: `unsupported_extension` = 1; **Errors**: `decode / OSError` = 1; **Clients** lists your IP, session and browser |

When it finishes, the app shows 50 captioned, 1 failed and 1 skipped, and the **Image error ratio** tile reads about 1.9 % (1 of 52 images).

### 10. Check the metric logic by hand

[`A4_Report.pdf`](A4_Report.pdf) shows the worked calculations from the recorded run (see [Metric verification report](#metric-verification-report-a4_reportpdf) below). To repeat them on your own run, these commands print the raw values each formula needs:

| Check | Raw values |
| --- | --- |
| Throughput (counter → rate) | Prometheus query `caption_images_processed_total{mode="bulk",status="success",model="blip-base"}[1m]` (Table tab, during the job) |
| Error ratio | `curl -s localhost:18000/metrics \| grep '^caption_images_processed_total' \| grep blip-base` |
| Average request time (Summary) | `curl -s localhost:18000/metrics \| grep -E '^caption_request_seconds_(sum\|count)'` |
| p95 latency (Histogram) | `curl -s localhost:18000/metrics \| grep '^caption_inference_seconds_bucket' \| grep 'mode="bulk"' \| grep blip-base` |
| Host CPU and app CPU share | `node_cpu_seconds_total{mode="idle"}` from `:9100/metrics` and `process_cpu_seconds_total` from `:18000/metrics`, read twice 10 s apart |
| Host memory | `curl -s localhost:9100/metrics \| grep -E '^node_memory_(MemTotal\|MemAvailable)_bytes'` |
| Disk free on `/` | `curl -s localhost:9100/metrics \| grep -E '^node_filesystem_(avail\|size)_bytes\{.*mountpoint="/"\}'` |

To compare the p95 exactly, run this Prometheus query, which uses the same lifetime bucket counts: `histogram_quantile(0.95, sum by (le) (caption_inference_seconds_bucket{mode="bulk",model="blip-base"}))`.

### 11. Stop and clean up

Press **Ctrl+C** in terminals 4, 3, 2 and 1, then:

```bash
docker compose down
rm -rf data
```

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `address already in use` | Another process holds the port. Find it with `ss -ltnp \| grep <port>` and stop it, or run `docker compose down` for port 3000 |
| Grafana panels say **No data** | Check that Prometheus (terminal 3) is running and http://localhost:9090/targets shows all targets UP. On Docker Desktop (macOS/Windows), turn on *Settings → Resources → Network → Enable host networking* (Docker Desktop 4.34+), since the compose file uses `network_mode: host` |
| `caption_app` target DOWN | The app isn't running, or was started with `streamlit run app.py` (its metrics only come up after the first page visit). Use `python run.py` |
| Model download fails or is slow | Re-run `python run.py`. Downloads resume from `~/.cache/huggingface`. Setting `HF_TOKEN` raises the Hugging Face rate limit |
| Machine slows down during bulk jobs | Limit inference threads, e.g. `TORCH_THREADS=4 python run.py` |
| Rates or percentiles look empty right after start-up | They need about 1 minute of samples (5 s scrape, 30 s to 1 m windows) |

## Metric verification report (`A4_Report.pdf`)

The assignment's *Black Box Metric Rule* requires the logic of each AI-generated metric to be calculated by hand. `A4_Report.pdf` (2 pages) does this for one recorded run of the full stack: the same run as the dashboard video (one single image, then a 25-entry ZIP holding 23 valid images, 1 truncated JPEG and 1 text file).

| Section | Contents |
| --- | --- |
| Summary | One table of all checks: hand-calculated value, Prometheus/dashboard value, and whether they match (10 of 10 match) |
| 1. Purpose and method | How the raw values were taken (exporter `/metrics` snapshots S0–S3 and raw Prometheus samples), the workload, and the AI-use disclosure for the calculations |
| 2. Hand calculations | For each metric: raw values in a table, the formula, the substituted arithmetic, the result, and the comparison |

The checks cover every metric type the app exports, plus node_exporter:

| Check | Metric type | Hand result | Compared with |
| --- | --- | --- | --- |
| M1 Throughput | Counter → rate | 0.6667 img/s | `rate()` and the Throughput panel: 0.6667 |
| M2 Image error ratio | Counter → ratio | 3.061 % lifetime, 4.000 % this run | Prometheus 3.061 %, dashboard tile 4.000 % |
| M3 Average request time | Summary `_sum ÷ _count` | 40.381 s lifetime, 33.823 s this job | Prometheus 40.381 s, app's own timer 33.8 s |
| M4 p95 inference latency | Histogram (bucket interpolation) | 1.8875 s | `histogram_quantile`: 1.8875 s |
| M5 Host CPU busy | node_exporter Counter | 69.0 % | Same-window PromQL 69.2 % |
| M6 App share of host CPU | process Counter | 38.5 % | Same-window PromQL 37.5 % |
| M7 Host memory used | Gauges | 67.69 % | Memory panel 67.31 % |
| M8 Disk free on `/` | Gauges | 48.49 % | Disk panel 48.49 % |

The numbers are specific to that run and machine. To check the logic on your own run, follow step 10 above and apply the formulas from the report to your raw values.

## Configuration

`run.py` starts the metrics server and loads the default model before Streamlit accepts traffic, so Prometheus sees the target as up immediately. Optional environment variables:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `METRICS_PORT` | `18000` | Port for the `/metrics` endpoint |
| `PRELOAD_MODEL` | `blip-base` | Model loaded at startup (`blip-base`, `vit-gpt2`, or empty) |
| `TORCH_THREADS` | torch default | CPU threads used for inference |
| `BULK_MAX_IMAGES` | `300` | Max images captioned per ZIP |
| `BULK_MAX_UNCOMPRESSED_MB` | `1024` | Max total uncompressed image bytes per ZIP |
| `MAX_IMAGE_MEGAPIXELS` | `40` | Images above this resolution are rejected |

## Monitoring stack

| Component | Address | Started by |
| --- | --- | --- |
| Caption app | `http://localhost:8501`, metrics on `http://localhost:18000/metrics` | `python run.py` |
| node_exporter | `http://localhost:9100/metrics` | `node_exporter ...` (terminal 1) |
| Prometheus | `http://localhost:9090` | `prometheus --config.file=prometheus.yml ...` (terminal 3) |
| Grafana | `http://localhost:3000` | `docker compose up grafana` (terminal 4) |

- **Scraping** (`prometheus.yml`): every 5s for `caption_app` (`:18000`), `node` (`:9100`) and `prometheus`. Each target carries a `service` label (`caption-app`, `host`, `monitoring`).
- **Recording rules** (`rules/recording_rules.yml`): host CPU/memory/swap/disk/network ratios and the app's p50/p95/p99 inference latency, throughput, error ratio, average request time (from the summary's `_sum`/`_count`), and the app's share of host CPU and RAM, so app spikes can be read directly against host load.
- **Grafana** (`docker-compose.yml`, `grafana/`): Grafana 13.2.4 on host networking, bound to `127.0.0.1:3000`. The Prometheus datasource (`grafana/provisioning/datasources/`) and the dashboard (`grafana/dashboards/caption-app.json`) are provisioned at startup, so no manual setup is needed. Anonymous users get read-only access; log in as `admin` / `admin` to edit.
- Reload Prometheus after editing rules with `curl -X POST localhost:9090/-/reload`.

## Grafana dashboard

`grafana/dashboards/caption-app.json` (*CPU Image Captioning · Service and Host Monitoring*) refreshes every 5s and reads top to bottom from high-level health to low-level resources:

1. **Service health**: app and node_exporter up/down, model loaded, uptime, active sessions, requests in flight, images captioned, image error ratio.
2. **Traffic**: images per second and requests per minute by mode, work in flight (running requests, pending bulk images, CPU-lock queue).
3. **Latency**: inference p50/p95/p99 from the histogram, p95 by mode, average request time from the Summary (`_sum ÷ _count`), latency heatmap, preprocessing and queue-wait p95.
4. **Correlation**: host CPU and RAM against the app's share of them on one axis, with the app's images per second and memory directly underneath on the same time axis.
5. **Host resources**: CPU by mode, load per core, memory breakdown, disk free space, disk and network throughput.
6. **Model and inputs**: model size, load time and threads, caption length, input resolution, tokens per second, upload volume, skipped ZIP entries, per-client requests and errors.

Grey shaded bands mark the time a bulk ZIP job was running, on every time-series panel.

How the plotting commandments are applied:

| Commandment | How |
| --- | --- |
| Colour-blind-safe palette | Fixed series colours from a validated categorical palette (checked for colour-vision-deficiency separation against Grafana's dark surface); ordered quantities (p50/p95/p99, load 1/5/15) use one blue hue from dark to light; status tiles pair colour with ▲/▼/● symbols and text |
| Markers that survive any medium | Every series has point markers and its own line style (solid, dashed, dotted, dash-dot), so charts still read in grayscale print or PDF |
| Named axes | Every y-axis is labelled with the quantity; the x-axis is wall-clock time, stated in each panel description |
| Scale and units | Each axis label ends with `(linear)` or `(log₂)`; every panel sets a unit (img/s, req/min, s, bytes, B/s, %) |
| Legend for more than one variable | Table legends with mean/max/last on every multi-series chart; single-series charts have none |
| Title and subtitle | Titles follow `Title · subtitle`, and each dashboard row is titled with the question it answers |
| Short description | Every panel has a description (ⓘ) telling its story; the header text explains the reading order |

PromQL in the dashboard was generated with AI. Each non-trivial query begins with a `# AI prompt: "..."` comment (PromQL comment syntax) giving the prompt used; the same convention is used in `rules/recording_rules.yml`. See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for the full AI usage disclosure and the hand calculations that check each metric's logic.

## Metrics

All application metrics are prefixed with `caption_`. The default `process_*` and `python_*` collectors are also exported.

| Type | Metric | Labels |
| --- | --- | --- |
| Counter | `caption_requests_total` | `mode`, `model`, `client_ip`, `session_id`, `browser` |
| Counter | `caption_images_processed_total` | `mode`, `model`, `status` |
| Counter | `caption_client_images_total` | `client_ip`, `session_id` |
| Counter | `caption_errors_total` | `mode`, `stage`, `error_type` |
| Counter | `caption_bulk_files_skipped_total` | `reason` |
| Counter | `caption_upload_bytes_total` | `mode` |
| Counter | `caption_tokens_generated_total` | `mode`, `model` |
| Counter | `caption_page_views_total` | `browser` |
| Gauge | `caption_active_requests` | `mode` |
| Gauge | `caption_inference_queue_length` | |
| Gauge | `caption_bulk_images_pending` | |
| Gauge | `caption_active_sessions` | |
| Gauge | `caption_model_loaded` | `model` |
| Gauge | `caption_model_memory_bytes` | `model` |
| Gauge | `caption_model_parameters` | `model` |
| Gauge | `caption_model_load_seconds` | `model` |
| Gauge | `caption_last_inference_seconds` | `mode`, `model` |
| Gauge | `caption_last_bulk_job_images` | |
| Gauge | `caption_last_success_timestamp_seconds` | |
| Gauge | `caption_torch_threads` | |
| Histogram | `caption_inference_seconds` | `mode`, `model` |
| Histogram | `caption_preprocess_seconds` | `mode` |
| Histogram | `caption_queue_wait_seconds` | `mode` |
| Histogram | `caption_bulk_job_seconds` | `model` |
| Histogram | `caption_bulk_job_images` | |
| Histogram | `caption_image_bytes` | `mode` |
| Histogram | `caption_image_megapixels` | `mode` |
| Histogram | `caption_length_words` | `model` |
| Summary | `caption_request_seconds` | `mode`, `model` |
| Summary | `caption_upload_size_bytes` | `mode` |
| Summary | `caption_bulk_throughput_images_per_second` | `model` |
| Enum | `caption_model_state` | `model` |
| Info | `caption_app_build_info` | |
