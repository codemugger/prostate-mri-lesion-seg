# Prostate MRI pipeline on CARPL (DICOM Push)

This folder packages the unchanged prostate MRI lesion segmentation pipeline
as a CARPL **DICOM Push** algorithm (CARPL DICOM Push Integration Guide v1.2).
The architecture copies the CXR-BMD integration already in production on
the SGH RAPIER host (`carpl-compatibility/`), adapted for multi-series MR
studies and a GPU pipeline.

```
CARPL ──C-STORE──▶ prostate-dicom-listener ──▶ prostate-redis ──▶ prostate-ai-worker (GPU)
 (AE PROSTATE_AI :11112, rapier network)       (job queue)        runs the unchanged app:
                                                                  python /app/prostate_mri_lesion_seg_app
                                                                    -i <job>/input -o <job>/output -m /models
CARPL ◀──C-STORE── prostate-dicom-sender ◀──────────────────────── results queued after audit gates
 (CARPL-RECIEVE@rapier-orthanc-prefect:4242)
```

## What CARPL receives

For each study the pipeline completes, two existing pipeline outputs are
sent **byte-for-byte unmodified**:

| File | DICOM object | Content |
|---|---|---|
| `combined_organ_lesion_RTSTRUCT.dcm` | RT Structure Set (Explicit VR LE) | The UroFusion-accepted canonical profile: ROI 1 `Prostate`, ROIs 2…N `Lesion N - AI PI-RADS X`, `CLOSED_PLANAR`, `UNAPPROVED` |
| `prostate_measurements_SR.dcm` | Comprehensive SR, TID 1500 | Prostate volume and LR/AP/SI dimensions, `COMPLETE` / `UNVERIFIED` |

Both objects keep the received StudyInstanceUID and have new Series and SOP
Instance UIDs. CARPL must match results to studies by **StudyInstanceUID**:
the SR truncates AccessionNumber to 16 characters and the RTSTRUCT has an
empty SeriesNumber.

A study in which no prostate is detected still returns the truthful
zero-ROI RTSTRUCT plus the SR (`SEND_EMPTY_RESULTS=true`). A study that fails
any audit gate returns **nothing**. It is recorded in the audit log and kept
for review, see [Failures](#failures).

## Inference is unchanged

- **Same image.** `prostate-carpl-ai-worker` is built `FROM` the validated
  inference image `prostate-mri-seg:<version>`, which is built from the
  unchanged `docker/Dockerfile`. `scripts/build_carpl_images.py` refuses to
  package unless:
  - the base image's layers are an exact prefix of the worker's layers;
  - `pip freeze` is identical in both images.
  - site-packages differs only by holoscan's wheel-axle finalisation: its
    `libholoscan_*.so.2` symlinks. Root runs of the base image, as on HIVE,
    create these at startup; the worker creates them at build time so it can
    run as a non-root user.
- **Same command.** The worker runs the app with exactly the command of that
  image's ENTRYPOINT, one subprocess per study.
- **Same audit.** Success is judged by the same `scripts/case_status.py`
  audit that HIVE batches use. The exit code alone is not trusted.
- **Byte-identical transport.** Received instances and returned results are
  never decoded or re-encoded. `carpl/tools/parity_check.py transport`
  proves this.

## Components

| Service / container | Image | Networks | Purpose |
|---|---|---|---|
| `prostate-dicom-listener` | `prostate-carpl-dicom-listener:<v>` | internal + `rapier-carpl_docker_backend` | Storage SCP; study-level triggering; dedup |
| `prostate-redis` | `redis:7.4-alpine` | internal only | Job and send queues (AOF persistence) |
| `prostate-ai-worker` | `prostate-carpl-ai-worker:<v>` | internal only | Runs the pipeline on the GPU; audit; send gates |
| `prostate-dicom-sender` | `prostate-carpl-dicom-sender:<v>` | internal + `rapier-carpl_docker_backend` | Storage SCU to CARPL; retries; dead-letter |
| `prostate-carpl-receiver` (dev only) | `prostate-carpl-receiver:<v>` | — | Local stand-in for CARPL's receiver |

The names are prefixed with `prostate-` so they cannot collide with the
CXR-BMD stack (`cxrbmd-*`, `dicom-listener:latest`) on the shared network.

## CARPL algorithm registration

Paste this into CARPL Platform → Algorithm Configuration. It follows the
CXR-BMD format: `ip` is the listener's container name on
`rapier-carpl_docker_backend`.

```json
{
  "ip": "prostate-dicom-listener",
  "port": "11112",
  "ae_title": "PROSTATE_AI",
  "modality": "MR",
  "algo_type": "dicom-push",
  "use_algo_specific_postprocessor": false
}
```

**Confirm with the CARPL team before go-live:**
1. The registration may omit `private_findings`, because our outputs are
   RT-Struct and SR rather than the private-tag JSON used by CXR-BMD.
2. `orthanc-prefect` accepts two result objects per study and links them by
   StudyInstanceUID.
3. Whether CARPL sends the whole MR study or one pre-filtered T2 / ADC /
   high-b series each. Both work. Pre-filtering removes series-selection
   ambiguity, see [Series selection](#series-selection).
4. The GPU on the RAPIER host can be reserved for `prostate-ai-worker`.
5. The AE title `PROSTATE_AI` is acceptable.

## 1. Build and package (internet-connected machine)

```bash
python scripts/build_carpl_images.py --version 1.0.0 --build-base
```

This builds the base inference image from `docker/Dockerfile`, then builds
the four CARPL images. It then checks that the layers and `pip freeze` are
identical, that the SDK is version 2.0.0, and that the worker imports load.
It also runs `tests/test_carpl_*.py` inside the listener image.

Finally it writes the deploy bundle:

```
dist/prostate-carpl-1.0.0/
├── images.tar                # base, worker, listener, sender, redis:7.4-alpine
├── images.tar.sha256
├── docker-compose-carpl.yml
├── .env.example
├── README.md                 # this file
└── build_manifest.json       # git commit, image IDs, SDK version, tar checksum
```

To reuse an inference image that has already been validated (for example
on HIVE), pass `--base-image prostate-mri-seg:<tag>` instead of `--build-base`.

## 2. Deploy on the RAPIER host

This follows the same steps as the CXR-BMD "change versions & re-deploy"
procedure. The example directory is `/home/<user>/prostate-carpl`.

```bash
cd /home/<user>/prostate-carpl          # bundle copied here
sha256sum -c images.tar.sha256
docker load -i images.tar

cp .env.example .env                    # set VERSION, PUID/PGID (id -u / id -g)
mkdir -p data/redis config models
```

Place **exactly** these seven files under `models/`, with nothing else, not
even a README:

```
models/organ/model.ts
models/fold0/model_best_fold0.pth.tar … models/fold4/model_best_fold4.pth.tar
models/classifier/model_best.pth.tar
```

The MONAI Deploy model loader rejects any extra file in the models root. The
worker checks this at startup and refuses to run if the layout is wrong.

If the models came from `scripts/package_models.py`, move its
`deployment_manifest.json` to `config/`. The worker then verifies every
checksum; set `REQUIRE_MODEL_MANIFEST=true` to make the manifest mandatory.

Start the stack:

```bash
docker network ls | grep rapier-carpl_docker_backend
docker compose -f docker-compose-carpl.yml up -d
docker compose -f docker-compose-carpl.yml ps          # all services "healthy"
docker logs --tail 50 prostate-ai-worker               # "Preflight passed"
docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py status
```

Then ask CARPL to run the checks from guide §8:
1. C-ECHO to `PROSTATE_AI@prostate-dicom-listener:11112`.
2. A test study push.
3. The RTSTRUCT and SR appear against the study in the CARPL viewer.

### Upgrade and rollback

```bash
docker load -i images.tar                      # new version, e.g. 1.0.1
sed -i 's/^VERSION=.*/VERSION=1.0.1/' .env
docker compose -f docker-compose-carpl.yml up -d --force-recreate
docker inspect -f '{{.Name}} {{.Config.Image}}' prostate-ai-worker prostate-dicom-listener prostate-dicom-sender
```

To roll back, set `VERSION` to the previous tag and run the same command.
`data/` is a bind mount, so queued work and the audit log survive container
re-creation.

## 3. Local end-to-end test (laptop with GPU)

```bash
cd carpl
mkdir -p data/redis config
VERSION=1.0.0 docker compose up -d --build     # needs prostate-mri-seg:1.0.0
python -m venv .venv && .venv/bin/pip install -r requirements-carpl.txt
.venv/bin/python tools/send_study.py ../test-data/ProstateX-0004 --port 11112
docker logs -f prostate-dev-ai-worker
ls data/test_carpl/*/                          # RTSTRUCT + SR received by the stub
docker exec prostate-dev-dicom-sender python /opt/carpl/carpl_ctl.py jobs
VERSION=1.0.0 docker compose down
```

To prove the CARPL path gives the same outputs as a direct run, run the base
image directly on the job's input folder, then compare the two outputs:

```bash
JOB=<job_id>
docker run --rm --gpus all --shm-size=1g \
  -v "$PWD/data/jobs/$JOB/input":/input:ro -v "$PWD/direct":/output -v "$PWD/../models":/models:ro \
  prostate-mri-seg:1.0.0
docker run --rm --entrypoint python -v "$PWD/..":/repo:ro -v "$PWD":/work:ro \
  prostate-carpl-ai-worker:1.0.0 /repo/carpl/tools/parity_check.py \
  outputs /work/direct /work/data/jobs/$JOB/output
```

## How studies are handled

### Study completeness (CARPL guide §4)

- **Storage.** Instances are stored as received, under
  `data/studies/<StudyUID>/<SeriesUID>/<SOPUID>.dcm`. This is the
  one-series-per-folder layout the pipeline requires.
- **Only MR is stored.** Other modalities are acknowledged but not stored
  (`ACCEPT_MODALITIES=MR`). This also means our own results are ignored if
  CARPL pushes them back.
- **When a job starts.** A study becomes a job only after it has had no open
  association and no new instance for `STUDY_QUIET_SECONDS` (default 60 s).
  It never starts on the first image.
- **Job input.** The job gets a hardlinked snapshot in
  `data/jobs/<job_id>/input/`, which always holds exactly one study.
- **Timeouts.** Network, ACSE and DIMSE timeouts abort a hung association,
  after which the quiet period applies.

### Deduplication and retransmission

- **Duplicate instances.** A duplicate SOP instance overwrites the same file.
- **Identical resend.** An identical resend of a study (same SOP set) is
  skipped while the earlier job is queued, running or delivered. If the
  earlier job failed, the resend is treated as a retry.
- **Late or extra instances.** These create a new job over the full set. The
  older job is marked `superseded` and sends nothing, even if it had already
  finished inference.
- **Crash recovery.** A job recovered after inference only re-queues its send
  and never re-runs the pipeline. Re-running would mint new SOP UIDs and send
  CARPL a second, different result set.

### Series selection

The pipeline keeps its existing selection behaviour: first match, in the
order the files are listed, for each of the T2/ADC/HIGHB rules in `app.py`.
When more than one series matches a rule, the worker records every candidate
and the series actually used in `data/jobs/<id>/selection.json` and the
event log. In the SGH annotation, 27.8% of studies have more than one
matching high-b series, e.g. `Ax FOCUS b1800` and
`Ax FOCUS b50/500/1000 Syn b1800`.

- `AMBIGUOUS_SELECTION_POLICY=send` (the default) returns results as usual.
- `hold` keeps those studies for manual review instead.

### Send gates

Results are queued for CARPL only if **all** of these hold:
- the `case_status` status is `SUCCESS_COMPLETE` or `SUCCESS_EMPTY_LESION`;
- both files exist;
- the combined RTSTRUCT matches the accepted `AI_PROSTATE` profile;
- the SOP classes are RT Structure Set and Comprehensive SR;
- the StudyInstanceUID equals the received study's;
- the PatientID equals the selected T2 series';
- the Series UIDs are new;
- the job has not been superseded.

### Failures

The following cases send nothing to CARPL:
- `FAIL_*` statuses, e.g. `FAIL_MISSING_MODALITY` when the study has no
  matching T2, ADC or high-b series;
- timeouts (`JOB_TIMEOUT_SECONDS`, default 900 s);
- any failed send gate.

Each case is recorded in the audit log and the job folder is kept for
`FAILED_RETENTION_DAYS`. Result sends retry with exponential backoff up to
`SEND_MAX_ATTEMPTS`, then move to the `prostate:send:failed` list. After the
cause is fixed, requeue with:

```bash
docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py requeue --queue send --all
docker exec prostate-dicom-sender python /opt/carpl/carpl_ctl.py requeue --queue jobs --job-id <id>
```

## Operations

| Where | What |
|---|---|
| `docker logs prostate-<service>` | Service logs (stdout, rotated at 20 MB × 5) |
| `data/audit/carpl_audit_log.csv` | One row per pipeline run. Same schema as HIVE `batch_audit_log.csv`; `patient_id` holds the job id |
| `data/audit/events.jsonl` | Lifecycle events: job queued or duplicate skipped, started, decision, delivered or failed, purged |
| `data/jobs/<id>/` | `job.json` (series received), `state.json` (history), `selection.json`, `pipeline.log`, `output/` |
| `carpl_ctl.py status` / `jobs` | Queue depths, heartbeats, job states |

### Patient data retention

`data/` contains patient DICOM. The services purge it automatically:

| Data | Purged |
|---|---|
| Received study stores | 6 h after last activity (`STUDY_RETENTION_HOURS`) |
| Delivered jobs | after 24 h (`DELIVERED_RETENTION_HOURS`) |
| Failed or held jobs | after 7 days (`FAILED_RETENTION_DAYS`) |

The audit CSV and event log contain only technical identifiers: job ids,
UIDs, AE titles and statuses.

All services run as `PUID:PGID`, so no root-owned patient files accumulate
on the host.

## Configuration

Every setting, with its default and meaning, is in [`.env.example`](.env.example).

## Troubleshooting

| Symptom | Check |
|---|---|
| Worker restarts every ~60 s | `docker logs prostate-ai-worker`: preflight errors, e.g. a models layout problem, a manifest mismatch, or no GPU |
| CARPL association rejected | The called AE must be `PROSTATE_AI` (`REQUIRE_CALLED_AET=true`). Also check `ALLOWED_CALLING_AES` |
| Study received but no job | Wait `STUDY_QUIET_SECONDS` after the last instance. `events.jsonl` shows `duplicate_skipped` for identical resends |
| Job `failed` with `FAIL_MISSING_MODALITY` | Check the series in `job.json` against the T2/ADC/HIGHB rules. The study usually lacks a matching series |
| Sends stuck in `send_failed` | CARPL receiver unreachable: check `DEST_*`, and `docker network inspect rapier-carpl_docker_backend` |
| `Permission denied` under `data/` | Create `data/` as the `PUID` user before `docker compose up` |
