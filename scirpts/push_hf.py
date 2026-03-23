import base64
import hashlib
import json
import os
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "artimes/artimes-yolov8n-260323-1629"
OUTPUT_DIR = ROOT / "model-bin" / "artimes-yolov8n-260323-1555"
COMMIT_MESSAGE = "Upload YOLOv8CenterPoint model"
PRIVATE = False
BRANCH = "main"
LFS_MULTIPART_CHUNK_SIZE = 64 * 1024 * 1024
RETRY_STATUS_CODES = {502, 503, 504}
MAX_RETRIES = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_summary(response: requests.Response) -> str:
    text = response.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."
    return f"{response.status_code} {response.reason}: {text or '<empty body>'}"


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(_response_summary(response), response=response) from exc


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    retry_label: str,
    **kwargs,
) -> requests.Response:
    last_response = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.request(method, url, **kwargs)
        last_response = response
        if response.status_code not in RETRY_STATUS_CODES:
            return response
        if attempt == MAX_RETRIES:
            return response
        wait_seconds = attempt
        print(
            f"{retry_label} failed with {_response_summary(response)}; "
            f"retrying in {wait_seconds}s ({attempt}/{MAX_RETRIES})"
        )
        time.sleep(wait_seconds)
    return last_response


def upload_lfs_file(
    session: requests.Session,
    endpoint: str,
    repo_id: str,
    file_path: Path,
    sha256: str,
) -> dict:
    response = session.post(
        f"{endpoint}/{repo_id}.git/info/lfs/objects/batch",
        json={
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{"oid": sha256, "size": file_path.stat().st_size}],
            "hash_algo": "sha256",
            "is_browser": True,
        },
        timeout=60,
    )
    raise_for_status_with_body(response)
    obj = response.json()["objects"][0]
    if obj.get("error"):
        raise RuntimeError(f"LFS batch error for {file_path.name}: {obj['error']}")

    upload = obj.get("actions", {}).get("upload")
    if not upload:
        return {"oid": sha256, "size": file_path.stat().st_size}

    headers = upload.get("header", {})
    if "chunk_size" in headers:
        chunk_size = int(headers["chunk_size"])
        part_urls = {int(k): v for k, v in headers.items() if k.isdigit()}
        parts = []
        with file_path.open("rb") as handle:
            part_number = 1
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                put_response = requests.put(part_urls[part_number], data=chunk, timeout=300)
                raise_for_status_with_body(put_response)
                etag = put_response.headers.get("ETag")
                if not etag:
                    raise RuntimeError(f"Missing ETag for multipart upload part {part_number}")
                parts.append({"PartNumber": part_number, "ETag": etag.strip('"')})
                part_number += 1

        complete_response = requests.post(
            upload["href"],
            json={"oid": sha256, "size": file_path.stat().st_size, "parts": parts},
            timeout=300,
        )
        raise_for_status_with_body(complete_response)
    else:
        with file_path.open("rb") as handle:
            put_response = requests.put(upload["href"], data=handle, headers=headers, timeout=300)
            raise_for_status_with_body(put_response)

    verify = obj.get("actions", {}).get("verify")
    if verify:
        verify_response = requests.post(
            verify["href"],
            json={"oid": sha256, "size": file_path.stat().st_size},
            timeout=60,
        )
        raise_for_status_with_body(verify_response)

    return {"oid": sha256, "size": file_path.stat().st_size}


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT")
    if not token:
        raise RuntimeError("Missing `HF_TOKEN` in environment.")
    if not endpoint:
        raise RuntimeError("Missing `HF_ENDPOINT` in environment.")
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output directory does not exist: {OUTPUT_DIR}")

    endpoint = endpoint.rstrip("/")
    namespace, name = REPO_ID.split("/", 1)

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    user_response = request_with_retry(
        session,
        "GET",
        f"{endpoint}/api/whoami-v2",
        retry_label="whoami",
        timeout=30,
    )
    raise_for_status_with_body(user_response)
    user = user_response.json()

    repo_response = request_with_retry(
        session,
        "GET",
        f"{endpoint}/api/models/{namespace}/{name}",
        retry_label="repo lookup",
        timeout=30,
    )
    if repo_response.status_code == 404:
        create_response = request_with_retry(
            session,
            "POST",
            f"{endpoint}/api/repos/create",
            json={
                "type": "model",
                "name": name,
                "organization": namespace,
                "private": PRIVATE,
            },
            retry_label="repo create",
            timeout=30,
        )
        raise_for_status_with_body(create_response)
    else:
        raise_for_status_with_body(repo_response)

    file_paths = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
    file_records = []
    for path in file_paths:
        file_records.append(
            {
                "path": path.relative_to(OUTPUT_DIR).as_posix(),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "file_path": path,
            }
        )

    preupload_response = request_with_retry(
        session,
        "POST",
        f"{endpoint}/api/models/{namespace}/{name}/preupload/{BRANCH}",
        json={
            "files": [
                {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
                for item in file_records
            ]
        },
        retry_label="preupload",
        timeout=60,
    )
    raise_for_status_with_body(preupload_response)
    preupload_map = {item["path"]: item for item in preupload_response.json()["files"]}

    commit_lines = [
        json.dumps(
            {
                "key": "header",
                "value": {"summary": COMMIT_MESSAGE, "description": ""},
            }
        )
    ]

    for item in file_records:
        preupload = preupload_map[item["path"]]
        if preupload.get("shouldIgnore"):
            continue

        if preupload["uploadMode"] == "lfs":
            lfs_info = upload_lfs_file(session, endpoint, REPO_ID, item["file_path"], item["sha256"])
            commit_lines.append(
                json.dumps(
                    {
                        "key": "lfsFile",
                        "value": {
                            "path": item["path"],
                            "oid": lfs_info["oid"],
                            "size": lfs_info["size"],
                            "algo": "sha256",
                        },
                    }
                )
            )
            continue

        content = base64.b64encode(item["file_path"].read_bytes()).decode("ascii")
        commit_lines.append(
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": item["path"],
                        "content": content,
                        "encoding": "base64",
                    },
                }
            )
        )

    commit_response = request_with_retry(
        session,
        "POST",
        f"{endpoint}/api/models/{namespace}/{name}/commit/{BRANCH}",
        data="\n".join(commit_lines),
        headers={"Content-Type": "application/x-ndjson"},
        retry_label="commit",
        timeout=300,
    )
    raise_for_status_with_body(commit_response)
    commit = commit_response.json()

    print(f"repo={endpoint}/{REPO_ID}")
    print(f"user={user.get('name')}")
    print(f"uploaded_from={OUTPUT_DIR}")
    print(f"commit={commit.get('commitOid') or commit.get('oid') or commit}")


if __name__ == "__main__":
    main()
