#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_GLOB = "2wikimqa/*.txt"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


@dataclass
class ServerHandle:
    process: subprocess.Popen[str]
    log_file: Any
    slot_save_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark segmented 1k-token prompt relay on llama-server. "
            "Each round appends a new input chunk, decodes a fixed number of tokens, "
            "feeds the generated tokens back into the running context, and records "
            "prefill/decode speed plus KV cache occupancy."
        )
    )
    parser.add_argument(
        "--prompt-files",
        nargs="*",
        default=None,
        help=(
            "Prompt files to benchmark. Defaults to bench-decoding/2wikimqa/*.txt "
            "relative to this script."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"llama-server base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--server-command",
        default="",
        help=(
            "Optional shell command to start llama-server. The script appends "
            "--slot-save-path automatically. If omitted, an already running server is used."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for llama-server readiness (default: 180)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for a single HTTP request (default: 600)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Input chunk size in tokens for each relay round (default: 1000)",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="Number of tokens to decode after each input chunk (default: 128)",
    )
    parser.add_argument(
        "--slot-id",
        type=int,
        default=0,
        help="Server slot id to reuse across all rounds (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Sampling seed for deterministic runs (default: 123)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="top-k for generation (default: 1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top-p for generation (default: 1.0)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="min-p for generation (default: 0.0)",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Label written into the result rows, e.g. baseline or snapkv (default: run)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for CSV/JSON results. Defaults to bench-decoding/results/<timestamp>",
    )
    parser.add_argument(
        "--keep-slot-dumps",
        action="store_true",
        help="Keep per-round slot save binaries instead of deleting them after measurement",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw HTTP responses and server startup command details",
    )
    return parser.parse_args()


def resolve_prompt_files(args: argparse.Namespace) -> list[Path]:
    if args.prompt_files:
        files = [Path(path).resolve() for path in args.prompt_files]
    else:
        files = sorted((SCRIPT_DIR / DEFAULT_PROMPT_GLOB.split("/")[0]).glob(DEFAULT_PROMPT_GLOB.split("/", 1)[1]))
        files = [path.resolve() for path in files]

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"prompt files not found: {', '.join(missing)}")
    if not files:
        raise SystemExit("no prompt files found")
    return files


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = (SCRIPT_DIR / "results" / f"{args.label}-{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def http_json(method: str, url: str, payload: Any | None, timeout_s: float) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return None
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"request failed for {method} {url}: {exc}") from exc


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            http_json("GET", join_url(base_url, "/slots"), None, timeout_s=5.0)
            return
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"llama-server did not become ready within {timeout_s:.0f}s: {last_error}")


def start_server(args: argparse.Namespace, output_dir: Path) -> ServerHandle | None:
    if not args.server_command:
        return None

    slot_save_path = output_dir / "slot-dumps"
    slot_save_path.mkdir(parents=True, exist_ok=True)

    command = args.server_command.strip()
    if "--slot-save-path" not in command:
        command += f" --slot-save-path {shlex.quote(str(slot_save_path))}"
    if "--slots" not in command and "--no-slots" not in command:
        command += " --slots"

    log_path = output_dir / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    if args.verbose:
        print(f"[server] starting: {command}", file=sys.stderr)
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
        text=True,
    )
    return ServerHandle(process=process, log_file=log_file, slot_save_path=slot_save_path)


def stop_server(handle: ServerHandle | None) -> None:
    if handle is None:
        return
    try:
        if handle.process.poll() is None:
            os.killpg(os.getpgid(handle.process.pid), signal.SIGTERM)
            try:
                handle.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
                handle.process.wait(timeout=5)
    finally:
        handle.log_file.close()


def get_slot_entry(base_url: str, slot_id: int, timeout_s: float) -> dict[str, Any]:
    slots = http_json("GET", join_url(base_url, "/slots"), None, timeout_s)
    if not isinstance(slots, list):
        raise RuntimeError(f"unexpected /slots response: {slots!r}")
    for entry in slots:
        if entry.get("id") == slot_id:
            return entry
    raise RuntimeError(f"slot {slot_id} not found in /slots response")


def erase_slot(base_url: str, slot_id: int, timeout_s: float) -> None:
    url = join_url(base_url, f"/slots/{slot_id}?action=erase")
    http_json("POST", url, {}, timeout_s)


def tokenize_text(base_url: str, text: str, timeout_s: float) -> list[int]:
    payload = {
        "content": text,
        "add_special": True,
        "parse_special": True,
        "with_pieces": False,
    }
    data = http_json("POST", join_url(base_url, "/tokenize"), payload, timeout_s)
    tokens = data.get("tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return tokens


def chunk_boundaries(total_tokens: int, chunk_size: int) -> list[int]:
    boundaries = []
    current = min(chunk_size, total_tokens)
    while current < total_tokens:
        boundaries.append(current)
        current += chunk_size
    boundaries.append(total_tokens)
    return boundaries


def save_slot_snapshot(
    base_url: str,
    slot_id: int,
    filename: str,
    timeout_s: float,
    keep_file: bool,
    slot_save_path: Path | None,
) -> dict[str, Any]:
    payload = {"filename": filename}
    data = http_json("POST", join_url(base_url, f"/slots/{slot_id}?action=save"), payload, timeout_s)
    if slot_save_path is not None and not keep_file:
        dump_path = slot_save_path / filename
        if dump_path.exists():
            dump_path.unlink()
    return data


def round_request_payload(
    prompt_tokens: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "prompt": prompt_tokens,
        "id_slot": args.slot_id,
        "cache_prompt": True,
        "stream": False,
        "return_tokens": True,
        "timings_per_token": False,
        "n_predict": args.decode_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
    }


def benchmark_case(
    prompt_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    slot_save_path: Path | None,
) -> list[dict[str, Any]]:
    case_name = prompt_path.stem
    text = prompt_path.read_text(encoding="utf-8")
    source_tokens = tokenize_text(args.base_url, text, args.request_timeout)
    boundaries = chunk_boundaries(len(source_tokens), args.chunk_size)

    print(
        f"[case] {case_name}: {len(source_tokens)} source tokens, "
        f"{len(boundaries)} relay rounds",
        file=sys.stderr,
    )

    erase_slot(args.base_url, args.slot_id, args.request_timeout)

    rows: list[dict[str, Any]] = []
    context_tokens: list[int] = []
    previous_input_end = 0

    for round_index, input_end in enumerate(boundaries, start=1):
        new_chunk = source_tokens[previous_input_end:input_end]
        previous_context_tokens = len(context_tokens)
        context_tokens.extend(new_chunk)
        prompt_tokens = list(context_tokens)

        response = http_json(
            "POST",
            join_url(args.base_url, "/completion"),
            round_request_payload(prompt_tokens, args),
            args.request_timeout,
        )

        generated_tokens = response.get("tokens") or []
        if not isinstance(generated_tokens, list) or not all(isinstance(token, int) for token in generated_tokens):
            raise RuntimeError(f"unexpected completion tokens for {case_name} round {round_index}: {response!r}")
        context_tokens.extend(generated_tokens)

        slot_dump_name = f"{case_name}-round-{round_index:02d}.bin"
        slot_save = save_slot_snapshot(
            args.base_url,
            args.slot_id,
            slot_dump_name,
            args.request_timeout,
            args.keep_slot_dumps,
            slot_save_path,
        )
        slot_entry = get_slot_entry(args.base_url, args.slot_id, args.request_timeout)
        next_token = slot_entry.get("next_token") or {}
        if isinstance(next_token, list):
            next_token = next_token[0] if next_token and isinstance(next_token[0], dict) else {}
        elif not isinstance(next_token, dict):
            next_token = {}

        timings = response.get("timings") or {}
        row = {
            "label": args.label,
            "case_name": case_name,
            "prompt_file": str(prompt_path),
            "round_index": round_index,
            "source_tokens_total": len(source_tokens),
            "source_tokens_used": input_end,
            "new_input_tokens": len(new_chunk),
            "context_tokens_before_round": previous_context_tokens,
            "prompt_tokens_this_round": len(prompt_tokens),
            "generated_tokens": len(generated_tokens),
            "context_tokens_after_round": len(context_tokens),
            "tokens_evaluated": response.get("tokens_evaluated"),
            "tokens_cached": response.get("tokens_cached"),
            "tokens_uncached": None,
            "truncated": response.get("truncated"),
            "content": response.get("content", ""),
            "content_preview": (response.get("content", "") or "")[:200],
            "generated_token_ids": generated_tokens,
            "prompt_per_second": timings.get("prompt_per_second"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_n": timings.get("prompt_n"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "predicted_ms": timings.get("predicted_ms"),
            "predicted_n": timings.get("predicted_n"),
            "cache_n": timings.get("cache_n"),
            "kv_saved_tokens": slot_save.get("n_saved"),
            "kv_saved_bytes": slot_save.get("n_written"),
            "kv_save_ms": (slot_save.get("timings") or {}).get("save_ms"),
            "slot_n_ctx": slot_entry.get("n_ctx"),
            "slot_is_processing": slot_entry.get("is_processing"),
            "slot_last_task": slot_entry.get("id_task"),
            "slot_next_decoded": next_token.get("n_decoded"),
        }

        if row["tokens_evaluated"] is not None and row["tokens_cached"] is not None:
            row["tokens_uncached"] = row["tokens_evaluated"] - row["tokens_cached"]

        rows.append(row)

        print(
            "[round] {case} #{idx:02d} input={inp} cached={cached} uncached={uncached} "
            "pp={pp:.2f} t/s tg={tg:.2f} t/s kv_saved={saved} toks {bytes_} bytes".format(
                case=case_name,
                idx=round_index,
                inp=row["prompt_tokens_this_round"],
                cached=row["tokens_cached"],
                uncached=row["tokens_uncached"],
                pp=float(row["prompt_per_second"] or 0.0),
                tg=float(row["predicted_per_second"] or 0.0),
                saved=row["kv_saved_tokens"],
                bytes_=row["kv_saved_bytes"],
            ),
            file=sys.stderr,
        )

        previous_input_end = input_end

    erase_slot(args.base_url, args.slot_id, args.request_timeout)
    return rows


def write_results(output_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    json_path.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "label",
        "case_name",
        "prompt_file",
        "round_index",
        "source_tokens_total",
        "source_tokens_used",
        "new_input_tokens",
        "context_tokens_before_round",
        "prompt_tokens_this_round",
        "generated_tokens",
        "context_tokens_after_round",
        "tokens_evaluated",
        "tokens_cached",
        "tokens_uncached",
        "truncated",
        "prompt_per_second",
        "prompt_ms",
        "prompt_n",
        "predicted_per_second",
        "predicted_ms",
        "predicted_n",
        "cache_n",
        "kv_saved_tokens",
        "kv_saved_bytes",
        "kv_save_ms",
        "slot_n_ctx",
        "slot_is_processing",
        "slot_last_task",
        "slot_next_decoded",
        "content_preview",
        "content",
        "generated_token_ids",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["generated_token_ids"] = json.dumps(csv_row["generated_token_ids"], ensure_ascii=False)
            writer.writerow(csv_row)


def main() -> int:
    args = parse_args()
    prompt_files = resolve_prompt_files(args)
    output_dir = make_output_dir(args)

    handle = start_server(args, output_dir)
    try:
        wait_for_server(args.base_url, args.startup_timeout)

        slot_save_path = handle.slot_save_path if handle is not None else None
        all_rows: list[dict[str, Any]] = []

        for prompt_path in prompt_files:
            all_rows.extend(benchmark_case(prompt_path, args, output_dir, slot_save_path))

        meta = {
            "label": args.label,
            "base_url": args.base_url,
            "chunk_size": args.chunk_size,
            "decode_tokens": args.decode_tokens,
            "slot_id": args.slot_id,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "prompt_files": [str(path) for path in prompt_files],
            "server_command": args.server_command,
            "output_dir": str(output_dir),
        }
        write_results(output_dir, all_rows, meta)

        if handle is not None and not args.keep_slot_dumps:
            shutil.rmtree(handle.slot_save_path, ignore_errors=True)

        print(f"[done] wrote {len(all_rows)} rows to {output_dir}", file=sys.stderr)
        return 0
    finally:
        stop_server(handle)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)