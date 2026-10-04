from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from queue import Empty


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "EDA_limpio.ipynb"
TIMEOUT_SECONDS = 600


def execute_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    temp_runtime = Path(tempfile.mkdtemp(prefix="eda_limpio_kernel_", dir="/private/tmp"))
    env = os.environ.copy()
    env["IPYTHONDIR"] = str(temp_runtime / "ipython")
    env["MPLCONFIGDIR"] = str(temp_runtime / "matplotlib")
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    km = KernelManager(kernel_name=nb.metadata.kernelspec.name)
    km.start_kernel(cwd=str(ROOT), env=env)
    kc = km.blocking_client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)

    try:
        execution_count = 0
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        total = len(code_cells)

        for position, cell in enumerate(code_cells, start=1):
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            msg_id = kc.execute(cell.source, allow_stdin=False, stop_on_error=True)
            failed = False

            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=TIMEOUT_SECONDS)
                except Empty as exc:
                    raise TimeoutError(f"Timeout en celda de código {position}/{total}") from exc

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue

                msg_type = msg["header"]["msg_type"]
                content = msg["content"]

                if msg_type == "status" and content.get("execution_state") == "idle":
                    break
                if msg_type == "execute_input":
                    cell.execution_count = content.get("execution_count", execution_count)
                elif msg_type == "stream":
                    cell.outputs.append(nbformat.v4.new_output(
                        "stream", name=content.get("name", "stdout"), text=content.get("text", "")
                    ))
                elif msg_type in {"display_data", "execute_result"}:
                    output_type = "execute_result" if msg_type == "execute_result" else "display_data"
                    kwargs = {
                        "output_type": output_type,
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                    if output_type == "execute_result":
                        kwargs["execution_count"] = content.get("execution_count", cell.execution_count)
                    cell.outputs.append(nbformat.v4.new_output(**kwargs))
                elif msg_type == "error":
                    failed = True
                    cell.outputs.append(nbformat.v4.new_output(
                        "error",
                        ename=content.get("ename", "Error"),
                        evalue=content.get("evalue", ""),
                        traceback=content.get("traceback", []),
                    ))
                elif msg_type == "clear_output":
                    cell.outputs = []

            reply = kc.get_shell_msg(timeout=TIMEOUT_SECONDS)
            if reply.get("content", {}).get("status") == "error":
                failed = True

            print(f"[{position:02d}/{total:02d}] {'ERROR' if failed else 'OK'}", flush=True)
            if failed:
                nbformat.write(nb, path)
                raise RuntimeError(f"La ejecución falló en la celda de código {position}/{total}")

        nb.metadata["execution"] = {
            "status": "completed",
            "code_cells": total,
        }
        nbformat.write(nb, path)
    finally:
        try:
            kc.stop_channels()
        finally:
            km.shutdown_kernel(now=True)


if __name__ == "__main__":
    execute_notebook(NOTEBOOK)
    print(f"Notebook ejecutado y guardado: {NOTEBOOK}")
