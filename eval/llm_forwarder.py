#!/usr/bin/env python3
"""Minimal TCP forwarder used to expose a local model server on port 80.

The pier egress proxy only allows ports 80 and 443, so a model served on an
arbitrary port (for example LiteLLM on 4000 or vLLM on 8005) is unreachable from
task containers. This forwarder listens on the docker bridge gateway and relays
to the real endpoint on the host loopback.

Run it through `deepswe_eval.py forwarder start`, which starts a container on
the host network so it can both bind 172.17.0.1:80 and reach 127.0.0.1:4000.
"""

from __future__ import annotations

import argparse
import asyncio


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    try:
        server_reader, server_writer = await asyncio.open_connection(
            target_host, target_port
        )
    except OSError:
        client_writer.close()
        return
    await asyncio.gather(
        pipe(client_reader, server_writer),
        pipe(server_reader, client_writer),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="172.17.0.1:80")
    parser.add_argument("--target", default="127.0.0.1:4000")
    args = parser.parse_args()

    listen_host, listen_port = args.listen.rsplit(":", 1)
    target_host, target_port = args.target.rsplit(":", 1)

    server = await asyncio.start_server(
        lambda r, w: handle(r, w, target_host, int(target_port)),
        listen_host,
        int(listen_port),
    )
    print(f"forwarding {args.listen} -> {args.target}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
