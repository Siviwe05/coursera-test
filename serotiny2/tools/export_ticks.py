"""
export_ticks.py -- run this ON YOUR MT5 MACHINE to export real bid/ask ticks.

    python -m serotiny2.tools.export_ticks --symbols EURUSD GBPUSD --days 120

Writes one gzip CSV per symbol per day:
    ticks/<BASE>/<YYYY-MM-DD>.csv.gz      columns: time_msc,bid,ask
plus ticks/<BASE>/meta.json with the broker symbol name, digits and an
estimate of the broker server's UTC offset (MT5 stamps ticks in SERVER
time, and sessions like London/NY need that offset to line up).

Resumable: days already exported are skipped, so just re-run it if it
stops. The current (incomplete) day is always re-exported.
Needs: pip install MetaTrader5 pandas
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone


def find_symbol(mt5, base):
    names = [s.name for s in (mt5.symbols_get() or [])]
    exact = [n for n in names if n.upper() == base.upper()]
    if exact:
        return exact[0]
    cands = sorted((n for n in names if base.upper() in n.upper()), key=len)
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD"])
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default="ticks")
    args = ap.parse_args()

    import MetaTrader5 as mt5
    import pandas as pd

    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize() failed: {mt5.last_error()}")

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for base in args.symbols:
        sym = find_symbol(mt5, base)
        if sym is None:
            print(f"[{base}] no matching symbol on this terminal, skipping")
            continue
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
        out_dir = os.path.join(args.out, base.upper())
        os.makedirs(out_dir, exist_ok=True)

        tick = mt5.symbol_info_tick(sym)
        offset_h = None
        if tick is not None and tick.time:
            offset_h = round((tick.time - time.time()) / 3600)   # only meaningful while the market is open
        meta = {"base": base.upper(), "broker_symbol": sym, "digits": info.digits if info else None,
                "point": info.point if info else None, "server_utc_offset_hours_estimate": offset_h,
                "exported_at_utc": datetime.now(timezone.utc).isoformat()}
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        total = 0
        for d in range(args.days, -1, -1):
            day = today - timedelta(days=d)
            path = os.path.join(out_dir, f"{day:%Y-%m-%d}.csv.gz")
            if d > 0 and os.path.exists(path):
                continue
            ticks = mt5.copy_ticks_range(sym, day, day + timedelta(days=1), mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                continue
            df = pd.DataFrame(ticks)[["time_msc", "bid", "ask"]]
            df = df[(df["bid"] > 0) & (df["ask"] > 0)]
            if len(df) == 0:
                continue
            df.to_csv(path, index=False, compression="gzip")
            total += len(df)
            print(f"[{base}] {day:%Y-%m-%d}: {len(df):>8} ticks", flush=True)
        print(f"[{base}] done ({sym}), {total} new ticks -> {out_dir}  | server offset ~ {offset_h}h")

    mt5.shutdown()


if __name__ == "__main__":
    main()
