"""
Multi-Stock Growth Strategy — Trailing Stop Ratchet + Laddering
====================================================================

GOAL: Grow starting capital as quickly as reasonably possible while
strictly capping how much any single trade can cost you.

CORE RISK RULES (the actual "loss limiting" mechanism):
  - Risk per trade: 2% of total account equity, max.
    (Position size is DERIVED from this, not picked arbitrarily --
     see calculate_position_size() below.)
  - Max concurrent positions: defined by TICKERS list length (default 3).
  - Cash reserve: at least RESERVE_CASH_PCT of total equity is always
    kept uninvested, specifically so laddering buys have real capital
    to work with when they trigger.
  - Cash-only: never uses margin. Every buy (initial or laddered) is
    checked against actual account cash before executing.

STRATEGY MECHANICS (same core logic as the original single-stock bot,
now running across multiple tickers with a shared cash pool):
  - Initial floor: sell everything if price drops 10% from entry
  - Ratchet: once up 10% from entry, floor tightens to 5% below the
    highest price seen; every additional 5% climb ratchets it up again
  - Laddering: buy more (same $ amount) if price drops 10% from entry;
    buy more again (2x that $ amount) if price drops 20% from entry

HOW TO USE:
  1. pip install -r requirements.txt
  2. Create .env with ALPACA_API_KEY / ALPACA_SECRET_KEY (paper account)
  3. Edit TICKERS below to your chosen stocks
  4. Run manually first: python multi_stock_strategy.py
  5. Schedule via the same run_loop.bat pattern used for the wheel bot

IMPORTANT:
  - This is PAPER TRADING ONLY (paper=True below). Treat this as an
    extended test -- the plan discussed is a minimum 4-6 weeks of
    paper results, across both up and down conditions, before funding
    with real capital.
  - Performance tracking (return, max drawdown, rule-trigger counts) is
    logged on every run so you have real data to evaluate, not just a
    feeling about how it went.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

# ---------------------------------------------------------------------------
# CONFIG -- edit this list to choose which stocks the strategy trades
# ---------------------------------------------------------------------------
TICKERS = ["SOFI", "NOK", "RIVN", "SOUN", "ONDS", "PATH"]

STATE_FILE = Path(__file__).parent / "state.json"
LOG_FILE = Path(__file__).parent / "strategy_log.txt"
LOCK_FILE = Path(__file__).parent / "strategy.lock"

RISK_PER_TRADE_PCT = 0.02      # max 2% of total equity at risk per position
RESERVE_CASH_PCT = 0.25        # always keep >=25% of equity uninvested

INITIAL_FLOOR_PCT = 0.10
RATCHET_TRIGGER_PCT = 0.10
RATCHET_TRAIL_PCT = 0.05
RATCHET_STEP_PCT = 0.05

LADDER_1_DROP_PCT = 0.10       # ladder buy #1: same $ as initial position
LADDER_2_DROP_PCT = 0.20       # ladder buy #2: 2x the initial $ amount

PAPER_TRADING = True

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not found. Check your .env file.")
    sys.exit(1)

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def log(message: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def notify(subject: str, body: str):
    """Sends an email alert for a real, meaningful event -- NOT called on
    routine 'still holding' check-ins, only on actual trades, floor
    breaches, ratchet updates, and errors worth knowing about right away.
    Silently does nothing if email isn't configured."""
    sender = os.environ.get("EMAIL_ADDRESS")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    recipient = os.environ.get("NOTIFY_TO_EMAIL", sender)
    if not sender or not app_password:
        return
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg["Subject"] = f"[Growth Strategy] {subject}"
    msg["From"] = sender
    msg["To"] = recipient
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [recipient], msg.as_string())
    except Exception as e:
        log(f"Email notification failed to send: {e}")


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "positions": {},          # ticker -> position state dict
        "performance": {
            "starting_equity": None,
            "peak_equity": None,
            "floor_breach_count": 0,
            "ladder_trigger_count": 0,
            "ratchet_update_count": 0,
        },
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_current_price(symbol: str) -> float:
    """Returns a usable current price, guarding against one-sided quotes
    (bid or ask == 0), which otherwise silently produce a garbage midpoint
    -- e.g. (16.16 + 0) / 2 = 8.08, which is exactly what caused the SOFI
    incident. Falls back to the latest trade price if the quote is bad."""
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quote = data_client.get_stock_latest_quote(request)[symbol]

    if quote.bid_price and quote.ask_price and quote.bid_price > 0 and quote.ask_price > 0:
        return (quote.bid_price + quote.ask_price) / 2

    log(f"{symbol}: one-sided or invalid quote (bid={quote.bid_price}, "
        f"ask={quote.ask_price}). Falling back to latest trade price.")
    trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol)
    trade = data_client.get_stock_latest_trade(trade_request)[symbol]
    return trade.price


def get_shares_held(symbol: str) -> float:
    try:
        position = trading_client.get_open_position(symbol)
        return float(position.qty)
    except APIError as e:
        if getattr(e, "status_code", None) == 404:
            return 0.0
        raise


def get_account_cash_and_equity():
    account = trading_client.get_account()
    return float(account.cash), float(account.equity)


def place_market_order(symbol: str, side: OrderSide, qty: float):
    order = MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
    )
    return trading_client.submit_order(order)


def place_and_verify_market_order(symbol: str, side: OrderSide, qty: float, max_checks: int = 6, wait_seconds: float = 2.0):
    """Submit a market order and actively confirm it filled before returning
    True. During market hours, market orders normally fill within a second
    or two -- if it's STILL not filled after several checks, something is
    wrong (as we saw happen once already), and we must NOT assume success.
    If the order is REJECTED by Alpaca outright (e.g. insufficient buying
    power due to unsettled funds), that's caught here and logged instead of
    crashing the whole script -- a single bad order should never take down
    the rest of the run."""
    import time as _time
    try:
        result = trading_client.submit_order(
            MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        )
    except APIError as e:
        log(f"{symbol}: ORDER REJECTED at submission -- {e}. "
            f"Likely insufficient buying power (possibly unsettled funds). "
            f"Skipping this trade -- will retry next check-in.")
        return False, None

    order_id = str(result.id)

    for _ in range(max_checks):
        _time.sleep(wait_seconds)
        order = trading_client.get_order_by_id(order_id)
        if order.status.value == "filled":
            return True, float(order.filled_avg_price) if order.filled_avg_price else None
        if order.status.value in ("canceled", "expired", "rejected"):
            log(f"{symbol}: order {order_id} was {order.status.value}, never filled.")
            return False, None

    log(f"{symbol}: WARNING -- order {order_id} still not confirmed filled after "
        f"{max_checks * wait_seconds:.0f}s. Treating as UNCONFIRMED -- taking no "
        f"further state changes until this resolves.")
    return False, None


MAX_PLAUSIBLE_MOVE_PCT = 0.25  # a >25% move in one 15-min check is treated as
                                 # suspect data, not acted on, until re-confirmed


def sanity_check_price(symbol: str, new_price: float, last_known_price: float) -> bool:
    """Returns False if the price move since the last check is implausibly
    large, suggesting bad/erroneous quote data rather than a real move.
    This is what should have caught the SOFI $15.15 -> $8.08 bad tick."""
    if last_known_price is None or last_known_price <= 0:
        return True  # nothing to compare against yet
    pct_move = abs(new_price - last_known_price) / last_known_price
    if pct_move > MAX_PLAUSIBLE_MOVE_PCT:
        log(f"{symbol}: SUSPICIOUS PRICE DATA -- last known ${last_known_price:.2f}, "
            f"now reading ${new_price:.2f} ({pct_move:+.1%} in one check-in). "
            f"This exceeds the {MAX_PLAUSIBLE_MOVE_PCT:.0%} plausibility threshold. "
            f"Treating as bad data -- skipping any action on {symbol} this run.")
        return False
    return True


def calculate_position_size(equity: float, price: float) -> float:
    """Position size is the SMALLER of two caps:
    1. Risk-based: sized so hitting the initial floor (a 10% drop) costs
       exactly RISK_PER_TRADE_PCT of total equity, not more.
    2. Fair-share: no single position may exceed an equal share of the
       capital actually available for investing (equity minus reserve),
       divided across the full ticker list -- this is what keeps position
       sizing sane as TICKERS grows (e.g. 3 tickers vs 6 tickers)."""
    risk_dollars = equity * RISK_PER_TRADE_PCT
    risk_based_dollars = risk_dollars / INITIAL_FLOOR_PCT

    investable_equity = equity * (1 - RESERVE_CASH_PCT)
    fair_share_dollars = investable_equity / len(TICKERS)

    position_dollars = min(risk_based_dollars, fair_share_dollars)
    shares = round(position_dollars / price, 4)  # fractional shares OK on Alpaca
    return shares


def update_performance_tracking(state: dict, equity: float):
    perf = state["performance"]
    if perf["starting_equity"] is None:
        perf["starting_equity"] = equity
        perf["peak_equity"] = equity
    if equity > perf["peak_equity"]:
        perf["peak_equity"] = equity

    total_return_pct = (equity - perf["starting_equity"]) / perf["starting_equity"] * 100
    drawdown_pct = (equity - perf["peak_equity"]) / perf["peak_equity"] * 100

    log(f"PERFORMANCE: equity=${equity:.2f}, total_return={total_return_pct:+.2f}%, "
        f"drawdown_from_peak={drawdown_pct:.2f}%, "
        f"floor_breaches={perf['floor_breach_count']}, "
        f"ladder_triggers={perf['ladder_trigger_count']}, "
        f"ratchet_updates={perf['ratchet_update_count']}")


def try_open_new_position(symbol: str, state: dict, cash: float, equity: float):
    """Open a new position in `symbol` if there's room under the reserve
    and risk rules. Returns updated cash if a trade was made."""
    if symbol in state["positions"]:
        return cash  # already have a position in this ticker

    reserve_required = equity * RESERVE_CASH_PCT
    cash_available_for_new_position = cash - reserve_required

    if cash_available_for_new_position <= 0:
        log(f"{symbol}: skipping new entry -- cash reserve requirement "
            f"(${reserve_required:.2f}) not met (cash=${cash:.2f}).")
        return cash

    price = get_current_price(symbol)
    shares = calculate_position_size(equity, price)
    cost = shares * price

    if cost > cash_available_for_new_position:
        log(f"{symbol}: skipping new entry -- position cost (${cost:.2f}) exceeds "
            f"cash available after reserve (${cash_available_for_new_position:.2f}).")
        return cash

    if shares <= 0:
        log(f"{symbol}: skipping new entry -- calculated share size is zero.")
        return cash

    filled, fill_price = place_and_verify_market_order(symbol, OrderSide.BUY, shares)
    if not filled:
        log(f"{symbol}: new entry order not confirmed filled. Not recording a position -- "
            f"will retry next run.")
        return cash

    actual_price = fill_price or price
    log(f"OPENED NEW POSITION: {symbol} -- {shares} shares @ ~${actual_price:.2f} "
        f"(cost ~${cost:.2f}, risking ~${equity * RISK_PER_TRADE_PCT:.2f} at the floor)")
    notify(f"Opened new position: {symbol}",
           f"Bought {shares} shares of {symbol} @ ~${actual_price:.2f} "
           f"(cost ~${cost:.2f}).")

    state["positions"][symbol] = {
        "entry_price": actual_price,
        "last_known_price": actual_price,
        "shares": shares,
        "floor": actual_price * (1 - INITIAL_FLOOR_PCT),
        "ratchet_active": False,
        "high_water_mark": actual_price,
        "ladder_1_triggered": False,
        "ladder_2_triggered": False,
    }
    save_state(state)
    return cash - cost


def manage_existing_position(symbol: str, state: dict, cash: float):
    pos = state["positions"][symbol]
    current_price = get_current_price(symbol)

    # SAFETY CHECK #1: reject implausible price moves before doing anything else.
    # This is what should have caught the SOFI bad-tick incident.
    last_known = pos.get("last_known_price", pos["entry_price"])
    if not sanity_check_price(symbol, current_price, last_known):
        return cash  # skip this symbol entirely this run; state untouched

    pos["last_known_price"] = current_price
    entry_price = pos["entry_price"]
    pct_change = (current_price - entry_price) / entry_price

    log(f"{symbol}: price=${current_price:.2f}, entry=${entry_price:.2f}, "
        f"change={pct_change:+.2%}, floor=${pos['floor']:.2f}, "
        f"ratchet_active={pos['ratchet_active']}")

    if not pos["ratchet_active"] and current_price >= entry_price * (1 + RATCHET_TRIGGER_PCT):
        pos["ratchet_active"] = True
        pos["high_water_mark"] = current_price
        new_floor = current_price * (1 - RATCHET_TRAIL_PCT)
        if new_floor > pos["floor"]:
            pos["floor"] = new_floor
        state["performance"]["ratchet_update_count"] += 1
        log(f"{symbol}: RATCHET ACTIVATED. New floor=${pos['floor']:.2f}")

    elif pos["ratchet_active"]:
        if current_price >= pos["high_water_mark"] * (1 + RATCHET_STEP_PCT):
            pos["high_water_mark"] = current_price
            new_floor = current_price * (1 - RATCHET_TRAIL_PCT)
            if new_floor > pos["floor"]:
                pos["floor"] = new_floor
                state["performance"]["ratchet_update_count"] += 1
                log(f"{symbol}: FLOOR RATCHETED UP to ${new_floor:.2f}")

    # SAFETY CHECK #2: never assume a sell filled. Verify before touching state.
    if current_price <= pos["floor"]:
        shares = get_shares_held(symbol)
        if shares > 0:
            log(f"{symbol}: FLOOR BREACHED (${current_price:.2f} <= ${pos['floor']:.2f}). "
                f"Submitting sell for {shares} shares.")
            filled, fill_price = place_and_verify_market_order(symbol, OrderSide.SELL, shares)
            if filled:
                log(f"{symbol}: SELL CONFIRMED FILLED @ ~${fill_price:.2f}. Position closed.")
                pnl = (fill_price - pos["entry_price"]) * shares
                notify(f"Floor breach - sold {symbol}",
                       f"Sold {shares} shares of {symbol} @ ~${fill_price:.2f} "
                       f"(entry was ${pos['entry_price']:.2f}). Approx P/L: ${pnl:.2f}.")
                state["performance"]["floor_breach_count"] += 1
                del state["positions"][symbol]
                save_state(state)
            else:
                log(f"{symbol}: Sell NOT confirmed filled. Leaving position in state as-is "
                    f"-- will re-check next run rather than assume it's closed.")
                save_state(state)
        else:
            # No shares actually held -- position was likely already closed
            # some other way. Safe to clear from state.
            del state["positions"][symbol]
            save_state(state)
        return cash

    # SAFETY CHECK #2 (again): same verify-before-crediting approach for laddering buys.
    if not pos["ladder_1_triggered"] and pct_change <= -LADDER_1_DROP_PCT:
        cost = pos["entry_price"] * pos["shares"]
        if cash >= cost:
            shares_to_buy = round(cost / current_price, 4)
            filled, fill_price = place_and_verify_market_order(symbol, OrderSide.BUY, shares_to_buy)
            if filled:
                pos["ladder_1_triggered"] = True
                pos["shares"] += shares_to_buy
                state["performance"]["ladder_trigger_count"] += 1
                cash -= shares_to_buy * (fill_price or current_price)
                log(f"{symbol}: LADDER 1 CONFIRMED FILLED. Bought {shares_to_buy} more shares "
                    f"@ ~${fill_price:.2f}.")
            else:
                log(f"{symbol}: LADDER 1 triggered but fill not confirmed. Will retry next run.")
        else:
            log(f"{symbol}: LADDER 1 threshold hit, but insufficient cash (need ~${cost:.2f}, "
                f"have ${cash:.2f}). Skipping to avoid margin.")

    if not pos["ladder_2_triggered"] and pct_change <= -LADDER_2_DROP_PCT:
        cost = (pos["entry_price"] * pos["shares"]) * 2
        if cash >= cost:
            shares_to_buy = round(cost / current_price, 4)
            filled, fill_price = place_and_verify_market_order(symbol, OrderSide.BUY, shares_to_buy)
            if filled:
                pos["ladder_2_triggered"] = True
                pos["shares"] += shares_to_buy
                state["performance"]["ladder_trigger_count"] += 1
                cash -= shares_to_buy * (fill_price or current_price)
                log(f"{symbol}: LADDER 2 CONFIRMED FILLED. Bought {shares_to_buy} more shares "
                    f"@ ~${fill_price:.2f}.")
            else:
                log(f"{symbol}: LADDER 2 triggered but fill not confirmed. Will retry next run.")
        else:
            log(f"{symbol}: LADDER 2 threshold hit, but insufficient cash (need ~${cost:.2f}, "
                f"have ${cash:.2f}). Skipping to avoid margin.")

    save_state(state)
    return cash


def run_strategy():
    state = load_state()
    cash, equity = get_account_cash_and_equity()
    log(f"Check-in. Cash=${cash:.2f}, Equity=${equity:.2f}, "
        f"Open positions: {list(state['positions'].keys())}")

    for symbol in list(state["positions"].keys()):
        cash = manage_existing_position(symbol, state, cash)

    for symbol in TICKERS:
        cash = try_open_new_position(symbol, state, cash, equity)

    update_performance_tracking(state, equity)
    save_state(state)


def acquire_lock() -> bool:
    """Returns True if the lock was acquired (safe to proceed), False if
    another instance appears to already be running. This directly prevents
    the failure mode that caused the Aug 10 incident: multiple concurrent
    copies of this script each independently deciding to buy the same
    ticker before any of them could save state.json, resulting in 5-6x
    oversized positions per symbol."""
    if LOCK_FILE.exists():
        # Check how old the lock is -- a genuinely stuck/crashed process
        # shouldn't be able to block this script forever. Anything older
        # than 5 minutes is treated as stale (a normal run takes seconds).
        age_seconds = time.time() - LOCK_FILE.stat().st_mtime
        if age_seconds < 300:
            log(f"LOCK HELD by another instance (age {age_seconds:.0f}s). "
                f"Refusing to run concurrently -- exiting cleanly. "
                f"This is the safeguard against the Aug 10 duplicate-buy bug.")
            return False
        else:
            log(f"Found a stale lock file (age {age_seconds:.0f}s, >5min old) "
                f"-- assuming a previous run crashed without cleaning up. "
                f"Removing it and proceeding.")
            LOCK_FILE.unlink()

    LOCK_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


if __name__ == "__main__":
    import time
    if not acquire_lock():
        sys.exit(0)
    try:
        run_strategy()
    except Exception as e:
        log(f"UNEXPECTED ERROR this run: {type(e).__name__}: {e}. "
            f"This run is aborting, but the loop will retry automatically "
            f"in ~15 minutes. No state was corrupted by this error.")
        notify("Unexpected error - worth a look",
               f"{type(e).__name__}: {e}\n\n"
               f"This run aborted, but the next scheduled run will retry "
               f"automatically. No state was corrupted.")
    finally:
        release_lock()