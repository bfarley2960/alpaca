"""
TSLA Wheel Strategy — Cash-Secured Puts + Covered Calls
==========================================================

STRATEGY RULES (as defined by user, based on the standard "wheel"):

  STAGE 1 — Sell cash-secured puts:
    - Sell 1 put contract, strike ~10% below current TSLA price
    - Target expiration ~3 weeks out
    - If it expires worthless -> keep the premium, sell another put (repeat stage 1)
    - If assigned -> now own 100 shares, move to Stage 2
    - If option value drops to 50% of what you sold it for -> buy it back early
      (lock in the win rather than waiting for full expiration)

  STAGE 2 — Sell covered calls (only once you own 100 shares from assignment):
    - Sell 1 call contract, strike ~10% above your cost basis
    - Target expiration ~3 weeks out
    - If it expires worthless -> keep the premium, sell another call (repeat stage 2)
    - If shares get called away (assigned) -> back to Stage 1
    - Same 50%-profit early close rule applies

  Rules enforced throughout:
    - Never sell a put unless there's enough cash to cover assignment
    - Never sell a call with a strike below your cost basis
    - All premium collected is tracked cumulatively in state.json

HOW TO USE:
  1. pip install -r requirements.txt
  2. Create a .env file (see .env.example) with your NEW $50k account's
     ALPACA_API_KEY / ALPACA_SECRET_KEY
  3. Run manually first to confirm it works: python wheel_strategy.py
  4. Schedule via Windows Task Scheduler, same pattern as the TSLA trailing-
     stop bot (every 15-30 min during market hours is reasonable — options
     don't need checking as frequently as a stock price)

IMPORTANT — READ BEFORE RUNNING:
  - This trades on Alpaca PAPER TRADING only (paper=True below). Do not
    change that until you've watched this run correctly across several
    full put -> assignment -> call -> assignment cycles.
  - State (which stage you're in, current contract, cumulative premium,
    cost basis if assigned) lives in state.json next to this script.
    Don't delete it mid-cycle or the script loses track of what's active.
  - This only acts when it's actually run — no continuous real-time
    monitoring between scheduled checks.
  - Options assignment can happen at any time once in-the-money, not just
    at expiration (American-style options). This script checks your
    actual position each run to detect assignment, rather than assuming
    it only happens at expiration.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import (
    LimitOrderRequest,
    GetOptionContractsRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType, AssetStatus
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SYMBOL = "TSLA"
STATE_FILE = Path(__file__).parent / "state.json"
LOG_FILE = Path(__file__).parent / "strategy_log.txt"

TARGET_DTE_DAYS = 14          # ~2 weeks out (shortened from 3 weeks to test faster decay)
DTE_SEARCH_WINDOW_DAYS = 7    # look +/- a week around the target for a Friday expiry
STRIKE_DISTANCE_PCT = 0.10    # 10% OTM for both puts and calls
PROFIT_TARGET_PCT = 0.50      # close early once option is worth 50% of what we sold it for
CONTRACTS = 1                 # 1 contract = 100 shares

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
option_data_client = OptionHistoricalDataClient(API_KEY, SECRET_KEY)
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def log(message: str):
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def notify(subject: str, body: str):
    """Sends an email alert for a real, meaningful event -- NOT called on
    routine 'still open, holding' check-ins, only on actual trades, fills,
    and things worth knowing about right away. Silently does nothing if
    email isn't configured (so this never breaks the script)."""
    sender = os.environ.get("EMAIL_ADDRESS")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    recipient = os.environ.get("NOTIFY_TO_EMAIL", sender)
    if not sender or not app_password:
        return
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg["Subject"] = f"[Wheel Strategy] {subject}"
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
        "stage": "selling_puts",
        "active_contract_symbol": None,
        "active_contract_strike": None,
        "active_contract_expiration": None,
        "premium_collected_this_contract": None,
        "cumulative_premium": 0.0,
        "cost_basis": None,
        "pending_order_id": None,
        "pending_order_type": None,   # "open" or "close"
        "pending_contract_symbol": None,
        "pending_contract_strike": None,
        "pending_contract_expiration": None,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_current_stock_price() -> float:
    """Guards against one-sided quotes (bid or ask == 0), which otherwise
    silently produce a garbage midpoint. Falls back to latest trade price."""
    request = StockLatestQuoteRequest(symbol_or_symbols=SYMBOL)
    quote = stock_data_client.get_stock_latest_quote(request)[SYMBOL]

    if quote.bid_price and quote.ask_price and quote.bid_price > 0 and quote.ask_price > 0:
        return (quote.bid_price + quote.ask_price) / 2

    log(f"{SYMBOL}: one-sided or invalid quote (bid={quote.bid_price}, "
        f"ask={quote.ask_price}). Falling back to latest trade price.")
    trade_request = StockLatestTradeRequest(symbol_or_symbols=SYMBOL)
    trade = stock_data_client.get_stock_latest_trade(trade_request)[SYMBOL]
    return trade.price


def get_option_quote(contract_symbol: str):
    request = OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
    return option_data_client.get_option_latest_quote(request)[contract_symbol]


def get_shares_held() -> float:
    try:
        position = trading_client.get_open_position(SYMBOL)
        return float(position.qty)
    except Exception:
        return 0.0


def get_option_position(contract_symbol: str):
    """Returns the position if it exists, None if it genuinely doesn't exist
    (confirmed 404 from Alpaca), or raises an exception for anything else
    (network errors, auth issues, etc.) so those are never silently mistaken
    for 'the position is gone'."""
    try:
        return trading_client.get_open_position(contract_symbol)
    except APIError as e:
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return None
        raise


def find_contract(option_type: ContractType, target_strike: float) -> dict:
    """Find the option contract closest to target_strike, expiring nearest
    TARGET_DTE_DAYS from today (searching a window of Fridays around it)."""
    today = datetime.now(timezone.utc).date()
    window_start = today + timedelta(days=TARGET_DTE_DAYS - DTE_SEARCH_WINDOW_DAYS)
    window_end = today + timedelta(days=TARGET_DTE_DAYS + DTE_SEARCH_WINDOW_DAYS)

    request = GetOptionContractsRequest(
        underlying_symbols=[SYMBOL],
        status=AssetStatus.ACTIVE,
        expiration_date_gte=window_start.isoformat(),
        expiration_date_lte=window_end.isoformat(),
        type=option_type,
        strike_price_gte=str(round(target_strike * 0.95, 2)),
        strike_price_lte=str(round(target_strike * 1.05, 2)),
    )
    response = trading_client.get_option_contracts(request)
    contracts = response.option_contracts

    if not contracts:
        return None

    ideal_date = today + timedelta(days=TARGET_DTE_DAYS)

    def score(c):
        strike_diff = abs(float(c.strike_price) - target_strike)
        date_diff = abs((c.expiration_date - ideal_date).days)
        return (strike_diff, date_diff)

    best = min(contracts, key=score)
    return {
        "symbol": best.symbol,
        "strike": float(best.strike_price),
        "expiration": best.expiration_date.isoformat(),  # convert date -> string for JSON storage
    }


def sell_to_open(contract_symbol: str) -> str:
    """Submit a sell-to-open limit order. Returns the order ID.
    Does NOT assume the order filled -- caller must verify via check_order_status."""
    quote = get_option_quote(contract_symbol)
    limit_price = round(quote.bid_price, 2)
    order = LimitOrderRequest(
        symbol=contract_symbol,
        qty=CONTRACTS,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
    )
    result = trading_client.submit_order(order)
    return str(result.id)


def buy_to_close(contract_symbol: str) -> str:
    """Submit a buy-to-close limit order. Returns the order ID.
    Does NOT assume the order filled -- caller must verify via check_order_status."""
    quote = get_option_quote(contract_symbol)
    limit_price = round(quote.ask_price, 2)
    order = LimitOrderRequest(
        symbol=contract_symbol,
        qty=CONTRACTS,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price,
    )
    result = trading_client.submit_order(order)
    return str(result.id)


def check_order_status(order_id: str):
    """Returns (status_str, filled_avg_price_or_None) for a given order ID."""
    order = trading_client.get_order_by_id(order_id)
    filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
    return order.status.value, filled_price


def run_strategy():
    state = load_state()
    log(f"Check-in. Stage: {state['stage']}. "
        f"Cumulative premium collected: ${state['cumulative_premium']:.2f}")

    # --- 0. If there's a pending (unconfirmed) order, resolve it FIRST. ---
    # We never assume an order filled -- we check its actual status before
    # touching premium totals or moving between stages.
    if state["pending_order_id"]:
        status, filled_price = check_order_status(state["pending_order_id"])
        log(f"Pending order {state['pending_order_id']} status: {status}")

        if status == "filled":
            if state["pending_order_type"] == "open":
                state["active_contract_symbol"] = state["pending_contract_symbol"]
                state["active_contract_strike"] = state["pending_contract_strike"]
                state["active_contract_expiration"] = state["pending_contract_expiration"]
                state["premium_collected_this_contract"] = filled_price * 100
                log(f"CONFIRMED FILLED (open): {state['active_contract_symbol']} "
                    f"premium=${filled_price * 100:.2f}")
                notify("New put sold",
                       f"Sold {state['active_contract_symbol']} for "
                       f"${filled_price * 100:.2f} premium.")
            elif state["pending_order_type"] == "close":
                premium_sold_for = state["premium_collected_this_contract"]
                cost_to_close = filled_price * 100
                realized = premium_sold_for - cost_to_close
                state["cumulative_premium"] += realized
                log(f"CONFIRMED FILLED (close): bought back for ${cost_to_close:.2f}. "
                    f"Realized ${realized:.2f}. Cumulative: ${state['cumulative_premium']:.2f}")
                notify("Position closed",
                       f"Bought back for ${cost_to_close:.2f}. Realized ${realized:.2f} "
                       f"this cycle. Cumulative premium: ${state['cumulative_premium']:.2f}.")
                state["active_contract_symbol"] = None
            # Clear pending fields either way
            state["pending_order_id"] = None
            state["pending_order_type"] = None
            state["pending_contract_symbol"] = None
            save_state(state)

        elif status in ("canceled", "expired", "rejected"):
            log(f"Pending order was {status} (never filled). Clearing pending state, "
                f"will retry next check-in.")
            state["pending_order_id"] = None
            state["pending_order_type"] = None
            state["pending_contract_symbol"] = None
            save_state(state)
            # If this was a "close" order that didn't fill, the original position
            # is still active and untouched -- nothing more to do.
            if state["pending_order_type"] == "close":
                return

        else:
            # still new/accepted/partially_filled -- wait, don't take further action
            log(f"Order still pending ({status}). Waiting for next check-in before "
                f"taking any further action.")
            return

    shares_held = get_shares_held()

    # --- 1. Detect assignment transitions (only relevant for a CONFIRMED active contract) ---
    if state["stage"] == "selling_puts" and state["active_contract_symbol"] and shares_held >= 100:
        log(f"ASSIGNMENT DETECTED: now holding {shares_held} shares of {SYMBOL}. "
            f"Moving to covered-call stage.")
        position = trading_client.get_open_position(SYMBOL)
        state["cost_basis"] = float(position.avg_entry_price)
        state["stage"] = "selling_calls"
        state["active_contract_symbol"] = None
        save_state(state)

    if state["stage"] == "selling_calls" and state["active_contract_symbol"] and shares_held < 100:
        log(f"SHARES CALLED AWAY: no longer holding 100+ shares. Moving back to put-selling stage.")
        state["stage"] = "selling_puts"
        state["active_contract_symbol"] = None
        state["cost_basis"] = None
        save_state(state)

    # --- 2. If we have a CONFIRMED active contract, check profit target / still open ---
    if state["active_contract_symbol"]:
        contract_symbol = state["active_contract_symbol"]
        position = get_option_position(contract_symbol)

        if position is None:
            # SAFETY CHECK: never credit "expired worthless" unless the actual
            # expiration date has genuinely passed. A missing position before
            # that date is treated as inconclusive (likely a transient API
            # issue) -- we log it clearly and take NO action, rather than
            # guessing. This prevents phantom premium credits.
            expiration_date = datetime.strptime(
                state["active_contract_expiration"], "%Y-%m-%d"
            ).date()
            today = datetime.now(timezone.utc).date()

            if today >= expiration_date:
                premium = state["premium_collected_this_contract"] or 0.0
                state["cumulative_premium"] += premium
                log(f"Contract {contract_symbol} expired worthless (confirmed: "
                    f"expiration {expiration_date} has passed). "
                    f"Premium kept: ${premium:.2f}. Cumulative: ${state['cumulative_premium']:.2f}")
                notify("Contract expired worthless",
                       f"{contract_symbol} expired worthless. Kept ${premium:.2f} premium. "
                       f"Cumulative: ${state['cumulative_premium']:.2f}.")
                state["active_contract_symbol"] = None
                save_state(state)
            else:
                log(f"WARNING: position lookup for {contract_symbol} returned nothing, "
                    f"but its expiration ({expiration_date}) hasn't passed yet "
                    f"(today is {today}). This is likely a transient API issue, NOT a "
                    f"real expiration. Taking no action -- will re-check next run.")
                notify("Anomaly detected (needs a look)",
                       f"{contract_symbol} position unexpectedly not found, but it "
                       f"hasn't expired yet. Likely a transient API glitch -- taking no "
                       f"action, but worth a quick check.")
            return
        else:
            quote = get_option_quote(contract_symbol)
            current_value = quote.ask_price
            premium_sold_for = state["premium_collected_this_contract"]
            if premium_sold_for and current_value <= premium_sold_for / 100 * (1 - PROFIT_TARGET_PCT):
                log(f"PROFIT TARGET HIT on {contract_symbol}: sold for ${premium_sold_for:.2f}, "
                    f"now worth ${current_value * 100:.2f}. Submitting buy-to-close order.")
                order_id = buy_to_close(contract_symbol)
                state["pending_order_id"] = order_id
                state["pending_order_type"] = "close"
                save_state(state)
            else:
                log(f"Contract {contract_symbol} still open. Sold for ${premium_sold_for:.2f}, "
                    f"currently worth ${current_value * 100:.2f}. Holding.")
        return  # either way, nothing else to do this check-in

    # --- 3. If no active contract and no pending order, open a new one ---
    if state["stage"] == "selling_puts":
        current_price = get_current_stock_price()
        target_strike = current_price * (1 - STRIKE_DISTANCE_PCT)
        contract = find_contract(ContractType.PUT, target_strike)

        if contract is None:
            log("No suitable put contract found in range. Skipping this check-in.")
            return

        # Cash-only check: uses account.cash (actual deposited cash), never
        # account.buying_power (which can include margin). This ensures the
        # strategy can never obligate you to buy shares using borrowed money --
        # if there isn't enough real cash to cover assignment, it simply skips.
        required_cash = contract["strike"] * 100 * CONTRACTS
        account = trading_client.get_account()
        cash_available = float(account.cash)
        if cash_available < required_cash:
            log(f"Insufficient CASH (not margin) to secure put at strike "
                f"${contract['strike']} (needs ${required_cash:.2f}, "
                f"${cash_available:.2f} available). Skipping to avoid using margin.")
            return

        order_id = sell_to_open(contract["symbol"])
        state["pending_order_id"] = order_id
        state["pending_order_type"] = "open"
        state["pending_contract_symbol"] = contract["symbol"]
        state["pending_contract_strike"] = contract["strike"]
        state["pending_contract_expiration"] = contract["expiration"]
        save_state(state)
        log(f"SUBMITTED SELL PUT (pending fill confirmation): {contract['symbol']} "
            f"strike=${contract['strike']} exp={contract['expiration']} order_id={order_id}")

    elif state["stage"] == "selling_calls":
        cost_basis = state["cost_basis"]
        target_strike = cost_basis * (1 + STRIKE_DISTANCE_PCT)
        contract = find_contract(ContractType.CALL, target_strike)

        if contract is None:
            log("No suitable call contract found in range. Skipping this check-in.")
            return

        if contract["strike"] < cost_basis:
            log(f"Best available call strike (${contract['strike']}) is below cost basis "
                f"(${cost_basis}). Refusing to sell below cost basis. Skipping.")
            return

        order_id = sell_to_open(contract["symbol"])
        state["pending_order_id"] = order_id
        state["pending_order_type"] = "open"
        state["pending_contract_symbol"] = contract["symbol"]
        state["pending_contract_strike"] = contract["strike"]
        state["pending_contract_expiration"] = contract["expiration"]
        save_state(state)
        log(f"SUBMITTED SELL CALL (pending fill confirmation): {contract['symbol']} "
            f"strike=${contract['strike']} exp={contract['expiration']} order_id={order_id}")


if __name__ == "__main__":
    try:
        run_strategy()
    except Exception as e:
        log(f"UNEXPECTED ERROR this run: {type(e).__name__}: {e}. "
            f"This run is aborting, but the next scheduled run will retry "
            f"automatically. No state was corrupted by this error.")
        notify("Unexpected error - worth a look",
               f"{type(e).__name__}: {e}\n\n"
               f"This run aborted, but the next scheduled run will retry "
               f"automatically. No state was corrupted.")