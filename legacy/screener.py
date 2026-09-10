"""
A 股 15 分钟量价异动筛选 + 历史回测
=====================================

策略规则（用户定义）：
  1. 过去 1 个月（默认 20 个交易日）单日最大涨幅 < 3%   (避免追高)
  2. 15 分钟线层面：
     (a) 当根 15min 涨幅 0% ~ 2%  (温和上涨)
     (b) 当根成交量 > 前一根 15min × 2  (放量)
     (c) 最近两个交易日的 15min 总成交量：今日 >= 昨日 × 0.9 (震荡/缩量后启动)

回测规则：
  - 标的：A股全市场（北交所/科创板/创业板均可，按股票池过滤）
  - 资金：每只 1 万元，等权
  - 9:30 开盘价按整百股买入
  - 持仓期 15 分钟检查一次：
      涨 1% → 止盈卖出
      跌 1% → 止损卖出
  - 15:00 仍未触发 → 收盘价强制平仓

数据源：
  - stock_zh_a_spot   新浪全市场快照（实时）
  - stock_zh_a_hist    东财日 K
  - stock_zh_a_minute  新浪 15min K（覆盖近 6 个月）

用法：
  python screener.py --mode=screen
  python screener.py --mode=backtest --year=2026 --month=8
  python screener.py --mode=backtest --date=2026-08-15
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============ 全局配置 ============
POOL_MIN_TURNOVER = 5_000_000       # 股票池：当日成交额 >= 500 万
POOL_EXCLUDE_PREFIX = ('bj',)        # 排除北交所（流动性差，规则不适用）
LOOKBACK_TRADING_DAYS = 20          # "近一个月" = 20 个交易日
INTRADAY_PCT_MIN = 0.0              # 15min 涨幅下限
INTRADAY_PCT_MAX = 2.0              # 15min 涨幅上限
VOL_MULT = 2.0                      # 当根量/前根量 倍数
YESTERDAY_VOL_RATIO = 0.9           # 今日量/昨日量 比值下限（>= 视为"昨日震荡/下跌"）
TP_PCT = 1.0                        # 止盈
SL_PCT = 1.0                        # 止损
CAPITAL_PER_STOCK = 10_000          # 每只 1 万元
LOT = 100                           # 整百股

OUTPUT_DIR = Path("results")
CACHE_DIR = Path(".cache")

# ============ 工具函数 ============

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def code_format(code: str) -> str:
    """stock_zh_a_spot 返回 bj/sh/sz 前缀，转 stock_zh_a_hist 的纯 6 位代码"""
    s = str(code).strip()
    if s.startswith(('sh', 'sz', 'bj')):
        return s[2:]
    return s


def market_prefix(code: str) -> str:
    """stock_zh_a_minute 需要带 sh/sz/bj 前缀"""
    s = str(code).strip()
    if s.startswith(('sh', 'sz', 'bj')):
        return s
    if s.startswith(('60', '68', '9')):
        return f"sh{s}"
    if s.startswith(('00', '30', '20')):
        return f"sz{s}"
    if s.startswith('8') or s.startswith('4'):
        return f"bj{s}"
    return s


def is_valid_pool_code(code: str) -> bool:
    s = str(code)
    if s.startswith(POOL_EXCLUDE_PREFIX):
        return False
    if 'ST' in s or '退' in s:
        return False
    return True


# ============ 数据获取 ============

_pool_cache = None


def get_stock_pool() -> pd.DataFrame:
    """拉一次全市场快照，过滤后作为股票池。"""
    global _pool_cache
    if _pool_cache is not None:
        return _pool_cache
    log("拉取全市场快照…")
    df = ak.stock_zh_a_spot()
    df = df.rename(columns={'代码': 'code', '名称': 'name',
                            '最新价': 'price', '涨跌幅': 'pct',
                            '成交额': 'turnover', '今开': 'open',
                            '昨收': 'prev_close'})
    df = df[~df['code'].str.startswith(POOL_EXCLUDE_PREFIX)]
    df = df[df['turnover'] >= POOL_MIN_TURNOVER]
    df = df[~df['name'].str.contains('ST|退', na=False)]
    df = df[df['price'] >= 1.0]
    log(f"股票池: {len(df)} 只")
    _pool_cache = df.reset_index(drop=True)
    return _pool_cache


def fetch_daily(code: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """拉日 K（前复权），失败重试。"""
    code6 = code_format(code)
    for i in range(retries):
        try:
            df = ak.stock_zh_a_hist(symbol=code6, period='daily',
                                    start_date=start.replace('-', ''),
                                    end_date=end.replace('-', ''),
                                    adjust='qfq')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(1.0)
    return pd.DataFrame()


def fetch_15min(code: str, retries: int = 3) -> pd.DataFrame:
    """拉 15min K 线（新浪源，前复权），失败重试。"""
    symbol = f"{market_prefix(code)}"
    for i in range(retries):
        try:
            df = ak.stock_zh_a_minute(symbol=symbol, period='15', adjust='qfq')
            if df is not None and not df.empty:
                df['day'] = pd.to_datetime(df['day'])
                return df
        except Exception as e:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(2.0)
    return pd.DataFrame()


# ============ 筛选规则 ============

def check_past_month_daily(code: str, target_date: str, lookback: int = LOOKBACK_TRADING_DAYS) -> bool:
    """近 N 个交易日单日最大涨幅 < 3%。"""
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    end = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    start = (target_dt - timedelta(days=lookback * 2 + 30)).strftime('%Y-%m-%d')
    df = fetch_daily(code, start, end)
    if df.empty or len(df) < lookback:
        return False
    last_n = df.tail(lookback)
    if '涨跌幅' not in last_n.columns:
        return False
    max_pct = last_n['涨跌幅'].max()
    return pd.notna(max_pct) and max_pct < 3.0


def check_15min_signal_on_bar(latest_bar: pd.Series, prev_bar: pd.Series,
                              recent_two_days_15min: pd.DataFrame) -> tuple:
    """
    在已拿到当日与昨日 15min K 的前提下，判定最近一根 15min 是否触发信号。
    返回 (是否触发, 涨幅%, 量比, 昨/今量比)。
    """
    try:
        open_p = float(latest_bar['open'])
        close_p = float(latest_bar['close'])
        cur_vol = float(latest_bar['volume'])
        prev_vol = float(prev_bar['volume'])
    except Exception:
        return (False, 0, 0, 0)

    if open_p <= 0 or prev_vol <= 0:
        return (False, 0, 0, 0)

    change_pct = (close_p - open_p) / open_p * 100.0
    vol_ratio = cur_vol / prev_vol

    # (a) 涨幅 0~2%
    if not (INTRADAY_PCT_MIN <= change_pct <= INTRADAY_PCT_MAX):
        return (False, change_pct, vol_ratio, 0)

    # (b) 当根量 > 前根量 × 2
    if vol_ratio < VOL_MULT:
        return (False, change_pct, vol_ratio, 0)

    # (c) 最近两个交易日的 15min 总成交量：今日 >= 昨日 × 0.9
    if recent_two_days_15min is None or recent_two_days_15min.empty:
        return (False, change_pct, vol_ratio, 0)

    daily_vol = recent_two_days_15min.groupby(
        recent_two_days_15min['day'].dt.date)['volume'].sum()
    if len(daily_vol) < 2:
        return (False, change_pct, vol_ratio, 0)

    last_two = daily_vol.tail(2)
    yesterday_vol = float(last_two.iloc[0])
    today_vol_so_far = float(last_two.iloc[1])
    if yesterday_vol <= 0:
        return (False, change_pct, vol_ratio, 0)
    ytd_ratio = today_vol_so_far / yesterday_vol
    if ytd_ratio < YESTERDAY_VOL_RATIO:
        return (False, change_pct, vol_ratio, ytd_ratio)

    return (True, change_pct, vol_ratio, ytd_ratio)


# ============ 实时筛选 ============

def screen_one(code: str, name: str, target_date: str) -> dict | None:
    """对单只股票执行实时筛选。"""
    # 1) 近一月单日涨幅 < 3%
    if not check_past_month_daily(code, target_date):
        return None

    # 2) 拉 15min K
    df = fetch_15min(code)
    if df.empty:
        return None

    target_dt = pd.to_datetime(target_date).date() if not isinstance(target_date, type(datetime.now().date())) else target_date
    if isinstance(target_date, str):
        target_dt = pd.to_datetime(target_date).date()
    else:
        target_dt = target_date

    today_bars = df[df['day'].dt.date == target_dt].reset_index(drop=True)
    if len(today_bars) < 2:
        return None

    # 取最近一根 15min
    latest = today_bars.iloc[-1]
    prev = today_bars.iloc[-2]

    # 最近两个交易日的 15min K（含今天）
    prev_date = target_dt - timedelta(days=1)
    window = df[(df['day'].dt.date >= prev_date - timedelta(days=5)) &
                (df['day'].dt.date <= target_dt)]
    recent_two = window[window['day'].dt.date >= prev_date]

    triggered, chg, volr, ytd = check_15min_signal_on_bar(latest, prev, recent_two)
    if not triggered:
        return None

    return {
        'code': code,
        'name': name,
        'date': str(target_dt),
        'bar_time': str(latest['day']),
        'open': float(latest['open']),
        'close': float(latest['close']),
        'change_pct': round(chg, 2),
        'vol_ratio': round(volr, 2),
        'today_yesterday_vol_ratio': round(ytd, 2),
    }


def run_screen(target_date: str | None = None, max_workers: int = 8) -> pd.DataFrame:
    pool = get_stock_pool()
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')

    log(f"实时筛选：target_date={target_date}, 池子={len(pool)} 只")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(screen_one, row['code'], row['name'], target_date): row
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="筛选进度"):
            try:
                r = fut.result(timeout=60)
            except Exception:
                r = None
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        log("无标的触发信号。")
    else:
        log(f"命中 {len(df)} 只")
    return df


# ============ 回测引擎 ============

def backtest_one_day(code: str, name: str, target_date: str,
                     capital: float = CAPITAL_PER_STOCK) -> dict | None:
    """
    对单只股票在指定交易日执行回测：
    - 9:30 开盘价买入（整百股）
    - 之后每根 15min 检查是否触发 1% 止盈 / 1% 止损
    - 收盘强制平仓
    """
    # 1) 近一月单日涨幅 < 3%（前提）
    if not check_past_month_daily(code, target_date):
        return None

    # 2) 当日是否存在入场信号（沿用实时筛选规则）
    df = fetch_15min(code)
    if df.empty:
        return None

    target_dt = pd.to_datetime(target_date).date()
    today_bars = df[df['day'].dt.date == target_dt].reset_index(drop=True)
    if len(today_bars) < 2:
        return None

    latest = today_bars.iloc[-1]
    prev = today_bars.iloc[-2]
    prev_date = target_dt - timedelta(days=1)
    window = df[(df['day'].dt.date >= prev_date - timedelta(days=5)) &
                (df['day'].dt.date <= target_dt)]
    recent_two = window[window['day'].dt.date >= prev_date]
    triggered, chg, volr, ytd = check_15min_signal_on_bar(latest, prev, recent_two)
    if not triggered:
        return None

    # 3) 入场：以当天 9:30 那根的开盘价买入
    open_bar = today_bars.iloc[0]
    open_price = float(open_bar['open'])
    if open_price <= 0:
        return None
    shares = int(capital / open_price / LOT) * LOT
    if shares <= 0:
        return None
    actual_capital = shares * open_price

    # 4) 逐根 15min 模拟卖出
    exit_price = None
    exit_bar = None
    exit_reason = None
    for i in range(1, len(today_bars)):
        cur = today_bars.iloc[i]
        cur_p = float(cur['close'])
        # 实际可成交价：用 close 近似（保守）
        pnl_pct = (cur_p - open_price) / open_price * 100.0
        if pnl_pct >= TP_PCT:
            exit_price = cur_p
            exit_bar = i
            exit_reason = '止盈'
            break
        if pnl_pct <= -SL_PCT:
            exit_price = cur_p
            exit_bar = i
            exit_reason = '止损'
            break
    if exit_price is None:
        last = today_bars.iloc[-1]
        exit_price = float(last['close'])
        exit_bar = len(today_bars) - 1
        exit_reason = '收盘平仓'

    pnl = (exit_price - open_price) * shares
    pnl_pct = (exit_price - open_price) / open_price * 100.0
    return {
        'code': code, 'name': name, 'date': target_date,
        'signal_time': str(latest['day']),
        'signal_change_pct': round(chg, 2),
        'signal_vol_ratio': round(volr, 2),
        'today_yesterday_vol_ratio': round(ytd, 2),
        'entry_price': round(open_price, 3),
        'exit_price': round(exit_price, 3),
        'shares': shares,
        'capital_used': round(actual_capital, 2),
        'pnl': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'exit_reason': exit_reason,
        'exit_bar': exit_bar,
    }


def run_backtest(year: int, month: int, max_workers: int = 8,
                 sample_size: int | None = None) -> pd.DataFrame:
    """对 year-month 的每个交易日跑回测。"""
    pool = get_stock_pool()
    if sample_size is not None and sample_size < len(pool):
        pool = pool.sample(n=sample_size, random_state=42).reset_index(drop=True)
        log(f"抽样股票池: {len(pool)} 只")

    # 生成月份内的所有日期，再过滤到交易日
    first = datetime(year, month, 1)
    if month == 12:
        last = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = datetime(year, month + 1, 1) - timedelta(days=1)
    all_dates = pd.date_range(first, last).strftime('%Y-%m-%d').tolist()

    # 用日 K 接口拉上证指数来确认交易日
    try:
        idx_df = ak.stock_zh_index_daily(symbol='sh000001')
        idx_df['date'] = pd.to_datetime(idx_df['date']).dt.strftime('%Y-%m-%d')
        trade_dates = [d for d in all_dates if d in set(idx_df['date'].tolist())]
    except Exception:
        trade_dates = all_dates
    log(f"交易日: {len(trade_dates)} 天（{trade_dates[0] if trade_dates else '无'} ~ {trade_dates[-1] if trade_dates else '无'}）")

    # 缓存：股票池 (code,name) -> 15min df，避免每天重复拉
    cache_file = CACHE_DIR / f"15min_cache_{year}_{month:02d}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log("预拉 15min K 线缓存…（一次性）")
    cache_15min = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_15min, row['code']): row['code'] for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="预拉15min"):
            code = futures[fut]
            try:
                df = fut.result(timeout=60)
                if df is not None and not df.empty:
                    cache_15min[code] = df
            except Exception:
                pass
    log(f"15min 缓存: {len(cache_15min)} 只")

    # 缓存：日 K
    log("预拉近 3 个月日 K 缓存…")
    daily_start = (first - timedelta(days=60)).strftime('%Y-%m-%d')
    daily_end = last.strftime('%Y-%m-%d')
    cache_daily = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_daily, row['code'], daily_start, daily_end): row['code']
                   for _, row in pool.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="预拉日K"):
            code = futures[fut]
            try:
                df = fut.result(timeout=60)
                if df is not None and not df.empty:
                    cache_daily[code] = df
            except Exception:
                pass
    log(f"日K 缓存: {len(cache_daily)} 只")

    # 用缓存做日级回测
    all_trades = []
    for td in trade_dates:
        log(f"回测 {td}…")
        day_results = []
        for _, row in pool.iterrows():
            code = row['code']
            name = row['name']
            ddf = cache_daily.get(code)
            if ddf is None or ddf.empty:
                continue
            target_dt = pd.to_datetime(td).date()
            # 近 20 个交易日单日涨幅 < 3%
            date_col = pd.to_datetime(ddf['日期']).dt.date
            ddf_pre = ddf[date_col < target_dt]
            if len(ddf_pre) < LOOKBACK_TRADING_DAYS:
                continue
            last20 = ddf_pre.tail(LOOKBACK_TRADING_DAYS)
            if '涨跌幅' not in last20.columns:
                continue
            if last20['涨跌幅'].max() >= 3.0:
                continue

            # 当日 15min 信号
            df15 = cache_15min.get(code)
            if df15 is None or df15.empty:
                continue
            today = df15[df15['day'].dt.date == target_dt].reset_index(drop=True)
            if len(today) < 2:
                continue
            latest = today.iloc[-1]
            prev = today.iloc[-2]
            prev_date = target_dt - timedelta(days=1)
            window = df15[(df15['day'].dt.date >= prev_date - timedelta(days=5)) &
                          (df15['day'].dt.date <= target_dt)]
            recent_two = window[window['day'].dt.date >= prev_date]
            triggered, chg, volr, ytd = check_15min_signal_on_bar(latest, prev, recent_two)
            if not triggered:
                continue

            # 入场
            open_price = float(today.iloc[0]['open'])
            if open_price <= 0:
                continue
            shares = int(CAPITAL_PER_STOCK / open_price / LOT) * LOT
            if shares <= 0:
                continue
            actual_capital = shares * open_price

            # 模拟卖出
            exit_price, exit_bar, exit_reason = None, None, None
            for i in range(1, len(today)):
                cur_p = float(today.iloc[i]['close'])
                pnl_pct = (cur_p - open_price) / open_price * 100.0
                if pnl_pct >= TP_PCT:
                    exit_price, exit_bar, exit_reason = cur_p, i, '止盈'
                    break
                if pnl_pct <= -SL_PCT:
                    exit_price, exit_bar, exit_reason = cur_p, i, '止损'
                    break
            if exit_price is None:
                last = today.iloc[-1]
                exit_price = float(last['close'])
                exit_bar = len(today) - 1
                exit_reason = '收盘平仓'

            pnl = (exit_price - open_price) * shares
            pnl_pct = (exit_price - open_price) / open_price * 100.0
            day_results.append({
                'code': code, 'name': name, 'date': td,
                'signal_time': str(latest['day']),
                'signal_change_pct': round(chg, 2),
                'signal_vol_ratio': round(volr, 2),
                'today_yesterday_vol_ratio': round(ytd, 2),
                'entry_price': round(open_price, 3),
                'exit_price': round(exit_price, 3),
                'shares': shares,
                'capital_used': round(actual_capital, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': exit_reason,
                'exit_bar': exit_bar,
            })

        log(f"  {td}: 命中 {len(day_results)} 笔")
        all_trades.extend(day_results)

    return pd.DataFrame(all_trades)


# ============ 报告输出 ============

def print_trade_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)
    if df.empty:
        print("  (无数据)")
        return
    cols = ['date', 'code', 'name', 'entry_price', 'exit_price', 'pnl_pct', 'pnl', 'exit_reason']
    show = df[cols].copy()
    show.columns = ['日期', '代码', '名称', '买入价', '卖出价', '盈亏%', '盈亏额', '结果']
    print(show.to_string(index=False))


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    flat = df[df['pnl'] == 0]
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    win_rate = len(wins) / len(df) * 100 if len(df) else 0
    avg_win = wins['pnl_pct'].mean() if len(wins) else 0
    avg_loss = losses['pnl_pct'].mean() if len(losses) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if len(losses) else float('inf')
    by_reason = df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']).round(2)
    daily = df.groupby('date')['pnl'].sum().round(2)
    return {
        'trades': len(df),
        'wins': len(wins),
        'losses': len(losses),
        'flat': len(flat),
        'win_rate': round(win_rate, 2),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
        'by_reason': by_reason,
        'daily_pnl': daily,
    }


def print_summary(s: dict):
    if not s:
        print("\n无回测结果。")
        return
    print("\n" + "=" * 60)
    print("  回测汇总")
    print("=" * 60)
    print(f"  总交易笔数   : {s['trades']}")
    print(f"  盈利 / 亏损 / 平 : {s['wins']} / {s['losses']} / {s['flat']}")
    print(f"  胜率         : {s['win_rate']}%")
    print(f"  总盈亏       : {s['total_pnl']:>10} 元")
    print(f"  单笔均盈亏   : {s['avg_pnl']:>10} 元")
    print(f"  盈利单均涨幅 : {s['avg_win_pct']}%")
    print(f"  亏损单均跌幅 : {s['avg_loss_pct']}%")
    print(f"  盈亏比(PF)   : {s['profit_factor']}")
    print("\n  按退出原因分组：")
    print(s['by_reason'].to_string())
    print("\n  每日盈亏：")
    print(s['daily_pnl'].to_string())


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description='A 股 15min 量价异动筛选 + 回测')
    parser.add_argument('--mode', choices=['screen', 'backtest'], required=True)
    parser.add_argument('--date', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--month', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sample', type=int, default=None, help='股票池抽样数（默认全量）')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == 'screen':
        df = run_screen(target_date=args.date, max_workers=args.workers)
        print_trade_table(df, "实时筛选命中")
        if not df.empty:
            f = out / f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(f, index=False, encoding='utf-8-sig')
            log(f"已保存：{f}")
    else:
        df = run_backtest(args.year, args.month, max_workers=args.workers,
                          sample_size=args.sample)
        if not df.empty:
            print_trade_table(df, f"{args.year}-{args.month:02d} 回测明细")
            s = summarize(df)
            print_summary(s)
            f_csv = out / f"backtest_{args.year}{args.month:02d}_trades.csv"
            df.to_csv(f_csv, index=False, encoding='utf-8-sig')
            log(f"明细已保存：{f_csv}")
            f_sum = out / f"backtest_{args.year}{args.month:02d}_summary.json"
            ss = {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in s.items()}
            with open(f_sum, 'w', encoding='utf-8') as fp:
                json.dump(ss, fp, ensure_ascii=False, indent=2, default=str)
            log(f"汇总已保存：{f_sum}")
        else:
            log("回测无任何成交。")


if __name__ == '__main__':
    main()
