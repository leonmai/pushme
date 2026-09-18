"""
盘中实时盯盘 (live_scout)
==========================
用途: 每个交易日的盘中, 按"增强版 v6"规则实时捕捉出现量价异动特征的个股。

增强版 v6 规则 (由三个月 302 笔回测调参得出, 见 results_tune/):
  1. 大盘择时: 上证指数当前点位 > MA20    (不满足 → 当日不出信号, 熊市不开仓)
     证据: PF 1.81 → 2.58, 胜率 68.9% → 73.7%
  2. 大前提: 近 5 个交易日累计跌幅 <= -3% (A级) / <= -1% (B级)
     证据: <=-3% 时 PF 2.51; 叠加 MA20 后 PF 4.89, 胜率 77.3%
  3. 信号根 15min: 涨幅 0~2% + 成交量 >= 前一根 ×2 + 当日累计量/昨日量 >= 0.9
     (量比阈值不再提高! 实测 >=2.5 反而变差: PF 1.81 → 1.62)
  4. 排除 11:15-11:30 / 13:00-13:15 / 14:45-15:00 三根 bar
  5. 出场: T+1~T+5 日K收盘 >= 买入价×1.01 即卖, 否则第5日强平 —— **不加止损**
     (实测任何止损都变差: -3% 止损使总盈亏 28.7k → 13.9k)

盘中性能设计:
  - "近5日跌幅"一天内不变 → 每天首次扫描算完后写入 state.json, 后续扫描复用
  - 全市场快照预筛 + 候选池并发拉 15min(不缓存), 单次运行约 30-60 秒
  - state.json 记录已推送信号, 避免同一根 bar 重复推送

用法:
  python live_scout.py                 # 正常盘中扫描 (自动判断交易时段/最新bar)
  python live_scout.py --force         # 强制扫描 (忽略时段, 用于收盘后复盘/测试)
  python live_scout.py --pool=800      # 候选池大小
  python live_scout.py --no-market-filter  # 忽略大盘择时
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
import re
import requests

# ---- 软依赖: akshare 仅作备用回退, 缺失不影响主流程 ----
try:
    import akshare as ak
except Exception:
    ak = None

sys.path.insert(0, str(Path(__file__).parent))
import screener_v2 as S
import push_notify as PN

# 注意: akshare 的 stock_zh_a_spot() 内部用 py_mini_racer 解析新浪行情, 在当前环境会稳定崩溃
# (mini_racer.dll 异常, 无法被 Python 捕获), 且东财接口被网络阻断 → 全部改为纯 requests 直连新浪
SINA_MKT_API = ('http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
                'Market_Center.getHQNodeData')
SINA_HQ = 'http://hq.sinajs.cn/list='
SINA_KLINE_API = ('https://quotes.sina.cn/cn/api/json_v2.php/'
                  'CN_MarketDataService.getKLineData')
# 2026-09-09: web.ifzq.gtimg.cn 会返回 501 (风控重定向), 必须用 ifzq.gtimg.cn
TX_KLINE_API = 'https://ifzq.gtimg.cn/appstock/app/fqkline/get'
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
       'Referer': 'http://vip.stock.finance.sina.com.cn/'}

LIVE_DIR = Path(__file__).parent / 'results_live'
STATE = LIVE_DIR / 'state.json'

# 增强版参数
# 2026-09-15: 用户确认弱势环境(上证3864 < MA20 3924、量比0.85 缩量)下推荐风险大, 重新默认开启大盘择时。
#   开启后: 上证 < MA20 当日直接不出信号 (日志提示"大盘在 MA20 下方 → 今日不出信号"), 规避破位日接飞刀。
#   如需临时对比回测, 可加 --no-market-filter 关闭。
MA20_FILTER = True          # 大盘择时开关 (默认开启: 上证<MA20 则当日不出信号)
DECLINE_A = -3.0            # A级: 近5日累计跌幅 <= -3%
DECLINE_B = -1.0            # B级: -3% < 跌幅 <= -1%
MIN_TURNOVER = 3_000_000    # 预筛最小成交额
TOP_N = 5                   # 每日最多关注/建仓只数 (按同期放量降序)
POOL_SIZE = 600             # 候选池 (按成交额)

# 2026-09-09 (v9): 同期放量门槛 0.9 -> 1.5, 并引入强度分级。
# 依据: 24 个月 / 11469 条信号, 按「今/昨量比」分档呈严格单调 ——
#     0.9~1.2  PF 1.33 净均 +0.61%   (原门槛 0.9 把这档全部放进来, 是主要拖累)
#     1.2~1.6  PF 1.64 净均 +1.03%
#     1.6~2.2  PF 1.90 净均 +1.40%
#     >2.2     PF 2.63 净均 +2.17%
#   口径换算: 实盘 13:30 的「同期口径」≈ 回测「全天口径」x 1.56 (实测中位比值),
#   故实盘 1.5 / 1.8 / 2.3 大致对应回测 0.96 / 1.15 / 1.47。
#   宁缺毋滥: 达不到 1.5 就不买, 不为了凑够 5 只而买入低质量信号。
SAME_VOL_MIN = 0.9          # 同期放量硬门槛 (低于此不出信号); 恢复至周一/v8 水平 0.9× 以提升推送密度
SAME_VOL_MID = 1.8          # ★★ 中等
SAME_VOL_STRONG = 2.3       # ★★★ 强信号 (对应回测全天 1.47, PF ~1.85)
CHASE_LIMIT_PCT = 3.0       # 追高上限: 现价较信号价涨超 3% 提示慎追
DAY_GAIN_MAX = 3.0        # 当日涨幅过滤: 扫描时剔除当日涨幅 > 此值的个股 (已大幅拉升, 不宜追高)
EXCLUDE_STAR_MARKET = True    # 科创板(688xxx)剔除: 用户要求计划任务不含科创板; 手动跑加 --allow-star 放开

# A股 15min bar 的 close-time 标记 (与 screener_v2 一致, 已排除的三根不列入)
BAR_TIMES = [(9, 45), (10, 0), (10, 15), (10, 30), (10, 45), (11, 0), (11, 15),
             (13, 30), (13, 45), (14, 0), (14, 15), (14, 30), (14, 45)]


def now_cst() -> datetime:
    """当前北京时间 (Asia/Shanghai)。

    关键: GitHub Actions / 云端 runner 的系统时钟是 UTC, 若直接用 datetime.now()
    会把 13:30 北京时间误判为 05:30 UTC 的「非交易时段」而直接退出。
    统一用上海时区, 本机(Win 已是中国时区)与云端行为一致。"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo('Asia/Shanghai'))
    except Exception:
        # 极少数环境缺 tzdata: 退化为 UTC+8
        return datetime.now() + timedelta(hours=8)


def log(m):
    print(f"[{now_cst().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------- 数据 (盘中一律不走磁盘缓存) ----------
def fetch_daily_qfq(code: str, days: int = 60) -> pd.DataFrame:
    """个股日K 前复权 — 纯 requests (腾讯主源 / 新浪备用)

    akshare 的日K接口 (stock_zh_a_hist / stock_zh_a_daily) 在多线程下会触发
    py_mini_racer 进程级崩溃 (无法被 Python 捕获), 2026-09-08 实盘复现 → 彻底绕开.
    腾讯源含当日实时数据, 且为前复权, 优于新浪 (只到昨日)."""
    sym = S.market_prefix(code)
    try:
        r = requests.get(TX_KLINE_API, params={'param': f'{sym},day,,,{days},qfq'},
                         headers=HDR, timeout=15)
        d = r.json()['data'][sym]
        k = d.get('qfqday') or d.get('day')
        if k:
            df = pd.DataFrame([x[:6] for x in k],
                              columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            return S._daily_eng_to_zh(df)
    except Exception:
        pass
    try:
        r = requests.get(SINA_KLINE_API, params={'symbol': sym, 'scale': '240',
                                                 'ma': 'no', 'datalen': str(days)},
                         headers=HDR, timeout=15)
        k = json.loads(r.text)
        if k:
            df = pd.DataFrame(k).rename(columns={'day': 'date'})
            return S._daily_eng_to_zh(df)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_index_daily(sym: str = 'sh000001', days: int = 60) -> pd.DataFrame:
    """指数日K — 纯 requests (腾讯), 返回英文列"""
    try:
        r = requests.get(TX_KLINE_API, params={'param': f'{sym},day,,,{days},'},
                         headers=HDR, timeout=15)
        d = r.json()['data'][sym]
        k = d.get('day') or d.get('qfqday')
        if k:
            df = pd.DataFrame([x[:6] for x in k],
                              columns=['date', 'open', 'close', 'high', 'low', 'volume'])
            for c in ('open', 'close', 'high', 'low', 'volume'):
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df
    except Exception:
        pass
    return pd.DataFrame()


def fetch_daily_live(code: str, start: str, end: str) -> pd.DataFrame:
    return fetch_daily_qfq(code)


def fetch_15min_live(code: str, retries: int = 2) -> pd.DataFrame:
    """15min K线 — 纯 requests 直连新浪

    akshare 的 stock_zh_a_minute() 同样依赖 py_mini_racer, 多线程下会导致进程级崩溃
    (mini_racer.dll 异常无法被 Python 捕获), 故一并替换为直连接口.
    返回字段含 amount(成交额), 比 baostock 更完整."""
    sym = S.market_prefix(code)
    for i in range(retries):
        try:
            r = requests.get(SINA_KLINE_API, timeout=20, headers=HDR,
                             params=dict(symbol=sym, scale='15', ma='no', datalen='320'))
            if r.status_code == 200 and r.text.strip() not in ('', 'null'):
                d = json.loads(r.text)
                if d:
                    df = pd.DataFrame(d)
                    df['day'] = pd.to_datetime(df['day'])
                    for c in ('open', 'high', 'low', 'close', 'volume', 'amount'):
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors='coerce')
                    return df
        except Exception:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(0.6)
    return pd.DataFrame()


def sina_page(page: int, num: int = 100) -> list:
    """新浪分页行情 (按成交额降序) — 返回标准 JSON, 不经过 mini_racer"""
    params = dict(page=page, num=num, sort='amount', asc=0, node='hs_a',
                  symbol='', _s_r_a='page')
    for _ in range(3):
        try:
            r = requests.get(SINA_MKT_API, params=params, headers=HDR, timeout=20)
            if r.status_code == 200 and r.text.strip() not in ('', 'null'):
                return json.loads(r.text)
        except Exception:
            time.sleep(1.0)
    return []


def fetch_snapshot(n: int) -> pd.DataFrame:
    """全市场快照: 取成交额前 n 只 (新浪已按成交额降序)"""
    pages = (n + 99) // 100
    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(sina_page, range(1, pages + 1)):
            rows.extend(res)
    if not rows:
        raise RuntimeError('全市场快照获取失败')
    df = pd.DataFrame(rows).rename(columns={
        'code': 'code', 'name': 'name', 'trade': 'price',
        'changepercent': 'pct', 'amount': 'turnover', 'volume': 'volume'})
    df['code'] = df['symbol'].apply(lambda s: s[2:] if str(s).startswith(('sh', 'sz', 'bj')) else str(s))
    for c in ('price', 'pct', 'turnover', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.reset_index(drop=True)


def fetch_names(codes: list) -> dict:
    """批量补查股票名称 (新浪 hq 接口, 每批 80 个)

    用途: 跌幅缓存是当日首次扫描建的, 之后成交额排名会变动,
    导致部分候选股不在当前快照里 → 用代码反查名称, 避免结果显示成代码"""
    out = {}
    syms = [S.market_prefix(c) for c in codes]
    for i in range(0, len(syms), 80):
        batch = syms[i:i + 80]
        try:
            r = requests.get(SINA_HQ + ','.join(batch), headers=HDR, timeout=15)
            r.encoding = 'gbk'
            for line in r.text.strip().split('\n'):
                m = re.match(r'var hq_str_(\w+)="([^"]*)"', line.strip())
                if m and m.group(2):
                    out[m.group(1)[2:]] = m.group(2).split(',')[0]
        except Exception:
            continue
        time.sleep(0.2)
    return out


def sina_index_now() -> float | None:
    """上证指数实时点位 (新浪 hq 接口)
    返回字段顺序: 名称, 最新价, 昨收, 今开, ...
    集合竞价期最新价为 0, 用今开兜底"""
    try:
        r = requests.get(SINA_HQ + 'sh000001', headers=HDR, timeout=10)
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split(',')
            cur = float(parts[1])
            if cur <= 0:
                cur = float(parts[3])      # 集合竞价期: 用今开
            if cur > 0:
                return cur
    except Exception:
        pass
    return None


def sina_stock_now(code: str):
    """单只股票实时价 (新浪 hq). 返回 (当前价, 昨收), 失败返回 (None, None)."""
    if not code:
        return None, None
    prefix = 'sh' if str(code)[0] in '68' else ('sz' if str(code)[0] in '03' else 'bj')
    try:
        r = requests.get(SINA_HQ + prefix + str(code), headers=HDR, timeout=10)
        m = re.search(r'"([^"]+)"', r.text)
        if m:
            parts = m.group(1).split(',')
            if len(parts) >= 4:
                cur = float(parts[3])
                pre = float(parts[2]) if parts[2] not in ('', '0') else None
                if cur > 0:
                    return cur, pre
    except Exception:
        pass
    return None, None


def market_ok() -> tuple[bool, str]:
    """大盘择时: 上证指数当前点位 vs MA20 (必须用实时点位, 日K源只到昨日会误判)"""
    closes = []
    # 1) 历史日K (到昨日)
    hist = fetch_index_daily('sh000001', 60)
    if not hist.empty:
        closes = list(hist['close'].astype(float))
    # 2) 实时点位 (东财指数接口常被限流, 优先新浪实时)
    cur, src = None, ''
    v = sina_index_now()
    if v:
        cur, src = v, '新浪实时'
    if cur is None:
        if closes:
            cur, src = float(closes[-1]), '日K收盘(实时源不可用)'
        else:
            return False, '大盘数据获取失败'
    if len(closes) < 19:
        return False, '大盘历史数据不足'
    ma20 = (sum(closes[-19:]) + cur) / 20.0
    return cur > ma20, f"上证 {cur:.1f} vs MA20 {ma20:.1f} [{src}] → {'可交易' if cur > ma20 else '不交易'}"


def last_completed_bar(now: datetime) -> datetime | None:
    """当前时刻最近一根已完成的 15min bar (close time)"""
    t = now.replace(second=0, microsecond=0)
    for h, m in reversed(BAR_TIMES):
        bt = t.replace(hour=h, minute=m)
        if now >= bt + timedelta(minutes=1):      # bar 完成后 1 分钟才算数 (等数据落库)
            return bt
    return None


# ---------- 信号 ----------
def _compute_bar_metrics(bars, prev_bars, i, ytd) -> dict | None:
    """计算第 i 根 15min bar 的全部量价指标; 防御性检查不通过返回 None。
    不含严格过滤器 —— find_all_signal_bars(取严格信号) 与 best_candidate_bar(取观察池代表bar) 共用。"""
    cur, prev = bars.iloc[i], bars.iloc[i - 1]
    if bool(cur.get('_ex', False)):
        return None
    try:
        op, cl = float(cur['open']), float(cur['close'])
        cv, pv = float(cur['volume']), float(prev['volume'])
    except Exception:
        return None
    # 防御: 当天最新 bar 数据未回填时 OHLC 是 nan, 跳过 (等下次扫描)
    if not (math.isfinite(op) and math.isfinite(cl)
            and math.isfinite(cv) and math.isfinite(pv)):
        return None
    if op <= 0 or pv <= 0 or cv <= 0:
        return None
    chg = (cl - op) / op * 100
    # 量比基线: 用前3根均值, 抵御新浪偶发废bar(单根 V=100 等异常小量)
    baseline = float(bars['volume'].iloc[max(0, i - 3):i].mean())
    if baseline <= 0:
        return None
    vr = cv / baseline
    # 同期放量倍数: 昨日 bar 数不足(停牌/新股)时不做, 避免分母失真造成假信号
    if len(prev_bars) < i + 1:
        return None
    today_cum = float(bars['volume'].iloc[:i + 1].sum())
    prev_cum = float(prev_bars['volume'].iloc[:i + 1].sum())
    if prev_cum <= 0:
        return None
    ytd_same = today_cum / prev_cum
    # 平静度 (用户偏好形态: 前面量能均匀且低迷, 信号根突然放大)
    # calm3_maxmin = 前3根量 最大/最小 (越小越均匀); burst3 = 信号根/前3根均量
    # 仅作标注展示, 不作为过滤 (三月回测: 过滤会损失 27%~53% 收益)
    calm3, burst3 = float('nan'), float('nan')
    if i >= 3:
        vs = []
        for j in range(i - 3, i):
            try:
                vs.append(float(bars.iloc[j]['volume']))
            except Exception:
                vs = []
                break
        if len(vs) == 3 and min(vs) > 0:
            av = sum(vs) / 3
            calm3, burst3 = max(vs) / min(vs), cv / av
    return {'signal_time': cur['day'], 'close': cl, 'change_pct': chg,
            'vol_ratio': vr, 'signal_bar_vol': cv, 'prev_bar_vol': pv,
            'ytd_vol_ratio': ytd, 'ytd_same': ytd_same,
            'calm3_maxmin': calm3, 'burst3': burst3,
            'triggered': (S.INTRADAY_PCT_MIN <= chg <= S.INTRADAY_PCT_MAX
                          and vr >= S.VOL_MULT and ytd_same >= SAME_VOL_MIN)}


def _prepare_days(df15, target_day):
    """取当日/前一日 bar 序列与 全天口径 today/prev 量比 ytd; 非法返回 None。"""
    df = df15.reset_index(drop=True)
    df['_ex'] = df['day'].apply(S.is_excluded_bar)
    days = sorted(df['day'].dt.date.unique())
    if target_day not in days:
        return None
    prev_days = [d for d in days if d < target_day]
    if not prev_days:
        return None
    prev_day = prev_days[-1]
    dv = df.groupby(df['day'].dt.date)['volume'].sum()
    yv = float(dv.get(prev_day, 0))
    tv = float(dv.get(target_day, 0))
    if yv <= 0:
        return None
    ytd = tv / yv          # 全天口径: 盘中 tv 只含已走完的 bar, 天然偏低, 仅作展示
    prev_bars = df[df['day'].dt.date == prev_day].reset_index(drop=True)
    bars = df[df['day'].dt.date == target_day].reset_index(drop=True)
    return bars, prev_bars, ytd


def find_all_signal_bars(df15: pd.DataFrame, target_day: date) -> list[dict]:
    """当日所有严格触发买入信号的 15min bar (盘中可能多根触发)。"""
    prep = _prepare_days(df15, target_day)
    if prep is None:
        return []
    bars, prev_bars, ytd = prep
    out = []
    for i in range(1, len(bars)):
        m = _compute_bar_metrics(bars, prev_bars, i, ytd)
        if m is not None and m['triggered']:
            out.append(m)
    return out


def best_candidate_bar(df15: pd.DataFrame, target_day: date) -> dict | None:
    """观察池代表 bar: 取当日 vol_ratio 最高的那根(最接近触发信号的技术状态)。
    即便未触发严格信号, 也给出最接近信号的技术指标, 供观察池(watchlist)展示。"""
    prep = _prepare_days(df15, target_day)
    if prep is None:
        return None
    bars, prev_bars, ytd = prep
    best = None
    for i in range(1, len(bars)):
        m = _compute_bar_metrics(bars, prev_bars, i, ytd)
        if m is None:
            continue
        if best is None or m['vol_ratio'] > best['vol_ratio']:
            best = m
    return best


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_state(st: dict):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding='utf-8')


def push_no_signal_heartbeat(reason: str, now, st=None) -> bool:
    """无信号时, 每个扫描周期(每半小时, 交易时段内)都推送一次"心跳"告知任务仍在运行。
    - 仅在交易时段 (9:15-15:30 北京) 推送
    - 无信号时每轮都推(不限制每日次数), 让用户知道任务还活着
    返回 True 表示本次实际推送了。"""
    offhours = (now.hour < 9 or (now.hour == 9 and now.minute < 15)
                or now.hour > 15 or (now.hour == 15 and now.minute > 30))
    if offhours:
        return False
    try:
        title = f"无信号提醒 {now.date()} {now:%H:%M}"
        text = (f"今日无个股触发信号。\n原因: {reason}\n\n"
                f"盯盘任务正常运行中 —— 每半小时推送一次心跳，让你知道它还在跑。")
        if PN.push_text(title, text):
            log(f"已推送'无信号'心跳: {reason}")
            return True
    except Exception as e:
        log(f"无信号心跳推送异常(忽略): {e}")
    return False


def push_market_signal(market_msg: str, now, st=None, close_note: bool = False) -> bool:
    """大盘信号单独推送: 每个扫描周期(交易时段)推一条短消息, 让用户掌握市场是否可交易。
    close_note=True 时(收盘附近延迟运行)追加"今日系统已自动运行"提示, 避免 GitHub cron 延迟到收盘后静默。"""
    try:
        title = f"大盘信号 {now.date()} {now:%H:%M}"
        hint = ''
        if '不交易' in market_msg or '下方' in market_msg:
            hint = '\n技术达标个股仍会进入观察池推送，供你自行决定是否关注（不推送买入建议）。'
        text = (f"{market_msg}\n\n"
                f"每半小时播报一次，让你随时知道市场是否可交易。{hint}")
        if close_note:
            text += "\n\n（收盘附近触发 · 今日系统已自动运行，数据截至收盘）"
        if PN.push_text(title, text):
            log(f"已推送大盘信号: {market_msg}" + (" [收盘附近]" if close_note else ""))
            return True
    except Exception as e:
        log(f"大盘信号推送异常(忽略): {e}")
    return False


def push_observation_pool(sigs, market_msg, now, st, market_blocked) -> int:
    """个股观察池推送: 复用原 top5 信号报告的表格格式(HTML), 每周期有候选即推。
    大盘不交易时 obs=True, 标题/提示改为'观察池·仅供参考', 不推送买入建议。"""
    if not sigs:
        return 0
    df = pd.DataFrame(sigs)
    df['code'] = df['code'].astype(str).str.zfill(6)
    # 实时价核对「现价 vs 信号价」偏离(同 save_out, 仅用于展示'现价偏离'列)
    try:
        syms = [('sh' if c[0] == '6' else 'sz') + c for c in df['code']]
        r = requests.get('https://hq.sinajs.cn/list=' + ','.join(syms),
                         headers={'Referer': 'https://finance.sina.com.cn'}, timeout=15)
        r.encoding = 'gbk'
        px = {}
        for line in r.text.strip().split('\n'):
            m = re.match(r'var hq_str_(\w+)="(.*)";', line.strip())
            if not m:
                continue
            f = m.group(2).split(',')
            if len(f) > 3:
                try:
                    v = float(f[3])
                except ValueError:
                    continue
                if v > 0:
                    px[m.group(1)[2:]] = v
        df['now_price'] = df['code'].map(px)
        df['chase_pct'] = ((df['now_price'] - df['close']) / df['close'] * 100).round(2)
    except Exception as e:
        log(f"观察池实时价获取失败(不影响展示): {e}")
        df['chase_pct'] = None
    html = build_live_html(df, now.strftime('%H:%M'), market_msg, obs=market_blocked)
    title = ("个股观察池 · 大盘不交易（仅供参考）" if market_blocked
             else f"盘中信号 {now.date()} · 候选 {len(sigs)} 只")
    try:
        if PN.push_html(title, html):
            log(f"已推送观察池: {len(sigs)} 只" + (" (大盘不交易·仅供参考)" if market_blocked else ""))
            return len(sigs)
    except Exception as e:
        log(f"观察池推送异常(忽略): {e}")
    return 0


# ---------- 主流程 ----------
def _make_row(m, code, name, decline_pct, day_pct, day_key):
    """把指标 dict(或 None) 拼成观察池/信号行所需的完整 dict (grade 由跌幅定级)。"""
    grade = 'A' if decline_pct <= DECLINE_A else 'B'

    def r2(x):
        try:
            return round(float(x), 2)
        except Exception:
            return None

    base = {'code': code, 'name': name, 'date': day_key, 'grade': grade,
            'decline_5d_pct': decline_pct, 'day_pct': day_pct,
            'calm3_maxmin': None, 'burst3': None}
    if m is None:
        base.update({'bar_time': '', 'close': float('nan'),
                     'bar_change_pct': float('nan'), 'vol_ratio': float('nan'),
                     'signal_bar_vol': 0, 'prev_bar_vol': 0,
                     'ytd_vol_ratio': float('nan'), 'ytd_same': float('nan')})
        return base
    base.update({'bar_time': m['signal_time'].strftime('%Y-%m-%d %H:%M'),
                 'close': round(float(m['close']), 3),
                 'bar_change_pct': r2(m['change_pct']),
                 'vol_ratio': r2(m['vol_ratio']),
                 'signal_bar_vol': int(m['signal_bar_vol']),
                 'prev_bar_vol': int(m['prev_bar_vol']),
                 'ytd_vol_ratio': r2(m['ytd_vol_ratio']),
                 'ytd_same': r2(m['ytd_same']),
                 'calm3_maxmin': r2(m.get('calm3_maxmin', float('nan'))),
                 'burst3': r2(m.get('burst3', float('nan')))})
    return base


def scan(args):
    now = now_cst()
    today = now.date()
    S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    st = load_state()

    bar = last_completed_bar(now)
    if bar is None and not args.force:
        log("当前不在有效 15min bar 时段 (或非交易时间), 退出。")
        return [], [], "未检查(非交易时段)", False
    if bar is not None:
        log(f"最新已完成 bar: {bar.strftime('%H:%M')}")

    # 1) 大盘择时
    market_msg = '未启用大盘择时'
    market_blocked = False
    if (MA20_FILTER or args.market_filter) and not args.no_market_filter:
        ok, msg = market_ok()
        log(f"大盘择时: {msg}")
        market_msg = msg
        if not ok:
            market_blocked = True
            log("!! 大盘在 MA20 下方 → 按规则今日不出买入信号 (熊市不开仓)")
            log("   但技术达标候选仍会进入观察池供你自行观察, 不推送买入建议。")
    else:
        log("大盘择时: 已关闭")

    # 2) 全市场快照 + 预筛 (用纯 requests 直连新浪, 绕开 mini_racer)
    log("拉取全市场快照…")
    snap = fetch_snapshot(args.pool + 200)
    snap = snap[~snap['code'].str.startswith(S.POOL_EXCLUDE_PREFIX)]
    snap = snap[~snap['name'].str.contains('ST|退', na=False)]
    # 科创板(688xxx)剔除: 用户要求计划任务不含科创板 (手动跑加 --allow-star 放开)
    if EXCLUDE_STAR_MARKET and not getattr(args, 'allow_star', False):
        n_star = int(snap['code'].astype(str).str.startswith('688').sum())
        snap = snap[~snap['code'].astype(str).str.startswith('688')]
    else:
        n_star = 0
    snap = snap[(snap['turnover'] >= MIN_TURNOVER) & (snap['price'] >= 1.0) &
                (snap['pct'] >= -6)]
    n_over_gain = int((snap['pct'] > DAY_GAIN_MAX).sum())
    snap = snap[snap['pct'] <= DAY_GAIN_MAX]
    snap = snap.sort_values('turnover', ascending=False).head(args.pool).reset_index(drop=True)
    log(f"候选池: {len(snap)} 只 (成交额前 {args.pool}, 已剔除 ST/北交/科创板/极端涨跌/当日涨幅>{DAY_GAIN_MAX}% 的 {n_over_gain} 只)")
    if n_star:
        log(f"  其中科创板(688xxx)剔除 {n_star} 只")

    # 3) 近5日跌幅 (每天算一次, 结果入 state)
    day_key = today.strftime('%Y-%m-%d')
    cached = st.get('decline', {}).get('date') == day_key
    if cached and not args.force:
        decl = st['decline']['data']
        crash = st['decline'].get('crash', {})
        log(f"复用今日已算的近5日跌幅 ({len(decl)} 只)")
    else:
        start = (today - timedelta(days=40)).strftime('%Y-%m-%d')
        end = today.strftime('%Y-%m-%d')
        decl = {}
        crash = {}

        def work(row):
            ddf = fetch_daily_live(row['code'], start, end)
            if ddf.empty:
                return None
            ok, diag = S.check_recent_decline(ddf, today, min_pct=99)  # 先算实际跌幅
            if not diag:
                return None
            crash_flag = S.has_recent_crash(ddf, today,
                                            window=S.CRASH_WINDOW,
                                            consec=S.CRASH_CONSEC,
                                            drop_pct=S.CRASH_DROP_PCT)
            return row['code'], round(diag['cum_pct'], 2), crash_flag

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, row) for _, row in snap.iterrows()]
            for f in as_completed(futs):
                try:
                    r = f.result()
                    if r:
                        decl[r[0]] = r[1]
                        crash[r[0]] = r[2]
                except Exception:
                    pass
        st['decline'] = {'date': day_key, 'data': decl, 'crash': crash}
        save_state(st)
        log(f"近5日累计跌幅计算完成: {len(decl)} 只")
    # 4) 候选: 近5日跌幅 <= DECLINE_B, 且剔除连续暴跌
    #    (最近 CRASH_WINDOW 天内出现 CRASH_CONSEC 个连续交易日, 单日跌幅都 > CRASH_DROP_PCT%)
    decline_pass = [c for c, p in decl.items() if p <= DECLINE_B]
    n_excl_crash = sum(1 for c in decline_pass if crash.get(c, False))
    cands = [(c, p) for c, p in decl.items()
             if p <= DECLINE_B and not crash.get(c, False)]
    cands.sort(key=lambda x: x[1])          # 跌得多的优先
    log(f"符合大前提(近5日跌>{abs(DECLINE_B)}%): {len(decline_pass)} 只, "
        f"剔除连续{S.CRASH_CONSEC}日跌超{S.CRASH_DROP_PCT:.0f}%共 {n_excl_crash} 只 → "
        f"扫描 15min: {len(cands)} 只")

    name_map = {r['code']: r['name'] for _, r in snap.iterrows()}
    pct_map = {r['code']: r['pct'] for _, r in snap.iterrows()}
    _missing = [c for c, _ in cands if c not in name_map]
    if _missing:
        name_map.update(fetch_names(_missing))

    if not cands:
        return [], [], market_msg, market_blocked

    # 5) 拉 15min: 既判严格买入信号(sigs), 也生成观察池候选行(cand_rows)
    sigs = []
    cand_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_15min_live, c): c for c, _ in cands}
        for f in tqdm_as_completed(futs):
            c = futs[f]
            decl = dict(cands)[c]
            dpct = round(pct_map.get(c, 0.0), 2)
            nm = name_map.get(c, c)
            try:
                df15 = f.result()
            except Exception:
                df15 = pd.DataFrame()
            if df15.empty:
                log(f"  {c} 无15min数据, 不列入观察池")
                continue
            # 严格买入信号
            for sg in find_all_signal_bars(df15, today):
                sigs.append(_make_row(sg, c, nm, decl, dpct, day_key))
            # 观察池代表 bar (vol_ratio 最高, 最接近触发); 无有效 bar 则不列入
            bm = best_candidate_bar(df15, today)
            if bm is None:
                log(f"  {c} 15min无有效bar, 不列入观察池")
                continue
            cand_rows.append(_make_row(bm, c, nm, decl, dpct, day_key))
    # 科创板(688xxx)兜底: 候选池与信号都拦
    if EXCLUDE_STAR_MARKET and not getattr(args, 'allow_star', False):
        before = len(sigs)
        sigs = [s for s in sigs if not str(s['code']).startswith('688')]
        if len(sigs) != before:
            log(f"  scan 阶段剔除科创板信号 {before - len(sigs)} 只")
        before_c = len(cand_rows)
        cand_rows = [r for r in cand_rows if not str(r['code']).startswith('688')]
        if len(cand_rows) != before_c:
            log(f"  scan 阶段剔除科创板候选 {before_c - len(cand_rows)} 只")
    # 观察池按 vol_ratio 降序 (最接近信号的排前)
    def _vr(r):
        v = r.get('vol_ratio')
        return v if isinstance(v, (int, float)) and math.isfinite(v) else -1.0
    cand_rows_sorted = sorted(cand_rows, key=_vr, reverse=True)
    log(f"观察池候选 {len(cand_rows_sorted)} 只 (含未触发严格信号的技术达标股)")
    if not sigs:
        log("本轮无个股触发严格买入信号。")
        st['last_run'] = now.strftime('%Y-%m-%d %H:%M')
        save_state(st)
        # 技术达标候选仍在 → 观察池照推; 大盘不交易时由 main 标"仅供参考"
        return cand_rows_sorted, [], market_msg, market_blocked

    # 6) 分级 + 按「同期放量倍数」排序
    #    v8 (2026-09-09): 24 个月 / 11469 条信号回测, 各排序口径下 Top5 组合:
    #      按 今/昨量比        PF 1.80  平均 +1.325%   ← 最优
    #      按 信号根绝对量      PF 1.41  平均 +0.699%
    #      按 近5日跌幅(跌多先) PF 1.38  平均 +0.770%
    #      按 瞬时量比(v7)      PF 1.35  平均 +0.660%
    #    瞬时量比是 bar 间噪音, 全天/同期放量更能反映「资金真的在进场」。
    for s in sigs:
        s['grade'] = 'A' if s['decline_5d_pct'] <= DECLINE_A else 'B'
    sigs.sort(key=lambda s: (-s['vol_ratio'], -s.get('ytd_same', 0)))
    # v9: 同一只票当天可能有多根 bar 触发, 只保留同期放量最强的那一根。
    #     (2026-09-09 湖南黄金 11:00 与 13:30 两根都上榜, 白占掉 TOP5 里 2 个名额)
    _seen, _dedup = set(), []
    for s in sigs:
        if s['code'] in _seen:
            continue
        _seen.add(s['code'])
        _dedup.append(s)
    if len(sigs) != len(_dedup):
        log(f"同票去重: {len(sigs)} -> {len(_dedup)} 只")
    sigs = _dedup
    for i, s in enumerate(sigs, 1):
        s['rank'] = i
        v = s.get('ytd_same') or 0
        s['strength'] = ('★★★' if v >= SAME_VOL_STRONG
                         else '★★' if v >= SAME_VOL_MID else '★')

    # 7) 去重 (同一 code+bar 只推一次)
    pushed = set(st.get('pushed', []))
    fresh, repeat = [], 0
    for s in sigs:
        key = f"{s['code']}@{s['bar_time']}"
        if key in pushed:
            repeat += 1
            continue
        fresh.append(s)
        pushed.add(key)
    # 观察池 = 候选表格(sigs), 由 main 统一以原 top5 报告格式推送, 此处不再单独维护 tracked 字典
    st['pushed'] = sorted(pushed)[-3000:]
    st['last_run'] = now.strftime('%Y-%m-%d %H:%M')
    save_state(st)
    log(f"触发 {len(sigs)} 个信号 (含已推送 {repeat} 个), 本轮新增 {len(fresh)} 个")
    # 观察池(cand_rows_sorted)每周期都推; 严格买入信号(sigs/fresh)仅在有触发时由 main 推
    if market_blocked:
        return cand_rows_sorted, sigs, market_msg, True
    return cand_rows_sorted, fresh, market_msg, False


def tqdm_as_completed(futs):
    """进度条; 缺失 tqdm 时退化为普通迭代, 不报错 (云端 requirements 不含 tqdm 也不影响)。"""
    try:
        from tqdm import tqdm
        for f in tqdm(as_completed(futs), total=len(futs), desc="15min扫描"):
            yield f
    except Exception:
        for f in as_completed(futs):
            yield f


def save_out(sigs: list, snap_time: str, market_msg: str = '', args=None):
    if not sigs:
        return None, None
    today = now_cst().date().strftime('%Y-%m-%d')
    df = pd.DataFrame(sigs)
    # code 统一为 6 位字符串, 否则读回 CSV 时变 int 会导致去重失效、同一只重复入库
    df['code'] = df['code'].astype(str).str.zfill(6)
    f_csv = LIVE_DIR / f"signals_{today}.csv"
    if f_csv.exists():
        old = pd.read_csv(f_csv, dtype={'code': str})
        old['code'] = old['code'].astype(str).str.zfill(6)
        df = pd.concat([old, df]).drop_duplicates(subset=['code', 'bar_time']).reset_index(drop=True)
    # 科创板(688xxx)剔除: 用户要求全部选股/回测/推送均不含科创板 (兜底——防止累计CSV里遗留的科创板行混进来)
    if EXCLUDE_STAR_MARKET:
        n_star_out = int(df['code'].astype(str).str.startswith('688').sum())
        if n_star_out:
            df = df[~df['code'].astype(str).str.startswith('688')].reset_index(drop=True)
            log(f"  输出阶段再剔除科创板(遗留/累计) {n_star_out} 只")
    key = 'vol_ratio' if 'vol_ratio' in df.columns else 'ytd_same'
    df = df.sort_values(key, ascending=False).reset_index(drop=True)

    # v9 追高保护: 用实时价核对「现价 vs 信号价」偏离。
    # 场景: 2026-09-09 湖南黄金 13:30 出信号 26.75, 14:09 已涨停 28.45 (+6.4%),
    #       等看到消息再下单就是追高/买不进。超过 CHASE_LIMIT_PCT 提示慎追。
    try:
        syms = [('sh' if c[0] == '6' else 'sz') + c for c in df['code']]
        r = requests.get('https://hq.sinajs.cn/list=' + ','.join(syms),
                         headers={'Referer': 'https://finance.sina.com.cn'}, timeout=15)
        r.encoding = 'gbk'
        px = {}
        for line in r.text.strip().split('\n'):
            m = re.match(r'var hq_str_(\w+)="(.*)";', line.strip())
            if not m:
                continue
            f = m.group(2).split(',')
            if len(f) > 3:
                try:
                    v = float(f[3])
                except ValueError:
                    continue
                if v > 0:
                    px[m.group(1)[2:]] = v
        df['now_price'] = df['code'].map(px)
        df['chase_pct'] = ((df['now_price'] - df['close']) / df['close'] * 100).round(2)
        df['chase_warn'] = df['chase_pct'].apply(
            lambda v: '慎追' if pd.notna(v) and v >= CHASE_LIMIT_PCT
            else ('' if pd.notna(v) else ''))
    except Exception as e:
        log(f"实时价获取失败(不影响信号): {e}")
        df['now_price'] = None
        df['chase_pct'] = None
        df['chase_warn'] = ''

    if 'rank' in df.columns:
        df = df.drop(columns=['rank'])
    df.insert(0, 'rank', range(1, len(df) + 1))
    df.to_csv(f_csv, index=False, encoding='utf-8-sig')

    # 当日 TOP 清单: 同票只保留同期放量最强的那根 (合并历史后可能重现同票多根 bar)
    f_top5 = LIVE_DIR / f"top5_{today}.csv"
    t5 = df.drop_duplicates(subset=['code'], keep='first').head(TOP_N).copy()
    t5['rank'] = range(1, len(t5) + 1)
    t5.to_csv(f_top5, index=False, encoding='utf-8-sig')

    f_html = LIVE_DIR / f"live_{today}.html"
    f_html.write_text(build_live_html(df, snap_time, market_msg), encoding='utf-8')
    return f_top5, f_html


def build_live_html(df: pd.DataFrame, snap_time: str, market_msg: str = '', obs: bool = False) -> str:
    # 同票只保留同期放量最强的那根, 避免一只票霸占多个名额
    df = df.drop_duplicates(subset=['code'], keep='first').reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        g = r['grade']
        cm = r.get('calm3_maxmin')
        b3 = r.get('burst3')
        cm_s = f"{cm:.2f}" if pd.notna(cm) else '—'
        b3_s = f"{b3:.2f}×" if pd.notna(b3) else '—'
        # 用户偏好形态: 前方量能均匀(calm<=1.5) 且 爆发明显(burst>=2) → 高亮
        calm_hit = (pd.notna(cm) and cm <= 1.5 and pd.notna(b3) and b3 >= 2.0)
        star = ' <span class="star" title="平静蓄势形态">★</span>' if calm_hit else ''
        rows.append(f"""<tr{' class="calm"' if calm_hit else ''}>
<td><span class="g g{g}">{g}</span></td>
<td>{r['bar_time'][11:]}</td>
<td><b>{r['name']}</b><div class="sub">{r['code']}</div></td>
<td class="num">{r['close']}</td>
<td class="num">{r['decline_5d_pct']:+.1f}%</td>
<td class="num">{r.get('day_pct', 0.0):+.1f}%</td>
<td class="num">{r['bar_change_pct']:+.2f}%</td>
<td class="num"><b>{r['vol_ratio']:.2f}×</b></td>
<td class="num"><b>{r.get('ytd_same', 0) if pd.notna(r.get('ytd_same')) else 0:.2f}×</b></td>
<td class="num">{r['signal_bar_vol']/1e4:.0f}万</td>
<td class="num">{r['ytd_vol_ratio']:.2f}</td>
        <td class="num">{cm_s}</td>
<td class="num">{b3_s}{star}</td></tr>""")

    t5_rows = []
    for i, (_, r) in enumerate(df.head(TOP_N).iterrows(), start=1):
        cm = r.get('calm3_maxmin')
        b3 = r.get('burst3')
        cm_s = f"{cm:.2f}" if pd.notna(cm) else '—'
        b3_s = f"{b3:.2f}×" if pd.notna(b3) else '—'
        calm_hit = (pd.notna(cm) and cm <= 1.5 and pd.notna(b3) and b3 >= 2.0)
        star = ' <span class="star" title="平静蓄势形态">★</span>' if calm_hit else ''
        yv = r.get('ytd_same', 0) if pd.notna(r.get('ytd_same')) else 0
        st = ('★★★' if yv >= SAME_VOL_STRONG
              else '★★' if yv >= SAME_VOL_MID else '★')
        cp = r.get('chase_pct')
        if pd.notna(cp):
            chase = (f'<span style="color:#d4352c;font-weight:700">'
                     f'{cp:+.1f}% 慎追</span>' if cp >= CHASE_LIMIT_PCT
                     else f'<span style="color:#8a93a6">{cp:+.1f}%</span>')
        else:
            chase = '—'
        t5_rows.append(f"""<tr>
<td class="rk">{i}</td>
<td><b>{r['name']}</b></td>
<td class="sub">{r['code']}</td>
<td>{r['bar_time'][11:]}</td>
<td class="num"><b>{r['close']}</b></td>
<td class="num">{r['decline_5d_pct']:+.1f}%</td>
<td class="num">{r.get('day_pct', 0.0):+.1f}%</td>
<td class="num"><b>{r['vol_ratio']:.2f}×</b></td>
<td class="num"><b>{yv:.2f}×</b></td>
<td class="num"><b>{st}</b></td>
<td class="num">{chase}</td>
<td class="num">{cm_s}</td>
<td class="num">{b3_s}{star}</td></tr>""")

    _h1 = ("个股观察池 · " if obs else "盘中实时信号 · ") + f"{now_cst().date()}（更新 {snap_time}）"
    _top5h = (f"观察池 TOP {TOP_N}　技术达标候选（截至 {snap_time}；非买入建议，仅供参考）"
              if obs else
              f"今日 TOP {TOP_N}　按瞬时量比降序　（截至 {snap_time}；同期放量 &lt; {SAME_VOL_MIN} 已全部过滤）")
    _tip_ops = ("⚠ 大盘不交易，以下为技术达标候选，<b>仅供参考、非买入建议</b>；买不买由你自行决定。"
                if obs else
                "操作：信号根收盘价买入，每只 1 万元；T+1 起 5 个交易日内日K收盘 ≥ 买入价 ×1.05 即卖，第 5 日收盘强平；<b>不加止损</b>。")
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>盘中信号 {now_cst().date()} </title><style>
body{{font-family:"Microsoft YaHei",sans-serif;background:#f5f6f8;color:#1f2430;margin:0;padding:24px}}
.wrap{{max-width:1100px;margin:0 auto;overflow-x:auto}}
h1{{font-size:20px}} .meta{{color:#8a93a6;font-size:13px;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}
th,td{{border:1px solid #e6e8ef;padding:4px 12px;text-align:center;white-space:nowrap;line-height:1.5}}
th{{background:#f0f2f7}} td.num{{font-variant-numeric:tabular-nums}}
.sub{{color:#a0a6b5;font-size:11px}}
.g{{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-weight:700;font-size:12px}}
.gA{{background:#d4352c}} .gB{{background:#e8a33d}}
.rule{{background:#fff;border:1px solid #e6e8ef;border-radius:10px;padding:12px 16px;font-size:13px;line-height:1.9;margin-bottom:14px}}
.mkt{{padding:8px 14px;border-radius:8px;font-size:13px;margin-bottom:12px;font-weight:600}}
.mkt.warn{{background:#fdeaea;color:#c0392b;border:1px solid #f5c6c6}}
.mkt.ok{{background:#eaf7ef;color:#0a8f4e;border:1px solid #bfe6cf}}
tr.calm{{background:#fffbe8}}
.star{{color:#d4a017;font-weight:700}}
.top5{{background:#fff;border:2px solid #d4352c;border-radius:10px;padding:14px 16px;margin-bottom:16px}}
.top5h{{font-size:15px;font-weight:700;color:#d4352c;margin-bottom:10px}}
table.t5{{font-size:14px}}
table.t5 th{{background:#fdeaea;color:#8f2019}}
table.t5 td.rk{{font-weight:700;color:#d4352c;font-size:15px}}
.tip{{margin-top:9px;font-size:12px;color:#5f6675;line-height:1.7}}
</style></head><body><div class="wrap">
<h1>{_h1}</h1>
<div class="meta">v8：近5日累计下跌 ＋ 15min 放量脉冲（量≥前根×2，涨幅0~2%）＋ 同期放量 ≥0.9×，按瞬时量比降序<br>出场：T+1 起 5 日内日K收盘 ≥ 买入价 ×1.05 止盈，第 5 日强平，<b>不加止损</b></div>
<div class="mkt {'warn' if ('不交易' in market_msg or '下方' in market_msg) else 'ok'}">大盘状态：{market_msg}{'　⚠ 按规则今日不宜开仓，以下信号仅供参考' if ('不交易' in market_msg or '下方' in market_msg) else ''}</div>
<div class="top5">
<div class="top5h">{_top5h}</div>
<table class="t5"><tr><th>#</th><th>名称</th><th>代码</th><th>触发bar</th><th>信号价</th><th>近5日跌</th><th>当日涨幅</th><th>瞬时量比</th><th>同期放量</th><th>强度</th><th>现价偏离</th><th>平静度</th><th>爆发</th></tr>
{''.join(t5_rows)}</table>
<div class="tip">{_tip_ops}<br>
强度：★★★ ≥{SAME_VOL_STRONG}×（回测 PF ~2.1）　★★ ≥{SAME_VOL_MID}×　★ ≥{SAME_VOL_MIN}×。<b>宁缺毋滥</b>——不足 {TOP_N} 只即视为不达标，不要为凑数。<br>
<b>现价偏离</b>标红「慎追」= 现价已比信号价高 {CHASE_LIMIT_PCT}% 以上，此时下单等同于追高，建议放弃或等回落。</div>
</div>
<div class="rule">
<b>A级</b>：近5日跌幅 ≤ -3%　<b>B级</b>：-3% &lt; 跌幅 ≤ -1%　（跌幅不含信号当天盘中）
<br><b>排序</b>：按<b>瞬时量比</b>降序（复刻周一推送；v8 同期放量排序回测 PF 1.80，与量比 1.35 接近）
<br><b>建议操作</b>：信号根收盘价买入，每只1万元；T+1~T+5 内日K收盘 ≥ 买入价×1.05 即卖出，否则第5日收盘强平；<b>不加止损</b>（实测止损反而更差）
<br><b>★ 平静蓄势形态</b>：前3根量能均匀（平静度 ≤1.5）且信号根放大 ≥2×，为你偏好的形态，仅作标注不做过滤
<br><b>当日涨幅过滤</b>：扫描时已剔除当日涨幅 &gt; {DAY_GAIN_MAX}% 的个股（已大幅拉升、不宜追高）
</div>
<table><tr><th>级别</th><th>触发bar</th><th>名称</th><th>信号价</th><th>近5日跌幅</th><th>当日涨幅</th><th>当根涨幅</th><th>瞬时量比</th><th>同期放量</th><th>信号根量</th><th>今/昨量(全天)</th><th>平静度</th><th>爆发</th></tr>
{''.join(rows)}</table>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=POOL_SIZE)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--force', action='store_true', help='强制扫描(忽略交易时段/复用缓存)')
    ap.add_argument('--no-market-filter', action='store_true')
    ap.add_argument('--market-filter', action='store_true',
                    help='强制启用上证>MA20 择时 (v8 默认已关闭)')
    ap.add_argument('--top', type=int, default=5, help='提示: 每日建议买入前 N 只')
    ap.add_argument('--allow-push-offhours', action='store_true',
                    help='允许非交易时段也推送微信(默认非交易时段静默, 防盘后延迟运行误推)')
    args = ap.parse_args()

    # 错开触发时间: 半点/整点触发后, 等15分钟BAR走完再扫 (延后约5分钟)
    # 放在最前面, 使后续的 scan / 大盘信号 / 跟踪都在 BAR 走完之后执行
    _n0 = now_cst()
    _in_trading = (9 <= _n0.hour < 15) or (_n0.hour == 15 and _n0.minute <= 5)
    if _in_trading:
        _base = (_n0.minute // 30) * 30          # 0 或 30
        _target = _n0.replace(minute=_base, second=0, microsecond=0) + timedelta(minutes=5)
        if _n0 < _target:
            _wait = int((_target - _n0).total_seconds())
            log(f"触发于 {_n0.strftime('%H:%M')}, 延后 {_wait}s 至 {_target.strftime('%H:%M')} 待15分钟BAR走完再扫描")
            time.sleep(min(_wait, 360))          # 最多睡6分钟, 防 runner 超时

    log("== 盘中实时盯盘 v7 ==")
    t0 = time.time()
    obs_pool, sigs, market_msg, market_blocked = scan(args)
    st = load_state()                     # 供有信号/无信号两分支共享 (记录 last_push_date 等)
    now = now_cst()
    offhours = (now.hour < 9 or (now.hour == 9 and now.minute < 15)
                or now.hour > 15 or (now.hour == 15 and now.minute > 30))
    # 收盘附近窗口(15:31-16:30): GitHub 免费 cron 常延迟到收盘后才触发, 此窗口仍补一条"大盘信号+今日已运行"确认, 避免静默
    close_digest = (now.hour == 15 and now.minute > 30) or (now.hour == 16 and now.minute <= 30)
    # 1) 大盘信号单独推送 (每周期, 交易时段; 收盘附近也补一条确认)
    if (not offhours) or close_digest:
        push_market_signal(market_msg, now, st, close_note=close_digest)
    # 2) 个股观察池推送 (每周期有技术达标候选即推; 大盘不交易时标"仅供参考")
    if not offhours and obs_pool:
        push_observation_pool(obs_pool, market_msg, now, st, market_blocked)
    elif not obs_pool:
        log("观察池为空（无技术达标候选），本轮静默。")
    if sigs:
        f_top5, f_html = save_out(sigs, now_cst().strftime('%H:%M'), market_msg, args)
        if f_top5:
            ad = pd.read_csv(f_top5)
            ad['code'] = ad['code'].astype(str).str.zfill(6)
            cols = ['bar_time', 'name', 'code', 'close', 'decline_5d_pct', 'vol_ratio']
            names = ['触发bar', '名称', '代码', '信号价', '近5日跌', '量比']
            if 'ytd_same' in ad.columns:
                cols.append('ytd_same'); names.append('同期放量')
            cols += ['calm3_maxmin', 'burst3']; names += ['平静度', '爆发']
            show = ad[cols].copy(); show.columns = names
            show.insert(0, '#', range(1, len(show) + 1))
            yv = ad['vol_ratio'].fillna(0)
            show['强度'] = yv.apply(lambda v: '★★★' if v >= SAME_VOL_STRONG
                                    else '★★' if v >= SAME_VOL_MID else '★').values
            if 'chase_pct' in ad.columns:
                show['现价偏离'] = ad['chase_pct'].apply(
                    lambda v: f'{v:+.1f}% 慎追' if pd.notna(v) and v >= CHASE_LIMIT_PCT
                    else (f'{v:+.1f}%' if pd.notna(v) else '—')).values
            print("\n" + "=" * 96)
            print(f"  Top{len(ad)} 严格买入信号（按量比降序）" + ("  [大盘不交易·仅供参考]" if market_blocked else ""))
            print(f"  同期放量门槛 {SAME_VOL_MIN}×｜不足 {TOP_N} 只即视为不达标，不要凑数")
            print("=" * 96)
            print(show.to_string(index=False))
            print()
            log(f"TOP{TOP_N} 清单: {f_top5}")
            log(f"完整报告: {f_html}")
        # 严格买入信号 → 推送 Top5 报告 (大盘不交易时不推买入建议, 仅观察池标仅供参考)
        if not market_blocked:
            try:
                html = Path(f_html).read_text(encoding='utf-8')
                title = f"盘中买入信号 {now.date()} · Top{TOP_N}"
                if PN.push_html(title, html):
                    log(f"已推送 Top{TOP_N} 买入信号报告")
            except Exception as e:
                log(f"Top{TOP_N} 报告推送异常(忽略): {e}")
    else:
        log("本轮无严格买入信号，仅推送技术达标观察池。")
    log(f"耗时 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
