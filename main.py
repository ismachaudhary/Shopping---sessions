import json
import os
import sys
from datetime import datetime, timezone

import duckdb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUT_DIR", "/output")
DAY = 86400000
WEEK = 7 * DAY


def iso_ms(s):
    d = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(d.timestamp()) * 1000


man = json.load(open(f"{DATA}/manifest.json"))
WS = iso_ms(man["window_start"])
WE = iso_ms(man["window_end"])

con = duckdb.connect()
os.makedirs("/tmp/duck", exist_ok=True)
con.execute("PRAGMA temp_directory='/tmp/duck'")

FIELD = '(?:"[^",]*"|[^",]*)'


def line_re(n):
    return "^" + FIELD + "(?:," + FIELD + "){" + str(n - 1) + "}$"


# ---- macros -------------------------------------------------------------
con.execute(
    r"""
CREATE MACRO uq(s) AS nullif(
  CASE WHEN length(s) >= 2 AND starts_with(s, '"') AND ends_with(s, '"')
       THEN substr(s, 2, length(s) - 2) ELSE s END, '');
CREATE MACRO ts_ms(s) AS CASE
  WHEN s IS NULL THEN NULL
  WHEN regexp_full_match(s, '[0-9]+') THEN try_cast(s AS BIGINT)
  WHEN regexp_full_match(s, '[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z')
    THEN epoch_ms(try_strptime(substr(s, 1, 19), '%Y-%m-%dT%H:%M:%S'))
  WHEN regexp_full_match(s, '[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9][+-]([01][0-9]|2[0-3]):[0-5][0-9]')
    THEN epoch_ms(try_strptime(substr(s, 1, 19), '%Y-%m-%dT%H:%M:%S'))
         - (CASE WHEN substr(s, 20, 1) = '-' THEN -1 ELSE 1 END)
           * (cast(substr(s, 21, 2) AS BIGINT) * 60 + cast(substr(s, 24, 2) AS BIGINT)) * 60000
  ELSE NULL END;
"""
)


import glob
import mmap


def has_cr(kind):
    for f in glob.glob(f"{DATA}/{kind}/*.csv"):
        try:
            with open(f, "rb") as fh:
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    if mm.find(b"\r") >= 0:
                        return True
                finally:
                    mm.close()
        except ValueError:  # empty file
            continue
    return False


def raw(kind):
    if has_cr(kind):
        # slow exact path: DuckDB's CSV reader would swallow a trailing \r
        return (
            f"(SELECT unnest(list_slice(string_split(content, chr(10)), 2, -1)) AS line "
            f"FROM read_text('{DATA}/{kind}/*.csv'))"
        )
    return (
        f"read_csv('{DATA}/{kind}/*.csv', columns={{'line':'VARCHAR'}}, "
        "delim='\x01', quote='', escape='', header=true, new_line='\\n', "
        "auto_detect=false, strict_mode=false, max_line_size=8388608)"
    )


# ---- links -> persons ------------------------------------------------------
con.execute(
    f"""
CREATE TEMP TABLE edges AS
SELECT a, b FROM (
  SELECT uq(f[1]) AS a, uq(f[2]) AS b FROM (
    SELECT string_split(line, ',') AS f FROM {raw('links')}
    WHERE regexp_full_match(line, '{line_re(2)}'))
) WHERE a IS NOT NULL AND b IS NOT NULL
"""
)
con.execute(
    """
CREATE TEMP TABLE lids AS
SELECT id, row_number() OVER () - 1 AS idx FROM (
  SELECT a AS id FROM edges UNION SELECT b FROM edges)
"""
)
n = con.execute("SELECT count(*) FROM lids").fetchone()[0]
if n > 0:
    arr = con.execute(
        "SELECT la.idx, lb.idx FROM edges e JOIN lids la ON e.a = la.id JOIN lids lb ON e.b = lb.id"
    ).fetchnumpy()
    ia = arr[list(arr.keys())[0]].astype(np.int64)
    ib = arr[list(arr.keys())[1]].astype(np.int64)
    g = coo_matrix((np.ones(len(ia), dtype=np.int8), (ia, ib)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    import pandas as pd  # noqa

    comp = pd.DataFrame({"idx": np.arange(n, dtype=np.int64), "comp": labels.astype(np.int64)})
    con.register("comp_df", comp)
    con.execute(
        """
    CREATE TEMP TABLE idmap AS
    SELECT l.id AS id, min(l.id) OVER (PARTITION BY c.comp) AS person
    FROM lids l JOIN comp_df c ON l.idx = c.idx
    """
    )
else:
    con.execute("CREATE TEMP TABLE idmap (id VARCHAR, person VARCHAR)")

# ---- products / campaigns ---------------------------------------------------
con.execute(
    f"""
CREATE TEMP TABLE prod AS
SELECT product_id, any_value(category) AS category FROM read_csv('{DATA}/products.csv',
  all_varchar=true, header=true) GROUP BY product_id;
CREATE TEMP TABLE camp AS
SELECT campaign_id, any_value(channel) AS channel FROM read_csv('{DATA}/campaigns.csv',
  all_varchar=true, header=true) GROUP BY campaign_id;
"""
)

# ---- events -------------------------------------------------------------------
con.execute(
    f"""
CREATE TEMP TABLE ev_dedup AS
SELECT event_id, arg_min(rec, ingest_ts) AS r FROM (
  SELECT uq(f[1]) AS event_id, uq(f[4]) AS ingest_ts,
         struct_pack(u := uq(f[2]), ts := uq(f[3]), it := uq(f[4]), et := uq(f[5]),
                     p := uq(f[6]), pr := uq(f[7])) AS rec
  FROM (SELECT string_split(line, ',') AS f FROM {raw('events')}
        WHERE regexp_full_match(line, '{line_re(7)}'))
) WHERE event_id IS NOT NULL
GROUP BY event_id
"""
)
con.execute(
    f"""
CREATE TEMP TABLE ev_ok AS
SELECT coalesce(m.person, d.u) AS person, d.ts AS ts, d.et AS et, d.p AS p, d.pr AS pr
FROM (
  SELECT r.u AS u, ts_ms(r.ts) AS ts, ts_ms(r.it) AS its, r.et AS et, r.p AS p,
         CASE WHEN r.et = 'purchase' THEN try_cast(r.pr AS BIGINT) END AS pr,
         regexp_full_match(coalesce(r.pr, ''), '[0-9]+') AS prdig
  FROM ev_dedup
) d LEFT JOIN idmap m ON d.u = m.id
WHERE d.u IS NOT NULL AND d.ts >= {WS} AND d.ts < {WE}
  AND d.et IN ('view', 'cart', 'purchase') AND d.p IS NOT NULL
  AND (d.et <> 'purchase' OR (d.prdig AND d.pr BETWEEN 1 AND 10000000))
  AND d.its - d.ts <= {DAY}
"""
)
con.execute("DROP TABLE ev_dedup")

con.execute(
    """
CREATE TEMP TABLE ev2 AS
SELECT person, ts, et, p, pr,
  sum(CASE WHEN prev IS NULL OR ts - prev > 1800000 THEN 1 ELSE 0 END)
    OVER (PARTITION BY person ORDER BY ts) AS sn
FROM (SELECT *, lag(ts) OVER (PARTITION BY person ORDER BY ts) AS prev FROM ev_ok)
"""
)
con.execute("DROP TABLE ev_ok")

con.execute(
    """
CREATE TEMP TABLE sess AS
SELECT person, sn, min(ts) AS st, count(*) AS n_ev,
       coalesce(sum(pr) FILTER (WHERE et = 'purchase'), 0) AS rev,
       count(*) FILTER (WHERE et = 'purchase') AS n_pur,
       cast(make_timestamp(min(ts) * 1000) AS DATE) AS day
FROM ev2 GROUP BY person, sn
"""
)
con.execute(
    """
CREATE TEMP TABLE pcat AS
SELECT x.person, s.day, x.cat, sum(x.v) AS v FROM (
  SELECT person, sn, coalesce(pd.category, 'unknown') AS cat, sum(pr) AS v
  FROM ev2 e LEFT JOIN prod pd ON e.p = pd.product_id
  WHERE e.et = 'purchase' GROUP BY person, sn, cat) x
JOIN sess s ON x.person = s.person AND x.sn = s.sn
GROUP BY x.person, s.day, x.cat
"""
)
con.execute("DROP TABLE ev2")

# ---- clicks ---------------------------------------------------------------------
con.execute(
    f"""
CREATE TEMP TABLE clk AS
SELECT coalesce(m.person, c.u) AS person, c.ts AS ts, c.cid AS cid, c.camp AS camp
FROM (
  SELECT click_id AS cid, any_value(u) AS u, any_value(ts_ms(t)) AS ts, any_value(camp) AS camp
  FROM (
    SELECT uq(f[1]) AS click_id, uq(f[2]) AS u, uq(f[3]) AS camp, uq(f[4]) AS t
    FROM (SELECT string_split(line, ',') AS f FROM {raw('clicks')}
          WHERE regexp_full_match(line, '{line_re(4)}'))
  ) WHERE click_id IS NOT NULL GROUP BY click_id
) c LEFT JOIN idmap m ON c.u = m.id
WHERE c.u IS NOT NULL AND c.ts >= {WS - WEEK} AND c.ts < {WE}
  AND c.camp IN (SELECT campaign_id FROM camp)
"""
)
# one click per (person, ts): smallest click_id
con.execute(
    """
CREATE TEMP TABLE clk1 AS
SELECT person, ts, min(cid) AS cid, min_by(camp, cid) AS camp FROM clk GROUP BY person, ts
"""
)

# credited campaign
con.execute(
    f"""
CREATE TEMP TABLE credit AS
SELECT s.person, s.sn, s.day, s.rev, s.n_pur,
       CASE WHEN c.ts IS NOT NULL AND s.st - c.ts <= {WEEK} THEN c.camp ELSE 'organic' END AS camp
FROM sess s ASOF LEFT JOIN clk1 c ON s.person = c.person AND s.st >= c.ts
"""
)

# assists: union of session-rank ranges per (person, campaign)
con.execute(
    f"""
CREATE TEMP TABLE rng AS
SELECT k.person, k.camp, lo.r_lo AS lo, hi.r_hi AS hi FROM
  (SELECT DISTINCT person, camp, ts FROM clk) k
  ASOF LEFT JOIN (SELECT person, st, sn + 1 AS r_lo FROM sess) lo
    ON k.person = lo.person AND k.ts > lo.st
  ASOF LEFT JOIN (SELECT person, st, sn AS r_hi FROM sess) hi
    ON k.person = hi.person AND k.ts + {WEEK} >= hi.st
"""
)
con.execute(
    """
CREATE TEMP TABLE rng2 AS
SELECT person, camp, coalesce(lo, 1) AS lo, hi FROM rng WHERE hi IS NOT NULL AND hi >= coalesce(lo, 1)
"""
)
con.execute(
    """
CREATE TEMP TABLE isl AS
SELECT person, camp, min(lo) AS a, max(hi) AS b FROM (
  SELECT *, sum(CASE WHEN ph IS NULL OR lo > ph THEN 1 ELSE 0 END)
        OVER (PARTITION BY person, camp ORDER BY lo, hi) AS g
  FROM (SELECT *, max(hi) OVER (PARTITION BY person, camp ORDER BY lo, hi
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS ph FROM rng2))
GROUP BY person, camp, g
"""
)
con.execute(
    """
CREATE TEMP TABLE pdays AS
SELECT person, day, min(sn) AS f, max(sn) AS l FROM sess GROUP BY person, day
"""
)
con.execute(
    """
CREATE TEMP TABLE assist AS
SELECT i.camp, d.day, sum(least(i.b, d.l) - greatest(i.a, d.f) + 1) AS n
FROM isl i JOIN pdays d ON i.person = d.person AND i.a <= d.l AND i.b >= d.f
GROUP BY i.camp, d.day
"""
)

# ---- outputs --------------------------------------------------------------------
os.makedirs(f"{OUT}/user_daily", exist_ok=True)
os.makedirs(f"{OUT}/campaign_daily", exist_ok=True)

con.execute(
    f"""
COPY (
  SELECT s.person AS user_id, s.day AS day, count(*)::BIGINT AS sessions, sum(s.n_ev)::BIGINT AS events,
         sum(s.rev)::BIGINT AS revenue_cents, any_value(t.cat) AS top_category
  FROM sess s
  LEFT JOIN (SELECT person, day, min_by(cat, (-v, cat)) AS cat FROM pcat GROUP BY person, day) t
    ON s.person = t.person AND s.day = t.day
  GROUP BY s.person, s.day
) TO '{OUT}/user_daily/part-0.parquet' (FORMAT parquet)
"""
)
con.execute(
    f"""
COPY (
  SELECT coalesce(c.camp, a.camp) AS campaign_id,
         CASE WHEN coalesce(c.camp, a.camp) = 'organic' THEN 'organic' ELSE cp.channel END AS channel,
         coalesce(c.day, a.day) AS day,
         coalesce(c.sessions, 0)::BIGINT AS sessions,
         coalesce(c.psess, 0)::BIGINT AS purchasing_sessions,
         coalesce(c.revenue, 0)::BIGINT AS revenue_cents,
         coalesce(a.n, 0)::BIGINT AS assisted_sessions
  FROM (SELECT camp, day, count(*) AS sessions, count(*) FILTER (WHERE n_pur > 0) AS psess,
               sum(rev) AS revenue FROM credit GROUP BY camp, day) c
  FULL JOIN assist a ON c.camp = a.camp AND c.day = a.day
  LEFT JOIN camp cp ON cp.campaign_id = coalesce(c.camp, a.camp)
) TO '{OUT}/campaign_daily/part-0.parquet' (FORMAT parquet)
"""
)
