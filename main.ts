// gta6-trophy-rarity-forecast-01 — a structural quantile forecaster for GTA VI's
// PlayStation 5 trophy earn rates on Exophase, written for the make.fun bounty.
//
// The contest scores, per trophy and per quantile level, the pinball loss on the logit
// scale, so both where the median sits and how honest the spread is get paid for.
//
// The one idea this forecast is built on: an earn rate is a ratio of two things that
// move on different clocks.
//
//   rate = holders / tracked
//
// `tracked` is the population Exophase counts for the game. It moves in steps: Exophase
// re-reads a game's counts on its own schedule, hours apart, and each refresh arrives as
// a jump (launch night: 638 tracked for four hours, then 3,868 in one step). `holders` is
// how many of those players hold the trophy, and it moves smoothly, upward, as players
// play. A refresh therefore dilutes every rate in a single step — every rate but a
// saturated 100% one falls — while between refreshes the same rates creep up.
//
// So the forecast does not extrapolate rates. It rebuilds the two factors from every read
// the case ships (each read carries both the rate and the tracked count, so holders is
// exactly rate x tracked / 100), forecasts each factor on its own clock, and divides at
// the end. That way a refresh already sitting inside the history is information — it
// shows up as a step in `tracked` and cancels out of `holders` — instead of being an
// outlier that drags a rate trend down.
//
// Where the history never saw a refresh (the practice cases: one flat launch-night page),
// there is no step to learn from, so the program falls back to the cadence and step size
// the kit documents for launch night, and dilutes the rates once if that refresh is due
// before the target.
//
// Standard library only, offline, deterministic: the case file is the only input.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's two-sided quantiles at those levels. */
const Z1 = 1.6448536269514722;
const Z2 = 0.6744897501960817;
/** The release, the origin every case's elapsed-time axis shares. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** How much of a fitted log-log move survives extrapolation past the last read. */
const DAMP_HOLDERS = 0.9;
const DAMP_TRACKED = 0.75;
/** Shrinkage of a trophy's own holder trend toward its neighbours' (in reads). */
const SHRINK_K = 5;
/** Spread, in logit units: per sqrt of the extrapolated log-time, and a refresh's own. */
const DRIFT_SIGMA = 0.55;
const REFRESH_SIGMA = 0.5;
/** Launch night's documented first refresh, used only when the history holds no step. */
const PRIOR_REFRESH_GROWTH = 3868 / 638;
/** The dilution one launch-night refresh leaves on the rates themselves: the tracked
 *  population multiplies by 6.06 while the trophies already earned grow only ~1.46x, so a
 *  rate falls to about a quarter. Calibrated on the documented baseline gate (the house
 *  baseline's practice loss of 1.51 lands at this factor). */
const PRIOR_DILUTION = 0.24;
/** The page's own bounds. A rate at the 100% ceiling carries no logit. */
const CEILING = 99.995;
const Q_MAX = 99.99;
const Q_MIN = 0.01;

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(Q_MAX, Math.max(Q_MIN, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (x: number): number => 100 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? (s[m] as number) : ((s[m - 1] as number) + (s[m] as number)) / 2;
}

/** Hours since the release, floored so the log axis stays finite on launch night. */
const hoursOf = (iso: string): number => Math.max(0.25, (Date.parse(iso) - RELEASE_MS) / 3_600_000);

/**
 * Theil-Sen slope: the median of every pair's slope. Half the pairs can be rubbish before
 * the estimate moves, which is what one refresh step is — a single violent jump that
 * would drag a least-squares line with it.
 */
function theilSen(xs: number[], ys: number[]): number {
  const slopes: number[] = [];
  for (let i = 0; i < xs.length; i += 1) {
    for (let j = i + 1; j < xs.length; j += 1) {
      const dx = (xs[j] as number) - (xs[i] as number);
      if (Math.abs(dx) < 1e-9) continue;
      slopes.push(((ys[j] as number) - (ys[i] as number)) / dx);
    }
  }
  return median(slopes);
}

/** The distinct tracked-population levels the history shows, each at its first read. */
function trackedLevels(history: Read[]): Array<{ hours: number; n: number }> {
  const levels: Array<{ hours: number; n: number }> = [];
  for (const r of history) {
    const n = r.players_tracked;
    if (!Number.isFinite(n) || n <= 0) continue;
    const top = levels[levels.length - 1];
    if (top === undefined || top.n !== n) levels.push({ hours: hoursOf(r.at), n });
  }
  return levels;
}

/** One trophy's holder series and the robust trend through it. */
interface HolderFit {
  slope: number;
  n: number;
  scatter: number;
  lastHolders: number;
  lastRate: number;
}

function fitHolders(c: Case, index: number): HolderFit {
  const xs: number[] = [];
  const ys: number[] = [];
  const last = c.history[c.history.length - 1] as Read;
  for (const read of c.history) {
    const rate = read.rates[index];
    if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
    if (rate >= CEILING) continue; // a saturated trophy says nothing about growth
    if (rate <= 0) continue;
    const holders = (rate / 100) * read.players_tracked;
    if (!(holders > 0)) continue;
    xs.push(Math.log(hoursOf(read.at)));
    ys.push(Math.log(holders));
  }
  const lastRate = last.rates[index] ?? 0;
  const lastHolders = lastRate > 0 && lastRate < CEILING ? (lastRate / 100) * last.players_tracked : 0;
  const fit: HolderFit = { slope: 0, n: xs.length, scatter: 0, lastHolders, lastRate };
  if (xs.length >= 2) {
    fit.slope = theilSen(xs, ys);
    const anchorX = Math.log(hoursOf(last.at));
    const anchorY = ys[ys.length - 1] as number;
    const res = ys.map((y, i) => Math.abs(y - (anchorY + fit.slope * ((xs[i] as number) - anchorX))));
    // Residuals live in log-holders; halve them into log-rate, which is the scoring axis.
    fit.scatter = 1.4826 * median(res) * 0.5;
  }
  return fit;
}

/** The band a trophy is shrunk toward: its tier crossed with how rare it is now. */
function bandOf(t: Trophy, rate: number): string {
  const band = rate >= 50 ? 'common' : rate >= 10 ? 'mid' : rate >= 1 ? 'thin' : 'rare';
  return `${t.tier}:${band}`;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const last = c.history[c.history.length - 1] as Read;
  const lastHours = hoursOf(last.at);
  const targetHours = hoursOf(c.target);
  const logRatio = Math.log(targetHours / lastHours);
  const fits = c.trophies.map((_, i) => fitHolders(c, i));

  // --- the population's own clock -------------------------------------------------
  const levels = trackedLevels(c.history);
  // No step anywhere in the history means the page was never re-read while it was being
  // watched: there is nothing to learn about refreshes from, so the documented launch-night
  // dilution stands in for one.
  const noStep = levels.length < 2;
  let trackedTarget = last.players_tracked;
  if (!noStep) {
    const xs = levels.map((l) => Math.log(l.hours));
    const ys = levels.map((l) => Math.log(l.n));
    const slope = theilSen(xs, ys);
    trackedTarget = Math.exp(Math.log(last.players_tracked) + DAMP_TRACKED * slope * logRatio);
  }
  // A population cannot shrink, and its growth over a short horizon is bounded by how
  // fast Exophase has been adding players.
  trackedTarget = Math.max(trackedTarget, last.players_tracked);

  // --- each trophy's holders, shrunk toward its band -------------------------------
  const byBand = new Map<string, number[]>();
  c.trophies.forEach((t, i) => {
    const list = byBand.get(bandOf(t, last.rates[i] ?? 0)) ?? [];
    list.push((fits[i] as HolderFit).slope);
    byBand.set(bandOf(t, last.rates[i] ?? 0), list);
  });
  const bandSlope = new Map<string, number>();
  for (const [key, slopes] of byBand) bandSlope.set(key, median(slopes));

  return c.trophies.map((t, i) => {
    const fit = fits[i] as HolderFit;
    const raw = last.rates[i] ?? 0;

    // A saturated trophy: every tracked player holds it, so it answers 100 whatever the
    // population does. Its rate cannot exceed the ceiling the page prints.
    if (raw >= CEILING) return { id: t.id, q: LEVELS.map(() => Q_MAX) };

    const prior = bandSlope.get(bandOf(t, raw)) ?? 0;
    const weight = fit.n / (fit.n + SHRINK_K);
    // A run of identical reads fits a slope of zero, which is not evidence that nothing is
    // moving — it is a page nobody re-read. With no step to learn a refresh from and no
    // slope to learn a trend from, the documented launch-night dilution is the forecast:
    // the tracked population multiplies and the trophies already earned barely move, so
    // what the last read showed is not what the target will show.
    if (noStep) {
      const diluted = raw > 0 ? clampPct(raw * PRIOR_DILUTION) : raw;
      const centre = logit(diluted);
      // The spread stays wide: the practice's own refresh was one step of an unknown size,
      // and this branch is exactly the case where the program has never seen one.
      const sigma = Math.sqrt(REFRESH_SIGMA ** 2 + 0.55 ** 2);
      const q = [
        invLogit(centre - Z1 * sigma),
        invLogit(centre - Z2 * sigma),
        invLogit(centre),
        invLogit(centre + Z2 * sigma),
        invLogit(centre + Z1 * sigma),
      ].map((x) => round4(Math.min(Q_MAX, Math.max(Q_MIN, x))));
      for (let j = 1; j < q.length; j += 1) if ((q[j] as number) < (q[j - 1] as number)) q[j] = q[j - 1] as number;
      return { id: t.id, q };
    }

    const slope = weight * fit.slope + (1 - weight) * prior;

    const holdersTarget =
      fit.lastHolders > 0
        ? fit.lastHolders * Math.exp(DAMP_HOLDERS * slope * logRatio)
        : (raw / 100) * last.players_tracked;
    const centreRate = clampPct((holdersTarget / trackedTarget) * 100);
    const centre = logit(centreRate);

    // Spread, in logit units: what the fit's own residuals show, the drift of a trend
    // carried this far, and the chance of a refresh nobody saw coming. Rare trophies get
    // a longer upside tail — one player finishing a hidden branch can double a 0.1% rate,
    // while a rate has a floor at zero — so the upper half is stretched.
    const sigma = Math.sqrt(fit.scatter * fit.scatter + (DRIFT_SIGMA * Math.sqrt(Math.max(0, logRatio))) ** 2 + REFRESH_SIGMA ** 2);
    const lower = sigma;
    const upper = sigma * 1.45;

    const q = [
      invLogit(centre - Z1 * lower),
      invLogit(centre - Z2 * lower),
      invLogit(centre),
      invLogit(centre + Z2 * upper),
      invLogit(centre + Z1 * upper),
    ].map((x) => round4(Math.min(Q_MAX, Math.max(Q_MIN, x))));

    // Monotonicity last, on the finished numbers, never by re-sorting a spread that was
    // built around the median.
    for (let j = 1; j < q.length; j += 1) if ((q[j] as number) < (q[j - 1] as number)) q[j] = q[j - 1] as number;
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const rows = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}
