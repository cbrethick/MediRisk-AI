/**
 * Display currency.
 *
 * The models are trained, stored and evaluated in US dollars, because MEPS
 * records US healthcare spending. Conversion happens here, at render time only,
 * so switching currency can never change a prediction.
 *
 * The rate is carried in model.json (written from `USD_TO_INR` in ml/config.py)
 * so that the app and the written report can never disagree about it.
 *
 * Caveat worth repeating wherever these numbers appear: a US cost converted into
 * rupees is still a US cost. The same treatment is several times cheaper in
 * India, so these are "US prices, shown in rupees" rather than Indian prices.
 */
let rate = 88;
let code: "INR" | "USD" = "INR";

export function setCurrency(c: { code?: string; usdRate?: number } | undefined) {
  if (c?.usdRate) rate = c.usdRate;
  if (c?.code === "USD" || c?.code === "INR") code = c.code;
}

export function getRate() {
  return rate;
}

/**
 * Format a USD amount for display.
 * Indian readers count in lakh and crore rather than millions, so large figures
 * are abbreviated the way they are actually spoken.
 */
export function money(usd: number): string {
  if (code === "USD") {
    return `$${Math.round(usd).toLocaleString("en-US")}`;
  }
  const inr = usd * rate;
  if (Math.abs(inr) >= 1e7) return `₹${(inr / 1e7).toFixed(2)} Cr`;
  if (Math.abs(inr) >= 1e5) return `₹${(inr / 1e5).toFixed(2)} L`;
  return `₹${Math.round(inr).toLocaleString("en-IN")}`;
}

/** Exact rupee figure, for the one place that should show the full number. */
export function moneyExact(usd: number): string {
  return code === "USD"
    ? `$${Math.round(usd).toLocaleString("en-US")}`
    : `₹${Math.round(usd * rate).toLocaleString("en-IN")}`;
}
