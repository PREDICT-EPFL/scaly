# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""Risk on a book of options: Black-Scholes prices, Greeks by AD, and implied volatilities by Newton.

Not an optimization problem: the point here is evaluating many small functions, and their
derivatives, fast. A European call on a stock at ``S`` with strike ``K``, expiry ``T``, volatility
``sigma`` and rate ``r`` is worth

    C = S Phi(d1) - K exp(-r T) Phi(d2),   d1 = (log(S/K) + (r + sigma^2/2) T) / (sigma sqrt T),  d2 = d1 - sigma sqrt T,

with ``Phi`` written through ``erf``. One option is a ``Function`` of its five inputs
``p = (S, K, T, sigma, r)``; ``Function.factory`` gives the price, its gradient (delta, -theta,
vega, rho, and the strike sensitivity) and its Hessian (gamma, vanna, volga, ...) in one C
function, and ``sc.vmap`` maps that over the whole book as one loop.

The inverse problem, the implied volatility that reproduces a quoted price, is an ``sc.roots.root``
in ``sigma`` bracketed by ``[SIGMA_LO, SIGMA_HI]``, solved by ``sc.roots.NewtonBisection``: Newton
steps with the vega from forward mode, a bisection whenever one would leave the bracket, which
shrinks as the price error changes sign. The per-option solver is again ``vmap``-ped over the book.
Checked against SciPy's ``brentq``.

Finally, the total value of the book and its gradient with respect to every market input come from
one reverse sweep through the ``vmap``, the adjoint-algorithmic-differentiation trick risk desks
use instead of bumping each input.

The generated C lands in ``examples/generated/option_greeks/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import optimize, special

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "option_greeks"
M = 2000  # options in the book
NAMES = ("S", "K", "T", "sigma", "r")
MAX_NEWTON = 40
SIGMA_LO, SIGMA_HI = 1e-4, 5.0  # the bracket, as brentq's below


def call_price(p: sc.Expr) -> sc.Expr:
  s, k, t, sigma, r = p[0], p[1], p[2], p[3], p[4]
  vol = sigma * t.sqrt()
  d1 = ((s / k).log() + (r + 0.5 * sigma * sigma) * t) / vol
  d2 = d1 - vol
  phi = lambda x: 0.5 * (1.0 + (x / np.sqrt(2.0)).erf())  # noqa: E731
  return s * phi(d1) - k * (-r * t).exp() * phi(d2)


@sc.function(5, output="C")
def option(p: sc.Expr) -> sc.Expr:
  return call_price(p).reshape((1,))


# Price, gradient and Hessian of one option in one function, then over the book as one loop.
option_greeks = option.factory("option_greeks", ["p"], ["C", sc.factory.Grad("C", "p"), sc.factory.Hess("C", "p")])


@sc.function(5 * M, output=sc.G("price", "grad", "hess"))
def book_greeks(book: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  price, grad, hess = (sc.vmap(option_greeks, M, [(book, 0, 5)], output=k) for k in range(3))
  return price, grad, hess


@sc.function(5 * M, output=sc.G("value", "sensitivities"))
def book_value(book: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  value = sc.vmap(option, M, [(book, 0, 5)]).sum()
  return value, sc.gradient(value, book)


# Implied volatility. The quote is (S, K, T, r, C_market); the price error increases with sigma.
@sc.roots.root(vars=sc.L("sigma", ()), params=sc.L("quote", 5), name="implied_vol")
def iv_equation(sigma: sc.Expr, quote: sc.Expr) -> sc.roots.RootSpec:
  price = call_price(sc.stack([quote[0], quote[1], quote[2], sigma, quote[3]]))
  return sc.roots.RootSpec(price - quote[4], lb=sc.const(SIGMA_LO), ub=sc.const(SIGMA_HI))


iv_solve = sc.roots.solver(iv_equation, sc.roots.NewtonBisection(tol=1e-12, max_iter=MAX_NEWTON), name="iv_newton")


@sc.function(5, output=sc.G("sigma", "iterations"))
def implied_vol(quote: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  guess = (2 * np.pi / quote[2]).sqrt() * quote[4] / quote[0]  # Brenner-Subrahmanyam
  sigma, info = iv_solve(sc.minimum(sc.maximum(guess, 0.05), 2.0), quote)
  return sigma.reshape((1,)), info.iter.reshape((1,))


@sc.function(5 * M, output=sc.G("sigma", "iterations"))
def book_implied_vol(quotes: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return sc.vmap(implied_vol, M, [(quotes, 0, 5)], output=0), sc.vmap(implied_vol, M, [(quotes, 0, 5)], output=1)


def random_book(seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  s = rng.uniform(80, 120, M)
  k = s * rng.uniform(0.8, 1.2, M)
  t = rng.uniform(0.1, 2.0, M)
  sigma = rng.uniform(0.1, 0.6, M)
  r = rng.uniform(0.0, 0.05, M)
  return np.stack([s, k, t, sigma, r], axis=1)


def price_numpy(p: np.ndarray) -> np.ndarray:
  s, k, t, sigma, r = p.T
  d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
  d2 = d1 - sigma * np.sqrt(t)
  return s * special.ndtr(d1) - k * np.exp(-r * t) * special.ndtr(d2)


def main() -> dict:
  book = random_book()
  flat = book.reshape(-1)
  price, grad, hess = book_greeks(flat)
  grad, hess = grad.reshape(M, 5), hess.reshape(M, 5, 5)

  start = time.perf_counter()
  for _ in range(20):
    book_greeks(flat)
  per_option = (time.perf_counter() - start) / 20 / M

  # References: closed-form delta, vega and gamma; central differences for the rest.
  s, k, t, sigma, r = book.T
  d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
  pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
  closed = {"delta": special.ndtr(d1), "vega": s * pdf * np.sqrt(t), "gamma": pdf / (s * sigma * np.sqrt(t))}
  eps = 1e-6 * np.maximum(1.0, np.abs(book))
  fd = np.stack([(price_numpy(book + np.eye(5)[j] * eps) - price_numpy(book - np.eye(5)[j] * eps)) / (2 * eps[:, j]) for j in range(5)], axis=1)

  quotes = np.stack([s, k, t, r, price], axis=1)
  iv, iterations = book_implied_vol(quotes.reshape(-1))
  iv_ref = np.array(
    [optimize.brentq(lambda v, q=q: price_numpy(np.array([[q[0], q[1], q[2], v, q[3]]]))[0] - q[4], 1e-4, 5.0, xtol=1e-14) for q in quotes[:200]]
  )

  value, sensitivities = book_value(flat)
  return {
    "price_error": np.abs(price - price_numpy(book)).max(),
    "delta_error": np.abs(grad[:, 0] - closed["delta"]).max(),
    "vega_error": np.abs(grad[:, 3] - closed["vega"]).max(),
    "gamma_error": np.abs(hess[:, 0, 0] - closed["gamma"]).max(),
    "gradient_vs_fd": np.abs(grad - fd).max(),
    "hessian_symmetry": np.abs(hess - hess.transpose(0, 2, 1)).max(),
    "per_option_seconds": per_option,
    "iv_error": np.abs(iv - sigma).max(),
    "iv_brentq_error": np.abs(iv[:200] - iv_ref).max(),
    "iterations": iterations,
    "value": value,
    "sensitivities_error": np.abs(sensitivities.reshape(M, 5) - grad).max(),
  }


if __name__ == "__main__":
  out = main()
  print(f"{M} calls priced with gradient and Hessian in {1e9 * out['per_option_seconds']:.0f} ns per option")
  print(
    f"price vs SciPy {out['price_error']:.1e}; delta {out['delta_error']:.1e}, vega {out['vega_error']:.1e}, gamma {out['gamma_error']:.1e} vs closed form"
  )
  print(f"all five first derivatives vs central differences {out['gradient_vs_fd']:.1e}; Hessians symmetric to {out['hessian_symmetry']:.1e}")
  print(
    f"implied vols recovered to {out['iv_error']:.1e} (brentq agrees to {out['iv_brentq_error']:.1e}) in {out['iterations'].mean():.1f} steps on average, at most {int(out['iterations'].max())}"
  )
  print(
    f"book value {float(out['value']):.2f}; its 5 x {M} sensitivities from one reverse sweep match the per-option gradients to {out['sensitivities_error']:.1e}"
  )
  for fn in (option_greeks, book_greeks, implied_vol, book_value):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
