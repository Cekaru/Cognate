// Package guard implements the locale-aware structural token guard. On a
// semantic hit it compares numbers, IDs, currencies, dates, and code
// identifiers between the two prompts — normalizing across locales first
// (1.000,50 vs 1,000.50; Eastern vs Western Arabic numerals; DD/MM vs MM/DD)
// — and rejects the match on mismatch. This is the security core; a guard
// written naively for English silently fails on exactly the non-English
// inputs the project exists to serve.
package guard
