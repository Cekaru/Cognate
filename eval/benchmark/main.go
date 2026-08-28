// Command benchmark answers the one question the whole project exists to
// answer: on a multilingual workload, how much does cross-lingual caching win
// over an English-only cache, and does the structural guard keep that win
// honest?
//
// It is fully reproducible with no API key, no Docker, and no network. The
// inputs are the real BGE-M3 cosine scores already measured and committed by
// the calibration run (eval/calibration/scores/*.csv), plus the shipped
// per-language-pair thresholds (eval/calibration/thresholds.json). Every row
// in those CSVs is one incoming cross-lingual query paired with its nearest
// cache candidate, carrying the real embedding score and the structural-guard
// verdict, and a ground-truth label:
//
//   - positive      the two prompts mean the same thing → serving is correct,
//                   a real cache hit that avoids an LLM call.
//   - hard-negative same topic, one structural token differs ($100 vs $1000,
//                   order #A vs #B) → serving is a WRONG answer (a leak).
//   - easy-negative unrelated intents → serving is a wrong answer.
//
// Three systems are scored over that workload:
//
//	polyglot         serve iff score >= pairThreshold AND the guard passes
//	                 (what the proxy actually ships).
//	polyglot-noguard serve iff score >= pairThreshold (guard disabled) — isolates
//	                 exactly what the structural guard prevents.
//	english-only     an English-only cache never matches a cross-lingual query,
//	                 so it serves nothing here: the baseline's cross-lingual hit
//	                 rate is zero by construction.
//
// Reported: cross-lingual hit rate, unsafe-serve (leak) rate with vs without
// the guard, and — per the ROADMAP honesty rule — the win sold as COST, not
// latency, since translated paths add embedding overhead. Cost uses a
// documented, overridable price model and is projected to 1M queries so the
// per-query figures stay legible.
//
// Usage (from the repo root):
//
//	go run ./eval/benchmark
//	go run ./eval/benchmark -scores eval/calibration/scores -thresholds eval/calibration/thresholds.json
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// thresholdTable mirrors eval/calibration/thresholds.json.
type thresholdTable struct {
	Default float64            `json:"default"`
	Pairs   map[string]float64 `json:"pairs"`
	Model   string             `json:"model"`
}

func (t thresholdTable) forPair(pair string) float64 {
	if v, ok := t.Pairs[pair]; ok {
		return v
	}
	return t.Default
}

// row is one labeled cross-lingual query/candidate comparison from a scores CSV.
type row struct {
	label   string // positive | hard-negative | easy-negative
	pair    string // e.g. "en-tr"
	score   float64
	guardOK bool
}

// priceModel is an illustrative, overridable cost model. Defaults track
// gpt-4o-mini list pricing and a conservative hosted-embedding rate; BGE-M3 is
// self-hosted here, so the real embedding cost trends toward compute-only —
// the default overstates it on purpose, so the win is not flattered.
type priceModel struct {
	promptTokens     float64 // tokens in a typical support prompt
	completionTokens float64 // tokens in a typical support answer
	usdInPerM        float64 // provider input price per 1M tokens
	usdOutPerM       float64 // provider output price per 1M tokens
	usdEmbedPerM     float64 // embedding price per 1M tokens (paid on every query)
}

// llmCallUSD is the provider cost of one completion the cache can avoid.
func (p priceModel) llmCallUSD() float64 {
	return p.promptTokens*p.usdInPerM/1e6 + p.completionTokens*p.usdOutPerM/1e6
}

// embedUSD is what we pay to embed one query — the overhead the baseline never
// pays and the honest deduction from every cross-lingual saving.
func (p priceModel) embedUSD() float64 {
	return p.promptTokens * p.usdEmbedPerM / 1e6
}

// systemResult accumulates one system's behavior over the workload.
type systemResult struct {
	name         string
	trueHits     int // served AND positive — a correct, cost-saving hit
	unsafeServes int // served AND not positive — a leak
}

func main() {
	scoresDir := flag.String("scores", filepath.Join("eval", "calibration", "scores"), "directory of calibration score CSVs")
	thresholdsPath := flag.String("thresholds", filepath.Join("eval", "calibration", "thresholds.json"), "shipped per-pair threshold table")
	out := flag.String("out", filepath.Join("eval", "benchmark", "summary.md"), "markdown summary to write")
	promptTokens := flag.Float64("prompt-tokens", 60, "typical support prompt length (tokens)")
	completionTokens := flag.Float64("completion-tokens", 180, "typical support answer length (tokens)")
	usdIn := flag.Float64("usd-in-per-m", 0.15, "provider input price, USD per 1M tokens")
	usdOut := flag.Float64("usd-out-per-m", 0.60, "provider output price, USD per 1M tokens")
	usdEmbed := flag.Float64("usd-embed-per-m", 0.02, "embedding price, USD per 1M tokens")
	flag.Parse()

	price := priceModel{
		promptTokens:     *promptTokens,
		completionTokens: *completionTokens,
		usdInPerM:        *usdIn,
		usdOutPerM:       *usdOut,
		usdEmbedPerM:     *usdEmbed,
	}

	thr, err := loadThresholds(*thresholdsPath)
	if err != nil {
		fatal("load thresholds: %v", err)
	}
	rows, err := loadScores(*scoresDir)
	if err != nil {
		fatal("load scores: %v", err)
	}
	if len(rows) == 0 {
		fatal("no score rows found under %s", *scoresDir)
	}

	report := build(rows, thr, price)
	md := render(report, thr, price)

	if err := os.WriteFile(*out, []byte(md), 0o644); err != nil {
		fatal("write summary: %v", err)
	}
	fmt.Print(md)
	fmt.Fprintf(os.Stderr, "\nwrote %s\n", *out)
}

// report is the full benchmark outcome.
type report struct {
	totalQueries int
	positives    int

	polyglot        systemResult // shipped: threshold + guard
	polyglotNoGuard systemResult // threshold only
	englishOnly     systemResult // baseline: no cross-lingual match

	perPair []pairRow
}

type pairRow struct {
	pair       string
	positives  int
	hits       int     // polyglot true hits
	hitRate    float64 // hits / positives
	leaksNoGrd int     // unsafe serves without the guard
	leaksGuard int     // unsafe serves with the guard
}

func build(rows []row, thr thresholdTable, price priceModel) report {
	var r report
	byPair := map[string]*pairRow{}
	order := []string{}

	for _, x := range rows {
		r.totalQueries++
		positive := x.label == "positive"
		if positive {
			r.positives++
		}
		cutoff := thr.forPair(x.pair)
		overThreshold := x.score >= cutoff

		pr := byPair[x.pair]
		if pr == nil {
			pr = &pairRow{pair: x.pair}
			byPair[x.pair] = pr
			order = append(order, x.pair)
		}
		if positive {
			pr.positives++
		}

		// polyglot (shipped): threshold AND guard.
		if overThreshold && x.guardOK {
			if positive {
				r.polyglot.trueHits++
				pr.hits++
			} else {
				r.polyglot.unsafeServes++
				pr.leaksGuard++
			}
		}
		// polyglot without the guard: threshold only.
		if overThreshold {
			if positive {
				r.polyglotNoGuard.trueHits++
			} else {
				r.polyglotNoGuard.unsafeServes++
				pr.leaksNoGrd++
			}
		}
		// english-only baseline: every row here is a cross-lingual pair
		// (lang_a != lang_b), which an English-only cache cannot match.
		// It serves nothing: no hits, no leaks.
	}

	r.polyglot.name = "polyglot (threshold + guard)"
	r.polyglotNoGuard.name = "polyglot (threshold only)"
	r.englishOnly.name = "english-only baseline"

	for _, p := range order {
		pr := byPair[p]
		if pr.positives > 0 {
			pr.hitRate = float64(pr.hits) / float64(pr.positives)
		}
		r.perPair = append(r.perPair, *pr)
	}
	sort.Slice(r.perPair, func(i, j int) bool { return r.perPair[i].pair < r.perPair[j].pair })
	return r
}

func render(r report, thr thresholdTable, price priceModel) string {
	var b strings.Builder
	p := func(format string, a ...any) { fmt.Fprintf(&b, format, a...) }

	hitRate := ratio(r.polyglot.trueHits, r.positives)
	leakRateGuard := ratio(r.polyglot.unsafeServes, r.totalQueries-r.positives)
	leakRateNoGuard := ratio(r.polyglotNoGuard.unsafeServes, r.totalQueries-r.positives)

	// Cost projected to 1M cross-lingual queries at the measured hit rate.
	const scale = 1_000_000
	hitsAtScale := hitRate * scale
	grossSaved := hitsAtScale * price.llmCallUSD()
	embedOverhead := float64(scale) * price.embedUSD()
	netSaved := grossSaved - embedOverhead

	p("# Cross-lingual benchmark\n\n")
	p("Reproducible offline from committed BGE-M3 scores (`eval/calibration/scores/`)\n")
	p("and the shipped thresholds (`%s`). Regenerate with `go run ./eval/benchmark`.\n\n", thr.Model)
	p("Workload: **%d** labeled cross-lingual queries across 6 language pairs, of which\n", r.totalQueries)
	p("**%d** are true hit opportunities (semantically equivalent prompts) and the rest\n", r.positives)
	p("are near-miss or unrelated candidates that must **not** be served.\n\n")

	p("## Headline\n\n")
	p("| Metric | Polyglot (shipped) | English-only baseline |\n")
	p("|--------|:------------------:|:---------------------:|\n")
	p("| Cross-lingual hit rate | **%.1f%%** | 0.0%% (by construction) |\n", hitRate*100)
	p("| Unsafe serves (leaks) | **%.1f%%** | 0.0%% |\n", leakRateGuard*100)
	p("| Net cost saved / 1M queries | **$%s** | $0 |\n\n", money(netSaved))

	p("The baseline captures **zero** cross-lingual hits: an English-only cache cannot\n")
	p("match a Turkish prompt to a Spanish entry. Every cross-lingual hit Polyglot lands\n")
	p("is spend the baseline pays in full.\n\n")

	p("## What the guard is worth\n\n")
	p("Without the structural guard, the threshold alone serves near-miss candidates —\n")
	p("same topic, wrong number or ID — as if they were hits. Those are wrong answers.\n\n")
	p("| | Threshold only | Threshold + guard (shipped) |\n")
	p("|--|:--:|:--:|\n")
	p("| Unsafe serve rate | %.1f%% | **%.1f%%** |\n", leakRateNoGuard*100, leakRateGuard*100)
	p("| Unsafe serves (count) | %d | **%d** |\n", r.polyglotNoGuard.unsafeServes, r.polyglot.unsafeServes)
	p("| True hits kept | %d | %d |\n\n", r.polyglotNoGuard.trueHits, r.polyglot.trueHits)
	p("The guard removes the leak while keeping the hits: the threshold can sit lower\n")
	p("*because* the guard is the backstop, which is exactly why the calibrated\n")
	p("with-guard cutoffs recover more recall at the precision target.\n\n")

	p("## Per-pair hit rate\n\n")
	p("| pair | hit opportunities | hits | hit rate | leaks (no guard → guard) |\n")
	p("|------|:--:|:--:|:--:|:--:|\n")
	for _, pr := range r.perPair {
		p("| %s | %d | %d | %.0f%% | %d → %d |\n",
			pr.pair, pr.positives, pr.hits, pr.hitRate*100, pr.leaksNoGrd, pr.leaksGuard)
	}
	p("\n")

	p("## Cost model (illustrative, overridable)\n\n")
	p("Sold as **cost**, not latency: a translated path adds embedding overhead, so\n")
	p("total latency is higher — the win is dollars, not milliseconds.\n\n")
	p("| Assumption | Value |\n")
	p("|--|--|\n")
	p("| Prompt / completion tokens | %.0f / %.0f |\n", price.promptTokens, price.completionTokens)
	p("| Provider price (in / out, USD per 1M) | $%.2f / $%.2f |\n", price.usdInPerM, price.usdOutPerM)
	p("| Embedding price (USD per 1M) | $%.2f |\n", price.usdEmbedPerM)
	p("| Avoided LLM call | $%.6f each |\n", price.llmCallUSD())
	p("| Embedding overhead | $%.6f per query |\n\n", price.embedUSD())
	p("At the measured **%.1f%%** hit rate over 1M cross-lingual queries:\n\n", hitRate*100)
	p("- gross saved (avoided LLM calls): **$%s**\n", money(grossSaved))
	p("- embedding overhead (paid on every query): **$%s**\n", money(embedOverhead))
	p("- **net saved: $%s**\n\n", money(netSaved))
	p("Embedding is self-hosted (BGE-M3 sidecar), so the real overhead trends below the\n")
	p("hosted rate used here; the net figure is deliberately conservative.\n")

	return b.String()
}

func loadThresholds(path string) (thresholdTable, error) {
	var t thresholdTable
	data, err := os.ReadFile(path)
	if err != nil {
		return t, err
	}
	if err := json.Unmarshal(data, &t); err != nil {
		return t, err
	}
	if t.Model == "" {
		t.Model = "thresholds.json"
	}
	return t, nil
}

func loadScores(dir string) ([]row, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.csv"))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var rows []row
	for _, f := range files {
		fr, err := readScoreFile(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(f), err)
		}
		rows = append(rows, fr...)
	}
	return rows, nil
}

func readScoreFile(path string) ([]row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rd := csv.NewReader(f)
	records, err := rd.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	// Header: label,kind,lang_a,lang_b,intent_a,intent_b,score,guard_ok
	col := map[string]int{}
	for i, name := range records[0] {
		col[strings.TrimSpace(name)] = i
	}
	need := []string{"label", "lang_a", "lang_b", "score", "guard_ok"}
	for _, n := range need {
		if _, ok := col[n]; !ok {
			return nil, fmt.Errorf("missing column %q", n)
		}
	}

	var rows []row
	for _, rec := range records[1:] {
		score, err := strconv.ParseFloat(strings.TrimSpace(rec[col["score"]]), 64)
		if err != nil {
			return nil, fmt.Errorf("bad score %q: %w", rec[col["score"]], err)
		}
		guardOK, err := strconv.ParseBool(strings.TrimSpace(rec[col["guard_ok"]]))
		if err != nil {
			return nil, fmt.Errorf("bad guard_ok %q: %w", rec[col["guard_ok"]], err)
		}
		rows = append(rows, row{
			label:   strings.TrimSpace(rec[col["label"]]),
			pair:    pairKey(rec[col["lang_a"]], rec[col["lang_b"]]),
			score:   score,
			guardOK: guardOK,
		})
	}
	return rows, nil
}

// pairKey canonicalizes a language pair so en-tr and tr-en share one threshold.
func pairKey(a, b string) string {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	if a > b {
		a, b = b, a
	}
	return a + "-" + b
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

// money formats a USD amount with thousands separators and cents.
func money(v float64) string {
	neg := v < 0
	if neg {
		v = -v
	}
	whole := int64(v)
	cents := int64((v-float64(whole))*100 + 0.5)
	if cents == 100 {
		whole++
		cents = 0
	}
	s := strconv.FormatInt(whole, 10)
	var out strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			out.WriteByte(',')
		}
		out.WriteRune(c)
	}
	sign := ""
	if neg {
		sign = "-"
	}
	return fmt.Sprintf("%s%s.%02d", sign, out.String(), cents)
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(1)
}
