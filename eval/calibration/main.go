// Command calibration produces the per-language-pair threshold table from the
// labeled intents set (intents.jsonl), using the real BGE-M3 sidecar.
//
// For every language pair it scores three sample classes by cosine similarity:
//
//   - positives: the same intent expressed in the two languages (must match)
//   - hard negatives: an intent vs its variant — same topic, one structural
//     token changed ($49.99 vs $89.99) or a semantic near-miss (cancel vs
//     pause). These must NOT match; they are the leak surface.
//   - easy negatives: unrelated intents across the two languages.
//
// It then runs ROC analysis per pair, picks the lowest threshold that meets
// the precision target (recall is what we buy, precision is what we owe), and
// reports what the structural token guard adds on top: how many hard negatives
// that beat the threshold are still caught, and how often the guard misfires
// on true positives.
//
// Outputs, all committed as the calibration artifact:
//
//	thresholds.json  per-pair table the proxy loads via THRESHOLDS_FILE
//	scores/          raw labeled cosine scores per pair (CSV)
//	roc/             ROC curve per pair (SVG)
//	summary.md       one table: AUC, threshold, recall, leak, guard effect
//
// Usage (with the sidecar up, e.g. `docker compose up -d sidecar`):
//
//	go run ./eval/calibration
//	go run ./eval/calibration -sidecar http://localhost:8000 -precision 0.99
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
	"github.com/kaanrumin/polyglot-cache/internal/guard"
)

var langs = []string{"en", "es", "tr", "zh"}

var langPairs = [][2]string{
	{"en", "es"}, {"en", "tr"}, {"en", "zh"},
	{"es", "tr"}, {"es", "zh"}, {"tr", "zh"},
}

type intent struct {
	ID      string            `json:"id"`
	Kind    string            `json:"kind"` // "token" (guard-catchable) or "semantic"
	Texts   map[string]string `json:"texts"`
	Variant map[string]string `json:"variant"`
}

// sample is one labeled cross-lingual comparison.
type sample struct {
	label    string // "positive", "hard-negative", "easy-negative"
	kind     string // "token", "semantic", or "cross-intent"
	intentA  string
	intentB  string
	score    float64
	guardOK  bool // structural guard verdict for this pair of texts
}

// pairReport is the calibration result for one language pair.
type pairReport struct {
	pair               string
	nPos, nHard, nEasy int
	auc                float64

	// Embedding-only operating point: what the threshold alone must do.
	threshold         float64
	recall, precision float64

	// System operating point: the engine always runs the guard after the
	// threshold, so guard-rejected negatives are not servable and the cutoff
	// can sit lower — this is the threshold the proxy ships with. Recall is
	// against all positives (guard false-fires count as losses).
	thresholdGuard         float64
	recallGuard, precGuard float64

	hardLeak               int // hard negatives >= embedding-only threshold
	hardLeakAfterGuard     int // ... that also pass the guard
	easyLeak               int
	guardFalseFires        int // positives the guard wrongly rejects
	tokenHard, tokenCaught int // guard performance on token-kind hard negatives
}

func main() {
	sidecar := flag.String("sidecar", envOr("EMBED_SIDECAR_URL", "http://localhost:8000"), "BGE-M3 sidecar base URL")
	intentsPath := flag.String("intents", "eval/calibration/intents.jsonl", "labeled intents file")
	outDir := flag.String("out", "eval/calibration", "output directory")
	precision := flag.Float64("precision", 0.99, "precision target for threshold selection")
	flag.Parse()

	intents, err := readIntents(*intentsPath)
	if err != nil {
		fatal("read intents: %v", err)
	}
	fmt.Printf("Loaded %d intents from %s\n", len(intents), *intentsPath)

	vecs, err := embedAll(*sidecar, intents)
	if err != nil {
		fatal("embed: %v", err)
	}

	// Structural tokens for the guard-effect analysis.
	tokens := map[string]cache.StructuralTokens{}
	for _, in := range intents {
		for _, l := range langs {
			tokens[in.ID+"/"+l] = guard.Extract(in.Texts[l], l)
			tokens[in.ID+"/var/"+l] = guard.Extract(in.Variant[l], l)
		}
	}

	var reports []pairReport
	pairThresholds := map[string]float64{}
	for _, lp := range langPairs {
		samples := buildSamples(intents, vecs, tokens, lp[0], lp[1])
		rep := analyze(pairKey(lp[0], lp[1]), samples, *precision)
		reports = append(reports, rep)
		// Ship the guard-aware cutoff: the engine always runs the guard after
		// the threshold, so this is the operating point that actually applies.
		pairThresholds[rep.pair] = round4(rep.thresholdGuard)

		if err := writeCSV(filepath.Join(*outDir, "scores", rep.pair+".csv"), lp[0], lp[1], samples); err != nil {
			fatal("write scores: %v", err)
		}
		if err := writeROC(filepath.Join(*outDir, "roc", rep.pair+".svg"), rep, samples); err != nil {
			fatal("write roc: %v", err)
		}
	}

	if err := writeThresholds(filepath.Join(*outDir, "thresholds.json"), pairThresholds, *precision); err != nil {
		fatal("write thresholds: %v", err)
	}
	if err := writeSummary(filepath.Join(*outDir, "summary.md"), reports, *precision, len(intents)); err != nil {
		fatal("write summary: %v", err)
	}

	printReports(reports, *precision)
}

// buildSamples assembles the labeled comparisons for one language pair.
func buildSamples(intents []intent, vecs map[string][]float64, tokens map[string]cache.StructuralTokens, a, b string) []sample {
	var out []sample
	n := len(intents)

	guardVerdict := func(keyA, keyB string) bool {
		ok, _ := guard.Compare(tokens[keyA], tokens[keyB])
		return ok
	}

	for i, in := range intents {
		// Positive: same intent, two languages.
		out = append(out, sample{
			label: "positive", kind: in.Kind, intentA: in.ID, intentB: in.ID,
			score:   cosine(vecs[in.ID+"/"+a], vecs[in.ID+"/"+b]),
			guardOK: guardVerdict(in.ID+"/"+a, in.ID+"/"+b),
		})
		// Hard negatives: intent vs its own variant, both directions (the two
		// directions compare different text pairs, so both are informative).
		out = append(out, sample{
			label: "hard-negative", kind: in.Kind, intentA: in.ID, intentB: in.ID + "/var",
			score:   cosine(vecs[in.ID+"/"+a], vecs[in.ID+"/var/"+b]),
			guardOK: guardVerdict(in.ID+"/"+a, in.ID+"/var/"+b),
		})
		out = append(out, sample{
			label: "hard-negative", kind: in.Kind, intentA: in.ID + "/var", intentB: in.ID,
			score:   cosine(vecs[in.ID+"/var/"+a], vecs[in.ID+"/"+b]),
			guardOK: guardVerdict(in.ID+"/var/"+a, in.ID+"/"+b),
		})
		// Easy negatives: unrelated intents at two fixed offsets.
		for _, off := range []int{1, 7} {
			j := (i + off) % n
			if j == i {
				continue
			}
			out = append(out, sample{
				label: "easy-negative", kind: "cross-intent", intentA: in.ID, intentB: intents[j].ID,
				score:   cosine(vecs[in.ID+"/"+a], vecs[intents[j].ID+"/"+b]),
				guardOK: guardVerdict(in.ID+"/"+a, intents[j].ID+"/"+b),
			})
		}
	}
	return out
}

// analyze runs the ROC and picks the lowest threshold meeting the precision
// target, i.e. the most recall the pair can afford at that safety level.
func analyze(pair string, samples []sample, precisionTarget float64) pairReport {
	rep := pairReport{pair: pair}

	// pos/neg drive the embedding-only analysis; the *Srv sets keep only the
	// samples the guard would let through — the servable population, which is
	// what the deployed threshold actually gates.
	var pos, neg, posSrv, negSrv []float64
	for _, s := range samples {
		switch s.label {
		case "positive":
			pos = append(pos, s.score)
			rep.nPos++
			if s.guardOK {
				posSrv = append(posSrv, s.score)
			} else {
				rep.guardFalseFires++
			}
		case "hard-negative":
			neg = append(neg, s.score)
			rep.nHard++
			if s.guardOK {
				negSrv = append(negSrv, s.score)
			}
			if s.kind == "token" {
				rep.tokenHard++
				if !s.guardOK {
					rep.tokenCaught++
				}
			}
		case "easy-negative":
			neg = append(neg, s.score)
			rep.nEasy++
			if s.guardOK {
				negSrv = append(negSrv, s.score)
			}
		}
	}
	rep.auc = auc(pos, neg)
	rep.threshold, rep.recall, rep.precision = chooseThreshold(pos, neg, precisionTarget)

	var recallSrv float64
	rep.thresholdGuard, recallSrv, rep.precGuard = chooseThreshold(posSrv, negSrv, precisionTarget)
	// Recall against ALL positives: a guard false-fire is a lost hit too.
	rep.recallGuard = recallSrv * float64(len(posSrv)) / float64(rep.nPos)

	for _, s := range samples {
		if s.score < rep.threshold {
			continue
		}
		switch s.label {
		case "hard-negative":
			rep.hardLeak++
			if s.guardOK {
				rep.hardLeakAfterGuard++
			}
		case "easy-negative":
			rep.easyLeak++
		}
	}
	return rep
}

// chooseThreshold scans candidate cutoffs (midpoints between adjacent distinct
// scores, plus the extremes) and returns the one with the highest recall whose
// precision meets the target; ties prefer higher precision, then the higher
// cutoff (more margin). If no cutoff reaches the target, it returns a cutoff
// above every negative (precision 1 at whatever recall remains).
func chooseThreshold(pos, neg []float64, target float64) (thr, recall, precision float64) {
	all := append(append([]float64{}, pos...), neg...)
	sort.Float64s(all)
	var candidates []float64
	for i := 1; i < len(all); i++ {
		if all[i] != all[i-1] {
			candidates = append(candidates, (all[i]+all[i-1])/2)
		}
	}
	candidates = append(candidates, all[0]-0.001, all[len(all)-1]+0.001)

	eval := func(t float64) (rec, prec float64) {
		tp, fp := 0, 0
		for _, s := range pos {
			if s >= t {
				tp++
			}
		}
		for _, s := range neg {
			if s >= t {
				fp++
			}
		}
		if tp+fp == 0 {
			return 0, 1
		}
		return float64(tp) / float64(len(pos)), float64(tp) / float64(tp+fp)
	}

	best := false
	for _, t := range candidates {
		rec, prec := eval(t)
		if prec < target {
			continue
		}
		better := !best || rec > recall || (rec == recall && prec > precision) ||
			(rec == recall && prec == precision && t > thr)
		if better {
			thr, recall, precision, best = t, rec, prec, true
		}
	}
	if !best {
		// Precision target unreachable: sit above every negative.
		maxNeg := math.Inf(-1)
		for _, s := range neg {
			maxNeg = math.Max(maxNeg, s)
		}
		thr = maxNeg + 0.001
		recall, precision = eval(thr)
	}
	return thr, recall, precision
}

// auc is the Mann-Whitney estimate: P(random positive > random negative),
// counting ties as half.
func auc(pos, neg []float64) float64 {
	if len(pos) == 0 || len(neg) == 0 {
		return math.NaN()
	}
	var sum float64
	for _, p := range pos {
		for _, n := range neg {
			switch {
			case p > n:
				sum += 1
			case p == n:
				sum += 0.5
			}
		}
	}
	return sum / float64(len(pos)*len(neg))
}

// ---- outputs ----

func writeCSV(path, langA, langB string, samples []sample) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	var b strings.Builder
	b.WriteString("label,kind,lang_a,lang_b,intent_a,intent_b,score,guard_ok\n")
	for _, s := range samples {
		fmt.Fprintf(&b, "%s,%s,%s,%s,%s,%s,%.6f,%v\n",
			s.label, s.kind, langA, langB, s.intentA, s.intentB, s.score, s.guardOK)
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// writeROC renders the ROC curve as a dependency-free SVG: FPR on x, TPR on y,
// the chance diagonal, and the chosen operating point.
func writeROC(path string, rep pairReport, samples []sample) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	var pos, neg []float64
	for _, s := range samples {
		if s.label == "positive" {
			pos = append(pos, s.score)
		} else {
			neg = append(neg, s.score)
		}
	}
	// Sweep thresholds from high to low: each distinct score is a cut.
	cuts := append(append([]float64{}, pos...), neg...)
	sort.Sort(sort.Reverse(sort.Float64Slice(cuts)))
	type pt struct{ fpr, tpr float64 }
	points := []pt{{0, 0}}
	for _, t := range cuts {
		tp, fp := 0, 0
		for _, s := range pos {
			if s >= t {
				tp++
			}
		}
		for _, s := range neg {
			if s >= t {
				fp++
			}
		}
		points = append(points, pt{float64(fp) / float64(len(neg)), float64(tp) / float64(len(pos))})
	}
	points = append(points, pt{1, 1})

	const w, h, m = 460, 460, 50 // canvas and margin
	x := func(v float64) float64 { return m + v*(w-2*m) }
	y := func(v float64) float64 { return h - m - v*(h-2*m) }

	var poly strings.Builder
	for i, p := range points {
		if i > 0 {
			poly.WriteByte(' ')
		}
		fmt.Fprintf(&poly, "%.1f,%.1f", x(p.fpr), y(p.tpr))
	}

	// Operating point at the chosen threshold.
	opTP, opFP := 0, 0
	for _, s := range pos {
		if s >= rep.threshold {
			opTP++
		}
	}
	for _, s := range neg {
		if s >= rep.threshold {
			opFP++
		}
	}
	opX := x(float64(opFP) / float64(len(neg)))
	opY := y(float64(opTP) / float64(len(pos)))

	svg := fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="monospace" font-size="12">
  <rect width="%d" height="%d" fill="white"/>
  <line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#888"/>
  <line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#888"/>
  <line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ccc" stroke-dasharray="4 4"/>
  <polyline points="%s" fill="none" stroke="#0b6" stroke-width="2"/>
  <circle cx="%.1f" cy="%.1f" r="4" fill="#d33"/>
  <text x="%.1f" y="%.1f" fill="#d33">thr %.3f</text>
  <text x="%d" y="%d">FPR</text>
  <text x="12" y="%d" transform="rotate(-90 12 %d)">TPR</text>
  <text x="%d" y="24" font-size="14">%s — ROC, AUC %.4f</text>
  <text x="%d" y="%d">0</text><text x="%d" y="%d">1</text><text x="%d" y="%d">1</text>
</svg>
`,
		w, h, w, h,
		w, h,
		m, h-m, w-m, h-m, // x axis
		m, m, m, h-m, // y axis
		x(0), y(0), x(1), y(1), // diagonal
		poly.String(),
		opX, opY,
		opX+8, opY-8, rep.threshold,
		w/2, h-14,
		h/2, h/2,
		m, rep.pair, rep.auc,
		m-4, h-m+16, w-m-4, h-m+16, m-16, m+4,
	)
	return os.WriteFile(path, []byte(svg), 0o644)
}

func writeThresholds(path string, pairs map[string]float64, precisionTarget float64) error {
	// The default for unseen pairs is the most conservative calibrated cutoff.
	def := 0.0
	for _, v := range pairs {
		def = math.Max(def, v)
	}
	doc := struct {
		Default         float64            `json:"default"`
		Pairs           map[string]float64 `json:"pairs"`
		Model           string             `json:"model"`
		PrecisionTarget float64            `json:"precision_target"`
		GeneratedAt     string             `json:"generated_at"`
	}{def, pairs, "BAAI/bge-m3", precisionTarget, time.Now().UTC().Format(time.RFC3339)}
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func writeSummary(path string, reports []pairReport, precisionTarget float64, nIntents int) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# Calibration summary\n\n")
	fmt.Fprintf(&b, "%d intents × 4 languages, precision target %.2f. Hard negatives are same-intent\n", nIntents, precisionTarget)
	fmt.Fprintf(&b, "variants (one structural token changed, or a semantic near-miss); easy negatives\n")
	fmt.Fprintf(&b, "are unrelated intents. `leak` counts negatives at or above the threshold.\n\n")
	b.WriteString("| pair | AUC | thr (embed only) | recall | thr (with guard) | recall | precision | guard catch (token hard-negs) | guard false-fires (positives) |\n")
	b.WriteString("|------|-----|------------------|--------|------------------|--------|-----------|-------------------------------|-------------------------------|\n")
	for _, r := range reports {
		fmt.Fprintf(&b, "| %s | %.4f | %.4f | %.0f%% | %.4f | %.0f%% | %.1f%% | %d/%d | %d/%d |\n",
			r.pair, r.auc, r.threshold, 100*r.recall,
			r.thresholdGuard, 100*r.recallGuard, 100*r.precGuard,
			r.tokenCaught, r.tokenHard, r.guardFalseFires, r.nPos)
	}
	b.WriteString("\nThe shipped `thresholds.json` uses the **with guard** cutoffs: the engine always\n")
	b.WriteString("runs the structural guard after the threshold, so guard-rejected near-misses are\n")
	b.WriteString("not servable and the threshold only has to separate what the guard cannot.\n")
	b.WriteString("\nGenerated by `go run ./eval/calibration`; raw scores in `scores/`, curves in `roc/`.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printReports(reports []pairReport, precisionTarget float64) {
	fmt.Printf("\nPer-pair calibration (precision target %.2f):\n", precisionTarget)
	fmt.Printf("%-8s %8s %12s %8s %12s %8s %10s %12s %12s\n",
		"pair", "AUC", "thr(embed)", "recall", "thr(guard)", "recall", "precision", "guard-catch", "false-fire")
	for _, r := range reports {
		fmt.Printf("%-8s %8.4f %12.4f %7.0f%% %12.4f %7.0f%% %9.1f%% %9d/%-3d %9d/%-3d\n",
			r.pair, r.auc, r.threshold, 100*r.recall,
			r.thresholdGuard, 100*r.recallGuard, 100*r.precGuard,
			r.tokenCaught, r.tokenHard, r.guardFalseFires, r.nPos)
	}
}

// ---- embedding ----

// embedAll embeds every (intent, lang) text and variant, keyed as
// "<id>/<lang>" and "<id>/var/<lang>", chunked to keep sidecar requests small.
func embedAll(base string, intents []intent) (map[string][]float64, error) {
	var keys, texts []string
	for _, in := range intents {
		for _, l := range langs {
			keys = append(keys, in.ID+"/"+l)
			texts = append(texts, in.Texts[l])
			keys = append(keys, in.ID+"/var/"+l)
			texts = append(texts, in.Variant[l])
		}
	}
	out := make(map[string][]float64, len(keys))
	const chunk = 64
	for i := 0; i < len(texts); i += chunk {
		end := min(i+chunk, len(texts))
		embs, err := embed(base, texts[i:end])
		if err != nil {
			return nil, err
		}
		if len(embs) != end-i {
			return nil, fmt.Errorf("got %d embeddings for %d texts", len(embs), end-i)
		}
		for j, e := range embs {
			out[keys[i+j]] = e
		}
		fmt.Printf("embedded %d/%d\n", end, len(texts))
	}
	return out, nil
}

func embed(base string, texts []string) ([][]float64, error) {
	body, _ := json.Marshal(map[string]any{"texts": texts})
	cli := &http.Client{Timeout: 5 * time.Minute}
	resp, err := cli.Post(base+"/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sidecar status %s", resp.Status)
	}
	var er struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, err
	}
	return er.Embeddings, nil
}

// ---- helpers ----

func readIntents(path string) ([]intent, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []intent
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var in intent
		if err := json.Unmarshal(line, &in); err != nil {
			return nil, err
		}
		for _, l := range langs {
			if in.Texts[l] == "" || in.Variant[l] == "" {
				return nil, fmt.Errorf("intent %q: missing %s text or variant", in.ID, l)
			}
		}
		out = append(out, in)
	}
	return out, sc.Err()
}

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// pairKey mirrors engine.PairKey: order-independent "a-b" with a <= b.
func pairKey(a, b string) string {
	if b < a {
		a, b = b, a
	}
	return a + "-" + b
}

func round4(v float64) float64 {
	return math.Round(v*1e4) / 1e4
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
