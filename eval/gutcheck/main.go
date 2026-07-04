// Command gutcheck measures cross-lingual embedding quality against the real
// BGE-M3 sidecar. For each intent in a pairs file it embeds all four language
// variants (EN/ES/TR/ZH) and reports whether equivalents (same intent, different
// language) sit closer in vector space than non-equivalents (different intent).
//
// It is the reality check the fake-embedder unit tests can't give: it proves the
// central thesis — that BGE-M3 places cross-lingual equivalents near each other —
// on real prompts, and it is the seed dataset for per-language-pair threshold
// calibration (the cosine distributions here become the labeled positives and
// hard-negatives that calibration turns into per-pair cutoffs).
//
// Usage (with the sidecar up, e.g. `docker compose up -d sidecar`):
//
//	go run ./eval/gutcheck
//	go run ./eval/gutcheck -sidecar http://localhost:8000 -pairs eval/demo/pairs.jsonl -threshold 0.85
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
	"sort"
	"time"
)

type pair struct {
	ID string `json:"id"`
	EN string `json:"en"`
	ES string `json:"es"`
	TR string `json:"tr"`
	ZH string `json:"zh"`
}

var langs = []string{"en", "es", "tr", "zh"}

func (p pair) text(l string) string {
	switch l {
	case "en":
		return p.EN
	case "es":
		return p.ES
	case "tr":
		return p.TR
	case "zh":
		return p.ZH
	}
	return ""
}

type embedResp struct {
	Dim        int         `json:"dim"`
	Embeddings [][]float64 `json:"embeddings"`
}

func main() {
	sidecar := flag.String("sidecar", envOr("EMBED_SIDECAR_URL", "http://localhost:8000"), "BGE-M3 sidecar base URL")
	pairsPath := flag.String("pairs", envOr("PAIRS", "eval/demo/pairs.jsonl"), "path to the intents JSONL file")
	threshold := flag.Float64("threshold", 0.85, "cross-lingual cutoff to evaluate")
	flag.Parse()

	pairs, err := readPairs(*pairsPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read pairs:", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d intents from %s\n", len(pairs), *pairsPath)

	// Flatten every (intent, lang) text into one embed request, index-aligned.
	type key struct {
		i    int
		lang string
	}
	var (
		texts []string
		keys  []key
	)
	for i, p := range pairs {
		for _, l := range langs {
			texts = append(texts, p.text(l))
			keys = append(keys, key{i, l})
		}
	}

	embs, err := embed(*sidecar, texts)
	if err != nil {
		fmt.Fprintln(os.Stderr, "embed:", err)
		os.Exit(1)
	}
	if len(embs) != len(texts) {
		fmt.Fprintf(os.Stderr, "got %d embeddings for %d texts\n", len(embs), len(texts))
		os.Exit(1)
	}
	vec := make([]map[string][]float64, len(pairs))
	for i := range vec {
		vec[i] = map[string][]float64{}
	}
	for n, k := range keys {
		vec[k.i][k.lang] = embs[n]
	}

	langPairs := [][2]string{{"es", "tr"}, {"es", "en"}, {"es", "zh"}, {"tr", "en"}, {"tr", "zh"}, {"en", "zh"}}

	fmt.Println()
	fmt.Println("Per-intent cross-lingual cosine (equivalents — should be HIGH):")
	fmt.Printf("%-18s %7s %7s %7s %7s %7s %7s | %7s\n", "intent", "es-tr", "es-en", "es-zh", "tr-en", "tr-zh", "en-zh", "min")
	var allEquivalent, esTrScores []float64
	perIntentMin := make([]float64, len(pairs))
	for i, p := range pairs {
		row := make([]float64, len(langPairs))
		mn := math.Inf(1)
		for j, lp := range langPairs {
			c := cosine(vec[i][lp[0]], vec[i][lp[1]])
			row[j] = c
			allEquivalent = append(allEquivalent, c)
			if lp[0] == "es" && lp[1] == "tr" {
				esTrScores = append(esTrScores, c)
			}
			if c < mn {
				mn = c
			}
		}
		perIntentMin[i] = mn
		fmt.Printf("%-18s %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f | %7.3f\n",
			p.ID, row[0], row[1], row[2], row[3], row[4], row[5], mn)
	}

	// Hard negatives: the SAME language pair but ACROSS different intents.
	var hardNeg []float64
	for a := range pairs {
		for b := range pairs {
			if a == b {
				continue
			}
			hardNeg = append(hardNeg, cosine(vec[a]["es"], vec[b]["tr"]))
		}
	}

	fmt.Println()
	fmt.Println("Distributions:")
	statLine("equivalents (all cross-lingual pairs)", allEquivalent)
	statLine("  of which es-tr only", esTrScores)
	statLine("non-equivalents (es[i] vs tr[j], i!=j)", hardNeg)

	eqMin, _, eqMean := minMaxMean(allEquivalent)
	_, negMax, negMean := minMaxMean(hardNeg)
	fmt.Println()
	fmt.Printf("Threshold %.2f gate:\n", *threshold)
	passEq := countGE(allEquivalent, *threshold)
	falseHit := countGE(hardNeg, *threshold)
	fmt.Printf("  equivalents admitted:      %d/%d (%.0f%%)\n", passEq, len(allEquivalent), 100*float64(passEq)/float64(len(allEquivalent)))
	fmt.Printf("  non-equivalents leaked:    %d/%d (%.2f%%)\n", falseHit, len(hardNeg), 100*float64(falseHit)/float64(len(hardNeg)))
	fmt.Printf("  worst equivalent cosine:   %.3f\n", eqMin)
	fmt.Printf("  best  non-equiv cosine:    %.3f\n", negMax)
	fmt.Printf("  mean equiv %.3f  vs  mean non-equiv %.3f  (gap %.3f)\n", eqMean, negMean, eqMean-negMean)

	intentsAllPass := 0
	for _, m := range perIntentMin {
		if m >= *threshold {
			intentsAllPass++
		}
	}
	fmt.Println()
	fmt.Printf("VERDICT: %d/%d intents have ALL 6 cross-lingual pairs >= %.2f; ", intentsAllPass, len(pairs), *threshold)
	if eqMin > negMax {
		fmt.Printf("equivalents and non-equivalents are SEPARABLE on this set (margin %.3f).\n", eqMin-negMax)
	} else {
		fmt.Printf("distributions OVERLAP by %.3f — a single global threshold can't cleanly split them (motivates per-pair calibration).\n", negMax-eqMin)
	}
}

func statLine(label string, xs []float64) {
	mn, mx, mean := minMaxMean(xs)
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	p50 := sorted[len(sorted)/2]
	fmt.Printf("  %-38s n=%3d  min=%.3f  p50=%.3f  mean=%.3f  max=%.3f\n", label, len(xs), mn, p50, mean, mx)
}

func minMaxMean(xs []float64) (mn, mx, mean float64) {
	mn, mx = math.Inf(1), math.Inf(-1)
	var sum float64
	for _, x := range xs {
		if x < mn {
			mn = x
		}
		if x > mx {
			mx = x
		}
		sum += x
	}
	return mn, mx, sum / float64(len(xs))
}

func countGE(xs []float64, t float64) int {
	n := 0
	for _, x := range xs {
		if x >= t {
			n++
		}
	}
	return n
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
	var er embedResp
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return nil, err
	}
	return er.Embeddings, nil
}

func readPairs(path string) ([]pair, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []pair
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var p pair
		if err := json.Unmarshal(line, &p); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, sc.Err()
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
