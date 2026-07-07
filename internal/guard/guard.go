package guard

import (
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/kaanrumin/polyglot-cache/internal/cache"
)

// Extraction order matters: each stage consumes its matches (replacing them
// with spaces) so a later, looser pattern never re-tokenizes part of an
// earlier, more specific one — a date is not three numbers, an order ID is not
// a number glued to letters.
var (
	reBacktick = regexp.MustCompile("`[^`]+`")
	reSnake    = regexp.MustCompile(`[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+`)
	reCamel    = regexp.MustCompile(`\b[a-z]+(?:[A-Z][a-z0-9]*)+\b`)
	reDotted   = regexp.MustCompile(`\b[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+\b`)

	reHashID = regexp.MustCompile(`#\d+`)
	// Letter-then-digit tokens may be hyphenated (GC-5501, GPT-4); digit-first
	// tokens only count as IDs when the letters are glued on (4562X, 20GB) —
	// a hyphenated digit-word compound (2-year, 30-day) is a number plus a
	// word, not an identifier.
	reAlnumID = regexp.MustCompile(`\b(?:[A-Za-z]+[-_]?\d|\d+[A-Za-z])[A-Za-z0-9_-]*\b`)

	// English ordinals are numbers, not IDs (5th, 2nd, ...).
	reOrdinalEN = regexp.MustCompile(`^(\d+)(?:st|nd|rd|th|ST|ND|RD|TH)$`)

	// Turkish case suffixes attach to numbers and IDs with an apostrophe
	// (100'ü, #4821'e, 15'inde); the suffix is grammatical, not structural.
	// English possessive 's is swept up too, harmlessly.
	reAposSuffix = regexp.MustCompile(`['’][\p{L}]{1,6}`)

	reDateZH    = regexp.MustCompile(`(\d{4})年(\d{1,2})月(\d{1,2})日`)
	reDateYMD   = regexp.MustCompile(`\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b`)
	reDateSlash = regexp.MustCompile(`\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b`)

	reZHNumber = regexp.MustCompile(`[0-9零一二三四五六七八九两十百千万亿]+`)
	reNumber   = regexp.MustCompile(`\d+(?:[.,]\d+)*`)
)

// currencyPatterns maps surface forms across the four target languages to ISO
// codes. Symbols and words both count; matching is case-insensitive and
// stem-based where the language inflects (dolardan, dólares, lirasına).
// Order is load-bearing: each match is consumed, and 美元/欧元/日元 must be
// claimed before the bare 元 in the CNY pattern, which therefore goes last.
var currencyPatterns = []struct {
	re   *regexp.Regexp
	code string
}{
	{regexp.MustCompile(`(?i)\$|\busd\b|\bdolla?rs?\b|\bdólar\p{L}*|\bdolar\p{L}*|美元|美金`), "USD"},
	{regexp.MustCompile(`(?i)€|\beur\b|\beuros?\b|\bavro\p{L}*|欧元`), "EUR"},
	{regexp.MustCompile(`(?i)£|\bgbp\b|\bpounds?\b|\bsterlin\p{L}*|\blibras?\b|英镑`), "GBP"},
	{regexp.MustCompile(`(?i)\bjpy\b|\byen\b|円|日元`), "JPY"},
	{regexp.MustCompile(`(?i)₺|\btl\b|\blira\p{L}*|里拉`), "TRY"},
	{regexp.MustCompile(`(?i)¥|￥|\bcny\b|\brmb\b|\byuan\b|元|人民币`), "CNY"},
}

// Extract parses the structural tokens of a prompt: locale-normalized numbers,
// dates, currencies, IDs, and code identifiers. lang is the detected language
// of the prompt and drives the locale rules (decimal separator, date order).
func Extract(text, lang string) cache.StructuralTokens {
	var t cache.StructuralTokens
	s := normalizeWidth(text)
	s = reAposSuffix.ReplaceAllString(s, " ")

	// Code identifiers.
	s = consume(s, reBacktick, func(m string) {
		t.CodeIdents = append(t.CodeIdents, strings.Trim(m, "`"))
	})
	// Dotted paths first: auth_service.py must not be split into a snake_case
	// stem plus a stray extension.
	for _, re := range []*regexp.Regexp{reDotted, reSnake, reCamel} {
		s = consume(s, re, func(m string) {
			t.CodeIdents = append(t.CodeIdents, m)
		})
	}

	// IDs. #-prefixed first, then mixed letter/digit tokens; English ordinals
	// are reclassified as numbers.
	s = consume(s, reHashID, func(m string) {
		t.IDs = append(t.IDs, strings.TrimPrefix(m, "#"))
	})
	s = consume(s, reAlnumID, func(m string) {
		if ord := reOrdinalEN.FindStringSubmatch(m); ord != nil {
			t.Numbers = append(t.Numbers, ord[1])
			return
		}
		t.IDs = append(t.IDs, strings.ToUpper(m))
	})

	// Dates, normalized to ISO yyyy-mm-dd.
	s = consumeDates(s, lang, &t.Dates)

	// Numbers: Chinese numerals (一百, 3万5千) first, then locale-aware digit
	// groups (1.000,50 vs 1,000.50).
	s = consumeZHNumbers(s, &t.Numbers)
	s = consume(s, reNumber, func(m string) {
		if v, ok := normalizeNumber(m, lang); ok {
			t.Numbers = append(t.Numbers, v)
		}
	})

	// Currencies scan what remains: the amounts were consumed above, but the
	// symbol or word survives in place. Matches are consumed so a compound
	// form (美元) is never re-counted by a later, looser pattern (元).
	for _, cp := range currencyPatterns {
		if cp.re.MatchString(s) {
			t.Currencies = append(t.Currencies, cp.code)
			s = cp.re.ReplaceAllString(s, " ")
		}
	}

	sortDedup(&t.Numbers)
	sortDedup(&t.IDs)
	sortDedup(&t.Currencies)
	sortDedup(&t.Dates)
	sortDedup(&t.CodeIdents)
	return t
}

// Compare reports whether two prompts are structurally equivalent. It returns
// ok=false and the first differing category when any normalized token set
// differs — the caller must then reject the semantic match and fall through to
// the real LLM. Both sides must come from Extract (sorted, deduplicated).
func Compare(a, b cache.StructuralTokens) (ok bool, category string) {
	switch {
	case !equal(a.Numbers, b.Numbers):
		return false, "numbers"
	case !equal(a.Currencies, b.Currencies):
		return false, "currencies"
	case !equal(a.Dates, b.Dates):
		return false, "dates"
	case !equal(a.IDs, b.IDs):
		return false, "ids"
	case !equal(a.CodeIdents, b.CodeIdents):
		return false, "code_idents"
	}
	return true, ""
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sortDedup(xs *[]string) {
	s := *xs
	sort.Strings(s)
	out := s[:0]
	for i, x := range s {
		if i == 0 || x != s[i-1] {
			out = append(out, x)
		}
	}
	*xs = out
}

// consume collects every match of re in s via fn and blanks the matched spans
// so later stages cannot re-tokenize them.
func consume(s string, re *regexp.Regexp, fn func(m string)) string {
	return re.ReplaceAllStringFunc(s, func(m string) string {
		fn(m)
		return " "
	})
}

// consumeZHNumbers extracts Chinese numeral phrases. Plain digit runs match
// the pattern too but stay in place for the locale-aware pass, and an
// article-一 before a classifier (一个 = "a") stays in place because the
// English side never extracts a number from "a" either.
func consumeZHNumbers(s string, numbers *[]string) string {
	locs := reZHNumber.FindAllStringIndex(s, -1)
	if locs == nil {
		return s
	}
	var b strings.Builder
	prev := 0
	for _, loc := range locs {
		m := s[loc[0]:loc[1]]
		if !containsHanNumeral(m) || isArticleYi(m, s[loc[1]:]) {
			continue
		}
		if v, ok := parseZHNumber(m); ok {
			*numbers = append(*numbers, formatNumber(v))
		}
		b.WriteString(s[prev:loc[0]])
		b.WriteByte(' ')
		prev = loc[1]
	}
	b.WriteString(s[prev:])
	return b.String()
}

// consumeDates extracts the three supported date shapes. Ambiguous dd/mm vs
// mm/dd order is resolved by the prompt's locale — English reads month-first,
// Turkish and Spanish day-first — and a component >12 overrides the
// convention. Candidates that fail validation (month 13) are left in place
// for the number pass.
func consumeDates(s, lang string, dates *[]string) string {
	type pattern struct {
		re   *regexp.Regexp
		norm func(m []string) (string, bool)
	}
	ymd := func(m []string) (string, bool) { return isoDate(m[1], m[2], m[3]) }
	patterns := []pattern{
		{reDateZH, ymd},
		{reDateYMD, ymd},
		{reDateSlash, func(m []string) (string, bool) {
			a, b, y := m[1], m[2], m[3]
			monthFirst := lang == "en"
			if atoi(a) > 12 {
				monthFirst = false
			} else if atoi(b) > 12 {
				monthFirst = true
			}
			if monthFirst {
				return isoDate(y, a, b)
			}
			return isoDate(y, b, a)
		}},
	}
	for _, p := range patterns {
		s = p.re.ReplaceAllStringFunc(s, func(whole string) string {
			m := p.re.FindStringSubmatch(whole)
			iso, ok := p.norm(m)
			if !ok {
				return whole
			}
			*dates = append(*dates, iso)
			return " "
		})
	}
	return s
}

func isoDate(y, m, d string) (string, bool) {
	yi, mi, di := atoi(y), atoi(m), atoi(d)
	if mi < 1 || mi > 12 || di < 1 || di > 31 || yi < 1900 || yi > 2100 {
		return "", false
	}
	return strconv.Itoa(yi) + "-" + pad2(mi) + "-" + pad2(di), true
}

func pad2(n int) string {
	if n < 10 {
		return "0" + strconv.Itoa(n)
	}
	return strconv.Itoa(n)
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}

// normalizeNumber canonicalizes a digit token under the prompt's locale:
// Turkish and Spanish write 1.000,50 where English writes 1,000.50. When both
// separators appear, the last one is the decimal mark regardless of locale;
// with a single kind of separator the locale convention decides.
func normalizeNumber(tok, lang string) (string, bool) {
	lastDot := strings.LastIndexByte(tok, '.')
	lastComma := strings.LastIndexByte(tok, ',')
	commaDecimal := lang == "tr" || lang == "es"

	var intPart, fracPart string
	switch {
	case lastDot >= 0 && lastComma >= 0:
		// Both present: the later one is the decimal mark.
		if lastDot > lastComma {
			intPart, fracPart = tok[:lastDot], tok[lastDot+1:]
		} else {
			intPart, fracPart = tok[:lastComma], tok[lastComma+1:]
		}
	case lastComma >= 0:
		// tr/es: comma is the decimal mark (1,5) unless repeated (1,000,000
		// pasted in English style). en/zh: groups of three are thousands.
		if strings.Count(tok, ",") > 1 || (!commaDecimal && isGrouped(tok, ',')) {
			intPart = tok
		} else {
			intPart, fracPart = tok[:lastComma], tok[lastComma+1:]
		}
	case lastDot >= 0:
		// tr/es: groups of three are thousands (1.000). en/zh: dot is the
		// decimal mark unless repeated.
		if strings.Count(tok, ".") > 1 || (commaDecimal && isGrouped(tok, '.')) {
			intPart = tok
		} else {
			intPart, fracPart = tok[:lastDot], tok[lastDot+1:]
		}
	default:
		intPart = tok
	}

	digits := strings.NewReplacer(".", "", ",", "").Replace(intPart)
	if fracPart == "" {
		fracPart = "0"
	}
	v, err := strconv.ParseFloat(digits+"."+fracPart, 64)
	if err != nil {
		return "", false
	}
	return formatNumber(v), true
}

// isGrouped reports whether tok is digits separated by sep in groups of three
// (1.000 / 12.345.678) — the unambiguous thousands-separator shape.
func isGrouped(tok string, sep byte) bool {
	parts := strings.Split(tok, string(sep))
	if len(parts) < 2 || len(parts[0]) == 0 || len(parts[0]) > 3 {
		return false
	}
	for _, p := range parts[1:] {
		if len(p) != 3 {
			return false
		}
	}
	return true
}

func formatNumber(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// normalizeWidth folds full-width digits and punctuation (１００，５０) to
// their ASCII forms so one tokenizer serves all four scripts.
func normalizeWidth(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r >= '０' && r <= '９':
			r = '0' + (r - '０')
		case r == '，':
			r = ','
		case r == '．':
			r = '.'
		case r == '＃':
			r = '#'
		}
		b.WriteRune(r)
	}
	return b.String()
}
