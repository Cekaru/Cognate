package guard

import (
	"reflect"
	"testing"
)

// The milestone case from the roadmap: a cross-lingual near-match with a
// different amount must be rejected no matter how high the cosine score is.
func TestRejects100vs1000CrossLingual(t *testing.T) {
	en := Extract("Transfer $100 to my savings account", "en")
	tr := Extract("Tasarruf hesabıma 1000 $ aktar", "tr")
	ok, cat := Compare(en, tr)
	if ok {
		t.Fatalf("guard admitted $100 vs $1000: %+v vs %+v", en, tr)
	}
	if cat != "numbers" {
		t.Fatalf("category = %q, want numbers", cat)
	}
}

// Equivalent prompts in all four languages must extract identical tokens —
// that equality is what lets a cross-lingual hit through the guard.
func TestCrossLingualEquivalence(t *testing.T) {
	prompts := map[string]string{
		"en": "Transfer $1,000.50 to account #4821 before 15/04/2024",
		"tr": "15/04/2024 tarihinden önce #4821 numaralı hesaba 1.000,50 $ aktar",
		"es": "Transfiere $1.000,50 a la cuenta #4821 antes del 15/04/2024",
		"zh": "在2024年4月15日之前向#4821账户转账1,000.50美元",
	}
	ref := Extract(prompts["en"], "en")
	if want := []string{"1000.5"}; !reflect.DeepEqual(ref.Numbers, want) {
		t.Fatalf("en numbers = %v, want %v", ref.Numbers, want)
	}
	if want := []string{"2024-04-15"}; !reflect.DeepEqual(ref.Dates, want) {
		t.Fatalf("en dates = %v, want %v", ref.Dates, want)
	}
	if want := []string{"USD"}; !reflect.DeepEqual(ref.Currencies, want) {
		t.Fatalf("en currencies = %v, want %v", ref.Currencies, want)
	}
	if want := []string{"4821"}; !reflect.DeepEqual(ref.IDs, want) {
		t.Fatalf("en ids = %v, want %v", ref.IDs, want)
	}
	for lang, p := range prompts {
		got := Extract(p, lang)
		if ok, cat := Compare(ref, got); !ok {
			t.Errorf("%s prompt mismatched en on %s: %+v vs %+v", lang, cat, ref, got)
		}
	}
}

func TestNumberNormalization(t *testing.T) {
	cases := []struct {
		text, lang string
		want       []string
	}{
		{"send 1,000.50 now", "en", []string{"1000.5"}},
		{"1.000,50 gönder", "tr", []string{"1000.5"}},
		{"envía 1.000,50", "es", []string{"1000.5"}},
		{"1.000 lira", "tr", []string{"1000"}},        // grouped dot = thousands in tr
		{"1.000 dollars", "en", []string{"1"}},        // decimal dot in en
		{"1,000 dollars", "en", []string{"1000"}},     // grouped comma = thousands in en
		{"1,5 litre su", "tr", []string{"1.5"}},       // decimal comma in tr
		{"3,5 kilómetros", "es", []string{"3.5"}},     // decimal comma in es
		{"1,000,000 people", "en", []string{"1000000"}},
		{"pi is 3.14", "en", []string{"3.14"}},
		{"the 3rd item", "en", []string{"3"}},   // ordinal is a number, not an ID
		{"a 2-year warranty", "en", []string{"2"}}, // digit-word compound is a number, not an ID
	}
	for _, c := range cases {
		got := Extract(c.text, c.lang)
		if !reflect.DeepEqual(got.Numbers, c.want) {
			t.Errorf("Extract(%q, %s).Numbers = %v, want %v", c.text, c.lang, got.Numbers, c.want)
		}
	}
}

func TestChineseNumerals(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"转账一百元", []string{"100"}},
		{"转账100元", []string{"100"}},
		{"一百零五个", []string{"105"}},
		{"二十三度", []string{"23"}},
		{"3万5千公里", []string{"35000"}},
		{"一亿二千万人", []string{"120000000"}},
		{"两千块钱的两倍", []string{"2", "2000"}},
		{"１００件", []string{"100"}}, // full-width digits
		// Article-一: 一个 is "a", not the number 1 — the English equivalent
		// ("ship a 25 kg package") extracts no number from "a" either.
		{"寄一个 25 kg 的包裹到德国", []string{"25"}},
		{"我要一杯咖啡", nil},
		{"十一个人", []string{"11"}}, // 十一 is a real numeral, not an article
	}
	for _, c := range cases {
		got := Extract(c.text, "zh")
		if !reflect.DeepEqual(got.Numbers, c.want) {
			t.Errorf("Extract(%q).Numbers = %v, want %v", c.text, got.Numbers, c.want)
		}
	}
}

func TestDates(t *testing.T) {
	cases := []struct {
		text, lang string
		want       []string
	}{
		{"due 04/15/2024", "en", []string{"2024-04-15"}},   // en month-first
		{"15/04/2024 tarihinde", "tr", []string{"2024-04-15"}}, // tr day-first
		{"el 15/04/2024", "es", []string{"2024-04-15"}},
		{"15.04.2024 tarihli fatura", "tr", []string{"2024-04-15"}}, // tr dotted date
		{"2024-04-15 deadline", "en", []string{"2024-04-15"}},
		{"2024年4月15日之前", "zh", []string{"2024-04-15"}},
		// Same written form, genuinely different dates by locale — the guard
		// must keep them apart, not paper over them.
		{"03/05/2024", "en", []string{"2024-03-05"}},
		{"03/05/2024", "tr", []string{"2024-05-03"}},
	}
	for _, c := range cases {
		got := Extract(c.text, c.lang)
		if !reflect.DeepEqual(got.Dates, c.want) {
			t.Errorf("Extract(%q, %s).Dates = %v, want %v", c.text, c.lang, got.Dates, c.want)
		}
	}
	// An invalid candidate (month 13) is not a date; its digits become numbers.
	got := Extract("13/13/2024", "en")
	if len(got.Dates) != 0 {
		t.Errorf("13/13/2024 parsed as date: %v", got.Dates)
	}
}

func TestCurrencies(t *testing.T) {
	cases := []struct {
		text, lang string
		want       []string
	}{
		{"costs 100 dollars", "en", []string{"USD"}},
		{"cuesta 100 dólares", "es", []string{"USD"}},
		{"100 dolar tutuyor", "tr", []string{"USD"}},
		{"价格是100美元", "zh", []string{"USD"}},
		{"价格是100元", "zh", []string{"CNY"}},
		{"100 TL gönder", "tr", []string{"TRY"}},
		{"500 lira borcum var", "tr", []string{"TRY"}},
		{"pay in euros and dollars", "en", []string{"EUR", "USD"}},
		{"兑换100日元", "zh", []string{"JPY"}},
	}
	for _, c := range cases {
		got := Extract(c.text, c.lang)
		if !reflect.DeepEqual(got.Currencies, c.want) {
			t.Errorf("Extract(%q).Currencies = %v, want %v", c.text, got.Currencies, c.want)
		}
	}
}

func TestIDsAndCodeIdents(t *testing.T) {
	got := Extract("run `make build` after fixing getUserById in auth_service.py, see ticket TRX-9981 and #123", "en")
	if want := []string{"123", "TRX-9981"}; !reflect.DeepEqual(got.IDs, want) {
		t.Errorf("IDs = %v, want %v", got.IDs, want)
	}
	if want := []string{"auth_service.py", "getUserById", "make build"}; !reflect.DeepEqual(got.CodeIdents, want) {
		t.Errorf("CodeIdents = %v, want %v", got.CodeIdents, want)
	}

	// Turkish case suffixes on IDs are grammatical, not structural.
	tr := Extract("TRX-9981'e bak ve #123'ü kapat", "tr")
	if ok, cat := Compare(got, tr); !ok && (cat == "ids") {
		t.Errorf("suffixed TR ids mismatched: %v vs %v", got.IDs, tr.IDs)
	}
	if want := []string{"123", "TRX-9981"}; !reflect.DeepEqual(tr.IDs, want) {
		t.Errorf("TR IDs = %v, want %v", tr.IDs, want)
	}
}

func TestCompareMismatchCategories(t *testing.T) {
	cases := []struct {
		a, b, aLang, bLang, wantCat string
	}{
		{"transfer $100", "transfer $1000", "en", "en", "numbers"},
		{"send 100 dollars", "100 euro gönder", "en", "tr", "currencies"},
		{"due 2024-04-15", "due 2024-04-16", "en", "en", "dates"},
		{"close ticket #123", "close ticket #124", "en", "en", "ids"},
		{"call get_user", "call get_users", "en", "en", "code_idents"},
		// Presence vs absence of a structural token is also a mismatch.
		{"transfer $100 to Alice", "transfer money to Alice", "en", "en", "numbers"},
	}
	for _, c := range cases {
		ok, cat := Compare(Extract(c.a, c.aLang), Extract(c.b, c.bLang))
		if ok || cat != c.wantCat {
			t.Errorf("Compare(%q, %q) = (%v, %q), want (false, %q)", c.a, c.b, ok, cat, c.wantCat)
		}
	}
}

func TestIdenticalTokensPass(t *testing.T) {
	a := Extract("What is the capital of France?", "en")
	b := Extract("Fransa'nın başkenti neresidir?", "tr")
	if ok, cat := Compare(a, b); !ok {
		t.Errorf("token-free prompts mismatched on %s: %+v vs %+v", cat, a, b)
	}
}

func TestParseZHNumber(t *testing.T) {
	cases := []struct {
		in   string
		want float64
	}{
		{"一百", 100},
		{"一百零五", 105},
		{"二十三", 23},
		{"十五", 15},
		{"两百", 200},
		{"三千五百", 3500},
		{"3万5千", 35000},
		{"一万", 10000},
		{"万", 10000},
		{"一亿二千万", 120000000},
	}
	for _, c := range cases {
		got, ok := parseZHNumber(c.in)
		if !ok || got != c.want {
			t.Errorf("parseZHNumber(%q) = (%v, %v), want %v", c.in, got, ok, c.want)
		}
	}
}
