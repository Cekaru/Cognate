package lang

import "testing"

func TestDetect(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"chinese", "法国的首都是什么", ZH},
		{"turkish", "Fransa'nın başkenti neresidir", TR},   // ı, ş
		{"spanish", "¿Cuál es la capital de Francia?", ES}, // ¿, plus later ñ-less but ¿ triggers
		{"spanish_enye", "¿Dónde está España?", ES},
		{"english", "What is the capital of France?", EN},
		{"empty", "", Unknown},
		{"digits_only", "12345", Unknown},
		{"mixed_han_wins", "capital 北京", ZH},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Detect(c.in); got != c.want {
				t.Fatalf("Detect(%q) = %q; want %q", c.in, got, c.want)
			}
		})
	}
}
