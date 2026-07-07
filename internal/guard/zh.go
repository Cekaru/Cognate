package guard

import "strings"

// Chinese numeral support: 一百 must equal 100, 3万5千 must equal 35000, or the
// guard silently under-protects the language it most needs to cover.

const hanNumerals = "零一二三四五六七八九两十百千万亿"

var hanDigits = map[rune]float64{
	'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
	'五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
}

var hanSmallUnits = map[rune]float64{'十': 10, '百': 100, '千': 1000}

var hanBigUnits = map[rune]float64{'万': 1e4, '亿': 1e8}

func containsHanNumeral(s string) bool {
	return strings.ContainsAny(s, hanNumerals)
}

// classifiers are the common Chinese measure words. A bare 一 in front of one
// (一个包裹) is the indefinite article — "a package", not "1 package" — and
// English never extracts a number from "a", so neither may the Chinese side,
// or every 一个/一件 phrase falsely mismatches its cross-lingual equivalent.
const classifiers = "个件张条本只封块盒台部次位名场道杯瓶篇辆颗座间家匹页笔幅双对份段句首曲"

// isArticleYi reports whether the numeral phrase is exactly 一 immediately
// followed by a classifier in the surrounding text.
func isArticleYi(phrase, rest string) bool {
	if phrase != "一" || rest == "" {
		return false
	}
	return strings.ContainsRune(classifiers, []rune(rest)[0])
}

// parseZHNumber evaluates a numeral phrase that may mix Han numerals, unit
// characters, and ASCII digits: 一百零五 = 105, 二十三 = 23, 3万5千 = 35000,
// 一亿二千万 = 120 000 000. It returns ok=false for the empty string; a bare
// unit like 十 is worth 10, matching colloquial use.
func parseZHNumber(s string) (float64, bool) {
	if s == "" {
		return 0, false
	}
	var total, section, num float64
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9':
			num = num*10 + float64(r-'0')
		case hanDigits[r] != 0 || r == '零':
			num = num*10 + hanDigits[r]
		case hanSmallUnits[r] != 0:
			if num == 0 {
				num = 1 // 十五 = 15, not 05
			}
			section += num * hanSmallUnits[r]
			num = 0
		case hanBigUnits[r] != 0:
			section += num
			num = 0
			if section == 0 {
				section = 1 // bare 万 = 10 000
			}
			total += section * hanBigUnits[r]
			section = 0
		default:
			return 0, false
		}
	}
	return total + section + num, true
}
