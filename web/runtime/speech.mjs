// Text to mouth shapes: kana by their vowel, Latin letters for Spanish, English and similar. null is a closed mouth.
const KANA = {
  a: "あぁかがさざただなはばぱまやゃらわゎ",
  i: "いぃきぎしじちぢにひびぴみりゐ",
  u: "うぅくぐすずつづぬふぶぷむゆゅるゔ",
  e: "えぇけげせぜてでねへべぺめれゑ",
  o: "おぉこごそぞとどのほぼぽもよょろを",
};
const VOWEL_OF = {};
for (const [vowel, chars] of Object.entries(KANA)) for (const c of chars) VOWEL_OF[c] = vowel;
// Small kana join the mora before them: きゃ is one mora sounding "a".
const SMALL = "ぁぃぅぇぉゃゅょゎ";
const LIPS = "mbpfv";
const PAUSE = /[\s、。,.!?！？…]/;

// [vowel or null, length in morae]
export function morae(text) {
  const out = [];
  let closure = false;
  for (const raw of text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()) {
    const code = raw.charCodeAt(0);
    const c = code >= 0x30a1 && code <= 0x30f6 ? String.fromCharCode(code - 0x60) : raw;
    const last = out[out.length - 1];
    if (c === "ー") {
      if (last) last[1] += 1;
    } else if (SMALL.includes(c) && last?.[0]) {
      last[0] = VOWEL_OF[c];
    } else if (VOWEL_OF[c]) {
      out.push([VOWEL_OF[c], 1]);
    } else if (c === "ん" || c === "っ") {
      out.push([null, 1]);
    } else if ("aeiou".includes(c)) {
      if (closure) out.push([null, 0.4]);
      closure = false;
      out.push([c, 1]);
    } else if (c === "y" && !"aeiou".includes(last?.[0] ?? "")) {
      out.push(["i", 1]);
    } else if (LIPS.includes(c)) {
      closure = true;
    } else if (PAUSE.test(c)) {
      closure = false;
      if (last && last[0] === null) last[1] = Math.max(last[1], c === " " ? 0.5 : 1.5);
      else out.push([null, c === " " ? 0.5 : 1.5]);
    }
  }
  return out;
}

// A clip keying mouth params at each mora center. mouth(level, vowel) gives the param values for one shape.
export function speechClip(text, mouth, rate = 8) {
  const keys = {};
  let time = 0.05;
  const key = (t, values) => {
    for (const [name, v] of Object.entries(values)) (keys[name] ??= []).push([t, v]);
  };
  const shut = mouth(0, "a");
  key(0, shut);
  for (const [vowel, length] of morae(text)) {
    const duration = length / rate;
    key(time + duration * 0.45, vowel ? mouth(0.85 + 0.15 * Math.random(), vowel) : shut);
    time += duration;
  }
  key(time + 0.1, shut);
  return { duration: time + 0.1, loop: false, tracks: Object.entries(keys).map(([param, k]) => ({ param, keys: k, linear: true })) };
}
