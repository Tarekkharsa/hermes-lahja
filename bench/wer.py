"""Word error rate for Arabic: strips diacritics/punctuation, folds alef/ya/ta-marbuta variants."""
import re, sys, unicodedata
DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
FOLD = str.maketrans({"أ":"ا","إ":"ا","آ":"ا","ٱ":"ا","ى":"ي","ؤ":"و","ئ":"ي","ة":"ه","گ":"ك"})
def norm(t):
    t = unicodedata.normalize("NFKC", t)
    t = DIAC.sub("", t).translate(FOLD)
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()
def edits(a, b):
    d = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j]+1, d[j-1]+1, p+(x != y))
    return d[len(b)]
if __name__ == "__main__":
    refs = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
    for sysdir in sys.argv[2:]:
        E = N = 0; rows = []
        for k, (_, dia, ref) in refs.items():
            hyp = open(f"{sysdir}/{k}.txt", encoding="utf-8").read()
            r, h = norm(ref), norm(hyp); e = edits(r, h); E += e; N += len(r)
            rows.append(f"{dia}:{e/len(r):.2f}")
        print(f"{sysdir.split('/')[-1]:8s} WER {E/N:.3f}  ", " ".join(rows))
