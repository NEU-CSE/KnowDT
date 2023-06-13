def read_file(file_name, dec_type="Greedy"):
    f = open(file_name, "r", encoding="utf-8")

    refs = []
    cands = []
    dec_str = f"{dec_type}:"

    for i, line in enumerate(f.readlines()):
        if i == 1:
            _, ppl, _, acc = line.strip("EVAL	Loss	PPL	Accuracy").split()
            #print(f"PPL: {ppl}\tAccuracy: {float(acc)*100}%")
        if line.startswith(dec_str):
            exp = line.strip(dec_str).strip("\n")
            cands.append(exp)
        if line.startswith("Ref:"):
            ref = line.strip("Ref:").strip("\n")
            refs.append(ref)

    return refs, cands, float(ppl), float(acc)


def process(file_name):
    refs, cands, p, a = read_file(file_name)
    f = open(f"data/hyps.txt", "w", encoding="utf-8")
    for line in cands:
        f.write(line + '\n')
    f.close()

    f = open(f"data/refs.txt", "w", encoding="utf-8")
    for line in refs:
        f.write(line + '\n')
    f.close()

