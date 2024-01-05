def read_scores_from_file(filename):
    scores = {}

    with open(filename, 'r') as file:
        n = int(next(file).strip())

        for _ in range(n):
            v, j_count = map(int, next(file).strip().split(" ")[:2])
            scores[v] = {}

            for _ in range(j_count):
                line = next(file).strip().split(" ")
                scores[v][frozenset(map(int, line[2:]))] = float(line[0])

    return scores
