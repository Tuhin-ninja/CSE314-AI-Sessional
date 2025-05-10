import random
import csv
import os
from collections import defaultdict
from typing import List, Tuple, Dict


class MaxCutSolver:
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.n = n
        self.edges = edges
        self.graph = defaultdict(list)
        for u, v, w in edges:
            self.graph[u].append((v, w))
            self.graph[v].append((u, w))


    def cut_weight(self, X: set, Y: set) -> int:
        if not X or not Y:
            print(f"Warning: One of the sets is empty! X: {len(X)}, Y: {len(Y)}")
            return 0

        total = 0
        seen = set()

        for u in X:
            for v, w in self.graph[u]:
                # Avoid double-counting
                if v in Y and (u, v) not in seen and (v, u) not in seen:
                    total += w
                    seen.add((u, v))

        return total

    def randomized_maxcut(self, iterations=100) -> int:
        total_weight = 0
        for _ in range(iterations):
            X, Y = set(), set()
            for v in range(self.n):
                (X if random.random() < 0.5 else Y).add(v)
            weight = self.cut_weight(X, Y)
            total_weight += weight
            if weight == 0:
                print("Randomized maxcut failed: Weight is 0")
        return total_weight // iterations

    def greedy_maxcut(self) -> int:
        u, v, _ = max(self.edges, key=lambda e: e[2])
        X, Y = {u}, {v}
        remaining = set(range(self.n)) - {u, v}
        for z in remaining:
            wx = sum(w for y, w in self.graph[z] if y in Y)
            wy = sum(w for x, w in self.graph[z] if x in X)
            (X if wx > wy else Y).add(z)
        weight = self.cut_weight(X, Y)
        if weight == 0:
            print("Greedy maxcut failed: Weight is 0")
        return weight

    def semi_greedy_maxcut(self, alpha=0.5) -> int:
        u, v, _ = max(self.edges, key=lambda e: e[2])
        X, Y = {u}, {v}
        U = set(range(self.n)) - {u, v}
        while U:
            sigma = {}
            for z in U:
                sigmaX = sum(w for y, w in self.graph[z] if y in X)
                sigmaY = sum(w for x, w in self.graph[z] if x in Y)
                sigma[z] = max(sigmaX, sigmaY)
            if not sigma:
                break
            wmin, wmax = min(sigma.values()), max(sigma.values())
            threshold = wmin + alpha * (wmax - wmin)
            RCL = [v for v in U if sigma[v] >= threshold]
            if not RCL:
                chosen = random.choice(list(U))
            else:
                chosen = random.choice(RCL)
            wx = sum(w for y, w in self.graph[chosen] if y in Y)
            wy = sum(w for x, w in self.graph[chosen] if x in X)
            (X if wx > wy else Y).add(chosen)
            U.remove(chosen)
        weight = self.cut_weight(X, Y)
        if weight == 0:
            print("Semi-Greedy maxcut failed: Weight is 0")
        return weight

    def local_search(self, X: set, Y: set) -> Tuple[set, set, int]:
        if not X or not Y:
            print(f"Warning: One of the sets is empty during local search! X: {len(X)}, Y: {len(Y)}")
            return X, Y, 0

        changed = True
        while changed:
            changed = False
            best_delta = 0
            best_move = None
            changed = False

            for v in list(X | Y):
                if v in X:
                    source, target = X, Y
                else:
                    source, target = Y, X

                sigma_source = sum(w for u, w in self.graph[v] if u in target)
                sigma_target = sum(w for u, w in self.graph[v] if u in source)

                delta = sigma_target - sigma_source  # Gain from moving to the other side

                if delta > best_delta:
                    best_delta = delta
                    best_move = (v, source, target)

            if best_move:
                v, source, target = best_move
                if v in source and v not in target:
                    source.remove(v)
                    target.add(v)
                    changed = True

                    # Check if the move leads to an empty set and undo if it does
                    if len(X) == 0 or len(Y) == 0:
                        print(f"Warning: One of the sets is empty after move. Undoing the move.")
                        source.add(v)
                        target.remove(v)
                        changed = False  # No more changes to be made
                        break

        weight = self.cut_weight(X, Y)
        if weight == 0:
            print("Local search failed: Weight is 0")
        return X, Y, weight

    def grasp(self, max_iter=5, alpha=0.5) -> Tuple[int,int]:
        best = 0
        local_search_values=0
        for iteration in range(max_iter):
            try:
                u, v, _ = max(self.edges, key=lambda e: e[2])
                X, Y = {u}, {v}
                U = set(range(self.n)) - {u, v}

                while U:
                    sigma = {}
                    for z in U:
                        sigmaX = sum(w for y, w in self.graph[z] if y in X)
                        sigmaY = sum(w for x, w in self.graph[z] if x in Y)
                        sigma[z] = max(sigmaX, sigmaY)

                    if not sigma:
                        break

                    wmin, wmax = min(sigma.values()), max(sigma.values())
                    threshold = wmin + alpha * (wmax - wmin)
                    RCL = [v for v in U if sigma[v] >= threshold]

                    if not RCL:
                        chosen = random.choice(list(U))
                    else:
                        chosen = random.choice(RCL)

                    wx = sum(w for y, w in self.graph[chosen] if y in Y)
                    wy = sum(w for x, w in self.graph[chosen] if x in X)
                    (X if wx > wy else Y).add(chosen)
                    U.remove(chosen)

                # Check if partition is valid (both sets should be non-empty)
                if not X or not Y:
                    print(f"Warning: Invalid partition detected at iteration {iteration}, skipping local search.")
                    continue  # Skip broken partitions

                print(f"[Iter {iteration}] X size: {len(X)}, Y size: {len(Y)}")
                print(f"Cut before local search: {self.cut_weight(X, Y)}")

                _, _, value = self.local_search(set(X), set(Y))
                local_search_values += value

                if value > best:
                    print(f"Best value updated: {value}")
                    best = value

                print(f"Cut after local search: {value}")

            except Exception as e:
                print(f"[GRASP Iteration {iteration}] Error: {e}")
                continue
        
        local_search_values = local_search_values // max_iter
        return best,local_search_values


def load_graph(filepath: str) -> Tuple[int, List[Tuple[int, int, int]]]:
    with open(filepath, 'r') as f:
        lines = f.readlines()

    print(f"--- Loading Graph from: {filepath} ---")
    print(f"First line: {lines[0].strip()}")
    n, m = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        try:
            u, v, w = map(int, line.strip().split())
            edges.append((u-1, v-1, w))
        except Exception as e:
            print(f"Skipping invalid line: {line.strip()} | Error: {e}")

    print(f"Total edges loaded: {len(edges)}")
    return n,m, edges


def run_and_save_results(input_folder: str, output_csv: str, student_id: str = "2105002", alpha=0.5):
    known_best = {
        'G1': 12078, 'G2': 12084, 'G3': 12077,
        'G11': 627, 'G12': 621, 'G13': 645,
        'G14': 3187, 'G15': 3169, 'G16': 3172,
        'G22': 14123, 'G23': 14129, 'G24': 14131,
        'G32': 1560, 'G33': 1537, 'G34': 1541,
        'G35': 8000, 'G36': 7996, 'G37': 8009,
        'G43': 7027, 'G44': 7022, 'G45': 7020,
        'G48': 6000, 'G49': 6000, 'G50': 5988
    }
    rows = []
    max_iter=5


    for file in sorted(os.listdir(input_folder), key=lambda x: int(x[1:x.index('.')])):
        if file.endswith('.rud') or file.endswith('.txt'):
            path = os.path.join(input_folder, file)
            try:
                print(f"Processing: {file}")
                n,m, edges = load_graph(path)
                solver = MaxCutSolver(n, edges)

                randomized = solver.randomized_maxcut()
                greedy = solver.greedy_maxcut()
                semi = solver.semi_greedy_maxcut(alpha=alpha)
                grasp,local_search_values = solver.grasp(max_iter, alpha=alpha)

                print(f"  Randomized: {randomized}, Greedy: {greedy}, Semi-Greedy: {semi}, GRASP: {grasp}")
                name = os.path.splitext(file)[0].upper()  # 'G1' from 'g1.txt'
                best_known = known_best.get(name, '')
                rows.append([name,n,m, randomized, greedy, semi, max_iter,local_search_values,max_iter,grasp,best_known])
            except Exception as e:
                print(f"Error processing {file}: {e}")



    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
        'Problem', '', '',
        'Constructive algorithm', '', '',
        'Local search', '', '',
        'GRASP', 
        'Known best solution or upper bound'
        ])
        writer.writerow([
        'Name', '|V| or n', '|E| or m',
        'Simple Randomized or Randomized-1', 'Simple Greedy or Greedy-1', 'Semi-greedy-1',
        'Simple local or local-1','',
        'GRASP-1'
        ])

        writer.writerow([
            '','','','','','','No. of iterations','Average Value','No. of iterations', 'Best Value'
        ])
        writer.writerows(rows)
    print(f"Results saved to {output_csv}")



if __name__ == "__main__":
    input_folder = "./graphs"
    output_csv = "2105002.csv"
    run_and_save_results(input_folder,output_csv)