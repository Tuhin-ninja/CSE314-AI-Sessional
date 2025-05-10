import csv
import matplotlib.pyplot as plt
from pylatex import Document, Section, Figure, NoEscape
from pylatex.utils import NoEscape  # Import escape function for LaTeX special characters

# Read CSV data, handling multi-row header
csv_file = '2105002.csv'
data = {}
with open(csv_file, 'r') as f:
    # Read all lines to manually handle the header
    lines = f.readlines()

# Debug: Print the raw lines to verify CSV content
print("Raw CSV lines:")
for i, line in enumerate(lines):
    print(f"Line {i}: {line.strip()}")

# Use the third row (subheader) as the effective fieldnames, aligning with data
header_row = lines[2].strip().split(',')
fieldnames = ['Name', '|V| or n', '|E| or m', 'Simple Randomized or Randomized-1', 'Simple Greedy or Greedy-1', 'Semi-greedy-1', 'Local Iterations', 'Average Value', 'GRASP Iterations', 'Best Value', 'Known best solution or upper bound']
data_rows = lines[3:]  # Data starts from the fourth line

# Debug: Print the data rows to verify
print("Data rows to process:")
for row in data_rows:
    print(row.strip())

# Parse data rows using csv.DictReader
reader = csv.DictReader(data_rows, fieldnames=fieldnames)
for row in reader:
    graph = row['Name']
    if not graph:  # Skip empty rows
        continue
    # Only process graphs G1 to G10
    graph_num = int(graph[1:]) if graph.startswith('G') and graph[1:].isdigit() else 0
    if not (1 <= graph_num <= 10):
        continue
    try:
        data[graph] = {
            'Randomized': int(row['Simple Randomized or Randomized-1']),
            'Greedy': int(row['Simple Greedy or Greedy-1']),
            'Semi-Greedy': int(row['Semi-greedy-1']),
            'Local Search': int(row['Average Value']),
            'GRASP': int(row['Best Value']),
            'Known Best': int(row['Known best solution or upper bound']) if row['Known best solution or upper bound'] else None,
            'Local Iterations': int(row['Local Iterations']),
            'GRASP Iterations': int(row['GRASP Iterations'])

        }
    except ValueError as e:
        print(f"Warning: Invalid value in row for graph {graph}. Assigning default values. Error: {e}")
        data[graph] = {
            'Randomized': 0,
            'Greedy': 0,
            'Semi-Greedy': 0,
            'Local Search': 0,
            'GRASP': 0,
            'Known Best': None
        }

# Generate plot
graphs = list(data.keys())
algorithms = ['Randomized', 'Greedy', 'Semi-Greedy', 'Local Search', 'GRASP']
values = {algo: [data[graph][algo] for graph in graphs] for algo in algorithms}
print(values)

plt.figure(figsize=(10, 6))
bar_width = 0.15
index = range(len(graphs))

for i, algo in enumerate(algorithms):
    plt.bar([j + i * bar_width for j in index], values[algo], bar_width, label=algo)

plt.xlabel('Graph')
plt.ylabel('Max Cut Value')
plt.title('Max Cut (Graph 1-10)')
plt.xticks([i + bar_width * 2 for i in index], graphs)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save plot
plot_file = 'max_cut_plot.png'
plt.savefig(plot_file)
plt.close()

# Create LaTeX document
doc = Document('report')
doc.preamble.append(NoEscape(r'\usepackage[utf8]{inputenc}'))
doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))
doc.preamble.append(NoEscape(r'\usepackage{amsmath}'))
doc.preamble.append(NoEscape(r'\usepackage{caption}'))
doc.preamble.append(NoEscape(r'\usepackage{subcaption}'))
doc.preamble.append(NoEscape(r'\usepackage[margin=1in]{geometry}'))
doc.preamble.append(NoEscape(r'\title{Report : Max-Cut Problem Analysis}'))
doc.preamble.append(NoEscape(r'\author{Khalid Hasan Tuhin - 2105002}'))
doc.preamble.append(NoEscape(r'\date{\today}'))

doc.append(NoEscape(r'\maketitle'))

with doc.create(Section('High-Level Algorithm Descriptions')):
    doc.append('The following algorithms were implemented to solve the Max-Cut problem:')
    doc.append(NoEscape(r'\begin{itemize}'))
    doc.append(NoEscape(r'\item \textbf{Randomized}: Assigns vertices to partitions \(X\) or \(Y\) randomly with equal probability, averaging results over multiple iterations.'))
    doc.append(NoEscape(r'\item \textbf{Greedy}: Starts with the edge of maximum weight, greedily assigning remaining vertices to maximize the current cut.'))
    doc.append(NoEscape(r'\item \textbf{Semi-Greedy}: Uses a restricted candidate list based on a greedy function and random selection, balancing greediness and randomness.'))
    doc.append(NoEscape(r'\item \textbf{Local Search}: Improves an initial solution by iteratively moving vertices to the partition that maximizes the cut until no further improvement is possible.'))
    doc.append(NoEscape(r'\item \textbf{GRASP}: Combines semi-greedy construction with local search, iterating multiple times to find the best solution.'))
    doc.append(NoEscape(r'\end{itemize}'))

with doc.create(Section('Comparison of Algorithms')):
    doc.append('Based on the results, the performance of algorithms varies across graphs:')
    doc.append(NoEscape(r'\begin{itemize}'))
    for graph in data:
        doc.append(NoEscape(r'\item For graph ' + NoEscape(graph) + ':'))
        doc.append(NoEscape(r'\begin{itemize}'))
        doc.append(NoEscape(r'\item Randomized: ' + NoEscape(str(data[graph]['Randomized']))))
        doc.append(NoEscape(r'\item Greedy: ' + NoEscape(str(data[graph]['Greedy']))))
        doc.append(NoEscape(r'\item Semi-Greedy: ' + NoEscape(str(data[graph]['Semi-Greedy']))))
        doc.append(NoEscape(r'\item Local Search: ' + NoEscape(str(data[graph]['Local Search'])) + ' (after ' + NoEscape(str(data[graph].get('Local Iterations', 'N/A'))) + ' iterations)'))
        doc.append(NoEscape(r'\item GRASP: ' + NoEscape(str(data[graph]['GRASP'])) + ' (after ' + NoEscape(str(data[graph].get('GRASP Iterations', 'N/A'))) + ' iterations)'))
        doc.append(NoEscape(r'\item Known Best: ' + NoEscape(str(data[graph]['Known Best'] if data[graph]['Known Best'] else 'N/A'))))
        doc.append(NoEscape(r'\end{itemize}'))
    doc.append(NoEscape(r'\end{itemize}'))
    doc.append('GRASP tends to perform closest to the known best solutions across graphs, indicating its effectiveness. Local Search values are notably lower, suggesting potential issues with local optima or initial partitions. Randomized consistently underperforms due to its lack of optimization.')

with doc.create(Section('Visualization')):
    with doc.create(Figure(position='h!')) as plot:
        plot.add_image(plot_file, width=NoEscape(r'0.8\textwidth'))
        plot.add_caption('Max Cut Values for Graphs G1-G10')

# Generate PDF
doc.generate_pdf('2105002', clean_tex=False)
print(f"Report generated as 2105002.pdf")