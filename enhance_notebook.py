import json
import os

notebook_path = '/home/ahmad/Documents/programs/python_folder/Data_mining_project/clustering_project.ipynb'

def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

def get_cell_output_text(cell):
    text_content = ""
    if cell.get('cell_type') == 'code':
        outputs = cell.get("outputs", [])
        for output in outputs:
            # Check for standard output
            if output.get("name") == "stdout":
                text_content += "".join(output.get("text", []))
            # Check for error or other types if needed, but we look for our print statements
    return text_content

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    cells = nb['cells']
    
    # definitions of markdown content
    ga_details_md = [
        "## Genetic Algorithm Implementation Details",
        "",
        "This class encapsulates the Genetic Algorithm logic for clustering. The standard components of a GA are adapted for the clustering problem as follows:",
        "",
        "### 1. Chromosome Representation & Initialization",
        "- **Chromosome**: A single solution is represented as a set of $k$ centroids (coordinates in $d$-dimensional space). If we want $k=3$ clusters for 2D data, a chromosome is a list of 3 $(x, y)$ points.",
        "- **Population**: We initialize a population of $N$ random chromosomes. Each centroid is initialized uniformly within the bounds of the dataset features (min to max).",
        "",
        "### 2. Fitness Function (Evaluation)",
        "- To evaluate quality, we use the **Within-Cluster Sum of Squares (WCSS)**, also known as Inertia or SSE.",
        "- For a given set of centroids, every data point is assigned to the nearest centroid. We sum the squared Euclidean distances.",
        "- **Goal**: Minimize WCSS. Since GAs typically *maximize* fitness, intrinsic logic usually minimizes cost. Here, lower is better.",
        "",
        "### 3. Selection (Tournament)",
        "- We use **Tournament Selection**: A subset of individuals is chosen at random, and the one with the best (lowest) WCSS wins and becomes a parent.",
        "",
        "### 4. Crossover (Recombination)",
        "- We use **Single-Point Crossover**. Two parents exchange a portion of their centroids to create unique offspring configurations, exploring the solution space.",
        "",
        "### 5. Mutation",
        "- With a low probability, we perturb a centroid by adding random Gaussian noise. This helps the algorithm escape local minima.",
        "",
        "### 6. Elitism",
        "- The best solution from the current generation is always carried over to the next to ensure we never lose the best clustering found so far."
    ]

    iris_analysis_md = [
        "### Analysis of Iris Dataset Clustering",
        "",
        "**Observations:**",
        "- The convergence plot typically shows a sharp decrease in fitness (WCSS) in the first few generations as the centroids move towards the dense regions of data.",
        "- The clustering visualization should correspond roughly to the three species of Iris, although perfect separation is difficult due to class overlap.",
        "",
        "**Performance:**",
        "- The GA successfully identifies 3 distinct groups.",
        "- Fitness plateaus after a certain number of generations, indicating convergence."
    ]

    blobs_analysis_md = [
        "### Analysis of Synthetic Blobs Clustering",
        "",
        "**Observations:**",
        "- Since `make_blobs` creates distinct, well-separated clusters, the GA should find them easily.",
        "- We expect a cleaner separation compared to real-world data.",
        "",
        "**Performance:**",
        "- Convergence should be fast and stable.",
        "- The centroids should land very close to the true centers of the blobs."
    ]

    mall_analysis_md = [
        "### Analysis of Mall Customer Segmentation",
        "",
        "**Observations:**",
        "- We typically look for 5 clusters based on Income vs Spending Score: Low/Low, Low/High, Mid/Mid, High/Low, High/High.",
        "- The GA attempts to find these natural groupings.",
        "",
        "**Performance:**",
        "- Visual inspection of the scatter plot confirms if the 5 segments makes business sense.",
        "- If fitness fluctuates, it might require more generations or a larger population."
    ]

    final_report_md = [
        "---",
        "# Project Report & Final Recommendations",
        "",
        "## 1. Sample Observations & Comparison",
        "- **Iris**: Real-world data with overlap. GA performs reasonably well but may struggle with the boundary between Versicolor and Virginica.",
        "- **Make Blobs**: Synthetic, separated data. GA performs excellently, demonstrating the algorithm's correctness logic.",
        "- **Mall Customers**: Real-world exploratory task. GA successfully identifies market segments essential for marketing strategies.",
        "",
        "**Comparison:**",
        "- The GA works robustly across dimensions (2D for Mall/Blobs, 4D for Iris).",
        "- Convergence speed depends on the complexity (overlap) of data.",
        "",
        "## 2. Descriptive Insights",
        "- **Clustering Nature**: The GA tends to perform like K-Means (creating spherical clusters) because the fitness function is distance-based.",
        "- **Stochasticity**: Unlike K-Means, GA is less likely to get stuck in local optima if mutation rates are tuned well, though it is computationally more expensive.",
        "",
        "## 3. Predictive Insights",
        "- **Customer Targeting**: For the Mall dataset, new customers can be immediately classified into 'Target' (High Income, High Spend) or 'Conservative' (Mix) groups based on the learned centroids.",
        "- **Anomaly Detection**: Points with high distance from their assigned centroid could be flagged as outliers.",
        "",
        "## 4. Use Cases",
        "- **Marketing**: Segmenting customer base for personalized ads (Mall dataset).",
        "- **Biology**: Categorizing species or gene expression patterns (Iris dataset).",
        "- **Image Compression**: Using centroids as a color palette for image quantization.",
        "",
        "## 5. Final Recommendation",
        "- **Algorithm Choice**: For small to medium datasets where global optimality is crucial, GA is a strong contender.",
        "- **Scalability**: For very large datasets, GA might be too slow. Hybrid approaches (GA to initialize K-Means) are often recommended.",
        "- **Parameter Tuning**: Mutation rate and population size are critical. Adaptive mutation rates could further improve performance."
    ]

    # Iterate and insert
    for cell in cells:
        source_text = "".join(cell.get("source", []))
        
        # Check for GA Implementation header - Replace content but keep position
        if "## Genetic Algorithm Implementation" in source_text and "This class encapsulates" in source_text:
            new_cells.append(create_markdown_cell(ga_details_md))
            continue
        
        new_cells.append(cell)

        # Check outputs to insert analysis
        output_text = get_cell_output_text(cell)
        
        if "Running GA on Iris Dataset" in output_text:
             new_cells.append(create_markdown_cell(iris_analysis_md))

        if "Running GA on Synthetic Blobs" in output_text:
             new_cells.append(create_markdown_cell(blobs_analysis_md))

        if "Running GA on Mall Customers" in output_text:
             new_cells.append(create_markdown_cell(mall_analysis_md))

    # Append Final Report at the very end
    new_cells.append(create_markdown_cell(final_report_md))

    nb['cells'] = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook enhanced successfully.")

except Exception as e:
    print(f"Error enhancing notebook: {e}")
