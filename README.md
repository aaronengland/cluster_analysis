# Customer Segmentation via Cluster Analysis

A customer segmentation pipeline that transforms raw e-commerce transaction data into interpretable customer segments using unsupervised machine learning. I built RFM (Recency, Frequency, Monetary) features along with additional behavioral features, applied K-Means clustering, and profiled the resulting segments to support actionable marketing strategies.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [0. Data Collection](#0-data-collection)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Preprocessing](#2-preprocessing)
  - [3. K-Means Clustering](#3-k-means-clustering)
  - [4. Cluster Profiling](#4-cluster-profiling)
  - [5. Model Comparison](#5-model-comparison)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [Getting Started](#getting-started)

---

## Overview

The goal of this project is to segment customers into distinct groups based on their purchasing behavior. Customer segmentation is one of the most common applications of unsupervised learning in business -- it allows a company to tailor marketing, promotions, and retention strategies to different types of customers rather than treating them all the same.

I approached this as a full end-to-end pipeline:

- **RFM feature engineering** to summarize each customer's purchasing behavior into meaningful metrics
- **Log transformation and z-score standardization** so that distance-based clustering algorithms treat all features fairly
- **Elbow method and silhouette analysis** to select the optimal number of clusters
- **Comprehensive cluster profiling** with radar charts, box plots, product analysis, and purchase timing
- **Model comparison** across K-Means, Agglomerative (hierarchical), and DBSCAN to validate the chosen approach
- **Serialized preprocessing and model artifacts** via joblib for reproducible inference

---

## Project Structure

```
cluster_analysis/
├── 00_data_collection/            # Download dataset and save to S3
│   ├── notebook.ipynb
│   └── output/
├── 01_eda/                        # Exploratory data analysis
│   ├── notebook.ipynb
│   └── output/                    # EDA visualizations
├── 02_preprocessing/              # Clean data, engineer customer features
│   ├── notebook.ipynb
│   └── output/                    # Feature distributions, summaries
├── 03_clustering/                 # K-Means model fitting
│   ├── notebook.ipynb
│   └── output/                    # Models, tuning results, plots
├── 04_profiling/                  # Cluster interpretation and profiling
│   ├── notebook.ipynb
│   └── output/                    # Radar charts, box plots, product analysis
├── 05_comparison/                 # K-Means vs Agglomerative vs DBSCAN
│   ├── notebook.ipynb
│   └── output/                    # Comparison visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

I used the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset from the UCI Machine Learning Repository. It contains real transactional data from a UK-based online retailer that sells unique all-occasion gift items. Many of the customers are wholesalers.

| Attribute | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Transactions | ~1,067,000 line items |
| Customers | ~4,300 (after cleaning) |
| Period | December 2009 -- December 2011 |
| Countries | 43 |
| Columns | Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country |

The raw data is stored as a parquet file in S3 (`s3://cluster-analysis-demo/00_data_collection/data.parquet`). I never save data files locally -- only small tables, models, and visualizations are stored in each notebook's `output/` directory.

---

## Methodology

### 0. Data Collection

I downloaded the Online Retail II dataset directly from UCI, which comes as a zip file containing an Excel spreadsheet. I extracted it, cleaned the column names to snake_case, cast mixed-type columns (like `invoice` and `stockcode`) to string so they play nicely with parquet, and saved the result to S3.

### 1. Exploratory Data Analysis

Before building any model, I needed to understand the data. The EDA notebook examines the raw transaction-level data to uncover patterns, anomalies, and data quality issues.

**What I found:**

- **Missing values**: `customer_id` has the highest missingness (~25%), which is expected -- guest checkouts don't have IDs. `description` has minor missingness. All other fields are complete.
- **Cancellations**: Invoices starting with "C" represent cancelled orders. These need to be excluded before building customer features.
- **Skewed distributions**: Both `quantity` and `price` are heavily right-skewed with extreme outliers, which told me I would need log transformations before clustering.
- **Dominance of UK**: The vast majority of customers and revenue come from the United Kingdom, with smaller contributions from European countries.

**Missing Values**

![Missing Values](01_eda/output/missing_values.png)

**Transaction Volume and Revenue Over Time**

![Transactions Over Time](01_eda/output/transactions_over_time.png)

**Top Countries**

![Top Countries](01_eda/output/top_countries.png)

**Distributions (Quantity and Price)**

![Distributions](01_eda/output/distributions.png)

**Cancellation Rate Over Time**

![Cancellations](01_eda/output/cancellation_rate.png)

**RFM Preview**

I also computed a preview of the RFM (Recency, Frequency, Monetary) features to understand what the customer-level data would look like. All three are heavily right-skewed, confirming the need for log transformation.

![RFM Preview](01_eda/output/rfm_preview.png)

**RFM Correlation Heatmap**

![Correlation Heatmap](01_eda/output/correlation_heatmap.png)

### 2. Preprocessing

This is where I transformed the raw transaction-level data into a customer-level feature matrix suitable for clustering. The preprocessing involved three steps:

**Step 1: Data Cleaning**

I removed records that would distort the customer features:
- Cancelled orders (invoices starting with "C")
- Rows with missing `customer_id` (can't assign to a customer)
- Rows with `quantity <= 0` or `price <= 0` (invalid transactions)

**Step 2: Feature Engineering**

For each customer, I computed 8 features that capture different aspects of purchasing behavior:

| Feature | Description | Why It Matters |
|---|---|---|
| `recency` | Days since last purchase (relative to latest date in data) | How recently a customer engaged -- recent customers are more likely to return |
| `frequency` | Number of unique invoices (orders) | How often they buy -- frequent buyers are loyal |
| `monetary` | Total revenue (quantity × price) | How much they spend -- high spenders are high value |
| `avg_order_value` | Average revenue per order | Spending intensity per visit |
| `avg_items_per_order` | Average total items per order | Bulk buyers vs. single-item purchasers |
| `unique_products` | Number of distinct products purchased | Breadth of interest in the catalog |
| `avg_unit_price` | Average price per item | Premium vs. budget shoppers |
| `tenure_days` | Days between first and last purchase | Length of relationship |

**Step 3: Visualization**

I plotted the distributions of all 8 features and their correlation heatmap to confirm they capture different dimensions of behavior.

**Feature Distributions**

![Feature Distributions](02_preprocessing/output/feature_distributions.png)

**Feature Correlation Heatmap**

![Correlation Heatmap](02_preprocessing/output/correlation_heatmap.png)

The customer-level feature matrix was saved to S3 for downstream use.

### 3. K-Means Clustering

K-Means is a distance-based algorithm, which means it assigns each customer to the cluster whose center (centroid) is closest. Because it uses distance, I needed to make sure all features were on comparable scales -- otherwise a feature like `monetary` (range: hundreds to thousands) would dominate a feature like `avg_unit_price` (range: single digits).

**Preprocessing for Clustering**

1. **Log transformation** (`log1p`): I applied this to the right-skewed features (`frequency`, `monetary`, `avg_order_value`, `avg_items_per_order`, `unique_products`, `avg_unit_price`) to compress the long tails and make distributions more symmetric.
2. **Z-score standardization**: I then standardized all 8 features to have mean 0 and standard deviation 1, so every feature contributes equally to the distance calculation.

Both transformations were saved as a serialized preprocessing pipeline via joblib, so I can apply the exact same transformations to new data at inference time.

**Selecting the Number of Clusters (K)**

Choosing K is one of the hardest parts of clustering. I used two complementary methods:

- **Elbow method**: I plotted the total within-cluster sum of squares (inertia) for K=2 through K=10. The "elbow" is the point where adding more clusters stops providing meaningful reduction in inertia. The `KneeLocator` algorithm detected the elbow at **K=5**.
- **Silhouette score**: This measures how similar each customer is to their own cluster compared to other clusters (range: -1 to 1, higher is better). The highest silhouette was at K=2, but K=2 is too coarse for actionable segmentation. K=5 provided a good balance.

**Elbow and Silhouette Analysis**

![Elbow Silhouette](03_clustering/output/elbow_silhouette.png)

**Final Model (K=5)**

I fit the final K-Means model with K=5 and achieved a silhouette score of **0.206**. The silhouette plot below shows how well-separated each cluster is:

![Silhouette Analysis](03_clustering/output/silhouette_analysis.png)

**Cluster Centroids (Standardized Heatmap)**

This heatmap shows how each cluster's average feature values compare to the overall population. Red means above average, blue means below average:

![Centroid Heatmap](03_clustering/output/centroid_heatmap.png)

**Feature Distributions by Cluster**

![Distributions by Cluster](03_clustering/output/distributions_by_cluster.png)

**Cluster Sizes**

![Cluster Sizes](03_clustering/output/cluster_sizes.png)

### 4. Cluster Profiling

After assigning each customer to a cluster, I needed to understand *what makes each cluster different*. This is where the analysis becomes actionable. I joined the cluster assignments back to the raw transaction data to analyze product preferences, geographic composition, and purchase timing.

**Radar Chart**

The radar chart provides a quick visual summary of each cluster's personality across all 8 features (normalized to 0--1 for comparability):

![Radar Chart](04_profiling/output/radar_chart.png)

**Box Plots**

Box plots show the full distribution (median, quartiles, outliers) of each feature within each cluster:

![Box Plots](04_profiling/output/box_plots.png)

**Top Products by Cluster**

Each cluster gravitates toward different products, which is valuable for targeted marketing:

![Top Products](04_profiling/output/top_products.png)

**Country Composition**

![Country Composition](04_profiling/output/country_composition.png)

**Purchase Timing**

I analyzed when each cluster tends to purchase -- by day of week and hour of day. This can inform when to send promotional emails or schedule flash sales:

![Purchase Timing](04_profiling/output/purchase_timing.png)

**Revenue Over Time by Cluster**

![Revenue Over Time](04_profiling/output/revenue_over_time.png)

### 5. Model Comparison

To validate that K-Means was the right algorithm for this data, I compared it against two alternatives:

- **Agglomerative clustering (Ward linkage)** at the same K=5 for a fair comparison
- **DBSCAN** across multiple `eps` values, selecting the best by silhouette score

| Model | K | Silhouette | Noise % | Max Cluster % | Min Cluster % |
|---|---|---|---|---|---|
| K-Means | 5 | 0.2060 | 0.0% | 28.0% | 1.0% |
| Agglomerative | 5 | 0.1767 | 0.0% | 41.9% | 1.0% |
| DBSCAN | 2 | 0.5716 | 1.2% | 98.6% | 0.3% |

DBSCAN achieved the highest silhouette score, but only because it put nearly all customers into a single cluster and flagged the rest as noise -- this is not useful for segmentation. Agglomerative clustering produced even more imbalanced clusters than K-Means. **K-Means provided the best balance of cluster quality and actionable segmentation.**

**Silhouette and Cluster Count Comparison**

![Metrics Comparison](05_comparison/output/metrics_comparison.png)

**PCA 2D Projection**

I projected the 8-dimensional feature space down to 2 dimensions using PCA to visually compare how each algorithm partitions the data:

![PCA Comparison](05_comparison/output/pca_comparison.png)

**Dendrogram (Hierarchical Structure)**

![Dendrogram](05_comparison/output/dendrogram.png)

**Cluster Size Comparison**

![Cluster Sizes Comparison](05_comparison/output/cluster_sizes_comparison.png)

---

## Results

**K-Means with K=5 is the champion model.** The five customer segments I identified are:

| Cluster | N Customers | Description |
|---|---|---|
| **0** | 1,078 (25.0%) | **Regular buyers** -- moderate recency (~46 days), ~4 orders, ~$933 total spend, broad product interest (~59 unique products), long tenure (~214 days) |
| **1** | 1,019 (23.6%) | **Lapsed low-value** -- high recency (~175 days), ~1 order, ~$201 total spend, narrow product interest (~15 products), very short tenure (~19 days) |
| **2** | 963 (22.3%) | **VIP / Champions** -- very recent (~30 days), ~12 orders, ~$6,734 total spend, widest product interest (~153 products), longest tenure (~289 days) |
| **3** | 44 (1.0%) | **High-ticket niche** -- high recency (~184 days), ~2 orders, ~$1,593 total spend but very high avg unit price (~$644), only ~2 unique products. These appear to be wholesale or specialty buyers |
| **4** | 1,208 (28.0%) | **Low-frequency mid-value** -- moderate recency (~107 days), ~2 orders, ~$883 total spend, high avg items per order (~416), moderate tenure (~40 days). Bulk purchasers who buy infrequently |

### Recommended Marketing Strategies

| Segment | Strategy |
|---|---|
| **Regular buyers (0)** | Loyalty programs, cross-sell based on browsing patterns, personalized recommendations |
| **Lapsed low-value (1)** | Win-back campaigns, discount offers to re-engage, "we miss you" emails |
| **VIP / Champions (2)** | Exclusive early access, premium support, referral incentives -- protect and nurture |
| **High-ticket niche (3)** | Dedicated account management, bulk pricing, custom catalogs |
| **Low-frequency mid-value (4)** | Frequency-building campaigns, subscription offers, reminders at predicted reorder intervals |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Log transformation before clustering | RFM features are heavily right-skewed; without log transform, a few high-spend customers would dominate the distance calculations and distort clusters |
| Z-score standardization | K-Means uses Euclidean distance, so features must be on comparable scales -- otherwise monetary ($0--$100K) would overpower frequency (1--50) |
| Elbow method + silhouette analysis | Using two methods provides more confidence in K selection than either alone |
| K=5 over K=2 | K=2 had the highest silhouette but is too coarse for actionable segmentation. K=5 balances statistical quality with business interpretability |
| Excluded cancellations for feature engineering | Cancelled orders would inflate frequency and distort monetary calculations |
| Excluded missing customer_id | Can't build customer-level features without knowing who the customer is |
| Serialized preprocessing pipeline | Guarantees identical transformations at inference time when assigning new customers to segments |
| Compared three algorithms | Validates that K-Means was the right choice for this data, not just the default |
| Profiled with raw transaction data | Joining clusters back to transactions reveals product preferences and timing patterns that cluster centroids alone cannot show |

---

## Getting Started

### Prerequisites

- Python 3.10+
- AWS credentials configured (for S3 data access)
- AWS SageMaker notebook instance (recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute the notebooks in order:

```
00_data_collection/notebook.ipynb     # Download dataset and save to S3
01_eda/notebook.ipynb                 # Explore the transaction data
02_preprocessing/notebook.ipynb       # Clean data and engineer customer features
03_clustering/notebook.ipynb          # Fit K-Means and select optimal K
04_profiling/notebook.ipynb           # Profile and interpret the segments
05_comparison/notebook.ipynb          # Compare K-Means vs alternatives
```

Each notebook is self-contained with its own helper classes, constants, and visualizations. All data flows through S3 -- only models, plots, and small summary tables are saved locally to each notebook's `output/` directory.
