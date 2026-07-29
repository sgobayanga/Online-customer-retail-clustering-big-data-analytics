"""
Functions for Online Retail customer segmentation and purchase behaviour analysis.

The workflow reproduces the analysis from the original Jupyter Notebook but
uses ordinary Python functions that can run directly in PyCharm.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from config import (
    CHART_DIR,
    CHOSEN_K,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    SHOW_PLOTS,
)

LOGGER = logging.getLogger(__name__)


def print_table(title: str, table: pd.DataFrame | pd.Series, rows: int = 10) -> None:
    """Print a readable preview in PyCharm's Run window."""
    print(f"\n{title}")
    print("-" * len(title))
    if isinstance(table, pd.Series):
        print(table.head(rows).to_string())
    else:
        print(table.head(rows).to_string(index=False))


def save_figure(file_name: str) -> None:
    """Save the current Matplotlib figure and optionally display it."""
    output_path = CHART_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    LOGGER.info("Saved chart: %s", output_path)


def _excel_column_index(cell_reference: str) -> int:
    """Convert an Excel reference such as E25 into a zero-based column index."""
    letters = re.match(r"([A-Za-z]+)", cell_reference)
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")

    index = 0
    for character in letters.group(1).upper():
        index = index * 26 + (ord(character) - 64)
    return index - 1


def _read_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    """Read the workbook's shared-string table used by XLSX files."""
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in workbook.namelist():
        return []

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = etree.fromstring(workbook.read(shared_strings_path))

    return [
        "".join(text_node.text or "" for text_node in item.iter(namespace + "t"))
        for item in root.findall(namespace + "si")
    ]


def _first_worksheet_path(workbook: zipfile.ZipFile) -> str:
    """Return the first worksheet XML file inside an XLSX workbook."""
    worksheet_paths = sorted(
        name
        for name in workbook.namelist()
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
    )
    if not worksheet_paths:
        raise ValueError("The Excel workbook does not contain a worksheet.")
    return worksheet_paths[0]


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the eight required columns using direct XLSX XML streaming.

    This method is intentionally used instead of pandas.read_excel because the
    UCI workbook contains more than 500,000 rows. Direct streaming is much
    faster and uses less memory in Jupyter and PyCharm.
    """
    LOGGER.info("Loading dataset from %s", data_path)
    LOGGER.info("Using memory-efficient XLSX streaming reader")

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    raw_columns: dict[str, list[Any]] = {column: [] for column in REQUIRED_COLUMNS}

    with zipfile.ZipFile(data_path) as workbook:
        shared_strings = _read_shared_strings(workbook)
        worksheet_path = _first_worksheet_path(workbook)

        with workbook.open(worksheet_path) as worksheet:
            row_iterator = etree.iterparse(
                worksheet,
                events=("end",),
                tag=namespace + "row",
                huge_tree=True,
            )

            header_by_index: dict[int, str] = {}
            loaded_rows = 0

            for _, row_element in row_iterator:
                row_number = int(row_element.get("r", "0"))
                row_values: dict[int, Any] = {}

                for cell in row_element.findall(namespace + "c"):
                    reference = cell.get("r")
                    if reference is None:
                        continue

                    column_index = _excel_column_index(reference)
                    cell_type = cell.get("t")
                    value_node = cell.find(namespace + "v")

                    if cell_type == "inlineStr":
                        inline_node = cell.find(namespace + "is")
                        value = (
                            "".join(
                                node.text or ""
                                for node in inline_node.iter(namespace + "t")
                            )
                            if inline_node is not None
                            else None
                        )
                    elif value_node is None:
                        value = None
                    else:
                        raw_value = value_node.text
                        if cell_type == "s":
                            value = shared_strings[int(raw_value)]
                        else:
                            value = raw_value

                    row_values[column_index] = value

                if row_number == 1:
                    header_by_index = {
                        index: str(value).strip()
                        for index, value in row_values.items()
                        if value is not None
                    }

                    missing_columns = sorted(
                        set(REQUIRED_COLUMNS) - set(header_by_index.values())
                    )
                    if missing_columns:
                        raise ValueError(
                            "The Excel file does not contain all expected columns. "
                            f"Missing columns: {missing_columns}"
                        )
                elif header_by_index:
                    for column_index, column_name in header_by_index.items():
                        if column_name in raw_columns:
                            raw_columns[column_name].append(
                                row_values.get(column_index)
                            )

                    loaded_rows += 1
                    if loaded_rows % 100_000 == 0:
                        LOGGER.info("Loaded %s transaction rows", f"{loaded_rows:,}")

                row_element.clear()
                while row_element.getprevious() is not None:
                    del row_element.getparent()[0]

    df = pd.DataFrame(raw_columns)

    # Convert numerical and date columns after streaming has completed.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")

    invoice_serial = pd.to_numeric(df["InvoiceDate"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(
        invoice_serial,
        unit="D",
        origin="1899-12-30",
        errors="coerce",
    ).dt.round("s")

    # Keep identifiers as text so values such as invoice and stock codes do not
    # become decimals or lose letters.
    df["InvoiceNo"] = df["InvoiceNo"].where(df["InvoiceNo"].notna(), None)
    df["StockCode"] = df["StockCode"].where(df["StockCode"].notna(), None)

    LOGGER.info("Loaded %s rows and %s columns", f"{len(df):,}", len(df.columns))
    return df


def iqr_outlier_summary(series: pd.Series) -> dict[str, float | int]:
    """Return IQR limits and the number of values outside those limits."""
    clean_series = series.dropna()
    q1 = float(clean_series.quantile(0.25))
    q3 = float(clean_series.quantile(0.75))
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    outlier_mask = (clean_series < lower_limit) | (clean_series > upper_limit)

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LowerLimit": lower_limit,
        "UpperLimit": upper_limit,
        "OutlierCount": int(outlier_mask.sum()),
    }


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect structure, missing data, duplicates, cancellations, and unusual values."""
    LOGGER.info("Inspecting dataset quality")

    missing_summary = (
        pd.DataFrame(
            {
                "Variable": df.columns,
                "MissingValues": df.isna().sum().values,
                "MissingPercentage": df.isna().mean().mul(100).values,
            }
        )
        .sort_values("MissingValues", ascending=False)
        .reset_index(drop=True)
    )

    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = duplicate_count / len(df) * 100

    invoice_text = df["InvoiceNo"].astype(str).str.strip()
    cancelled_mask = invoice_text.str.upper().str.startswith("C")
    cancelled_rows = df.loc[cancelled_mask].copy()

    quality_summary = pd.DataFrame(
        {
            "Issue": [
                "Missing CustomerID",
                "Missing Description",
                "Exact duplicate rows",
                "Cancelled transaction rows",
                "Negative quantity rows",
                "Zero quantity rows",
                "Negative UnitPrice rows",
                "Zero UnitPrice rows",
            ],
            "Count": [
                int(df["CustomerID"].isna().sum()),
                int(df["Description"].isna().sum()),
                duplicate_count,
                int(cancelled_mask.sum()),
                int((df["Quantity"] < 0).sum()),
                int((df["Quantity"] == 0).sum()),
                int((df["UnitPrice"] < 0).sum()),
                int((df["UnitPrice"] == 0).sum()),
            ],
        }
    )

    quantity_statistics = df["Quantity"].describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
    price_statistics = df["UnitPrice"].describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    quantity_outliers = iqr_outlier_summary(df["Quantity"])
    price_outliers = iqr_outlier_summary(df["UnitPrice"])

    missing_summary.to_csv(OUTPUT_DIR / "missing_value_summary.csv", index=False)
    quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    cancelled_rows.to_csv(OUTPUT_DIR / "cancelled_transactions.csv", index=False)
    quantity_statistics.to_csv(OUTPUT_DIR / "quantity_statistics.csv")
    price_statistics.to_csv(OUTPUT_DIR / "unit_price_statistics.csv")

    LOGGER.info(
        "Found %s duplicates and %s unique cancelled invoices",
        f"{duplicate_count:,}",
        f"{cancelled_rows['InvoiceNo'].nunique():,}",
    )

    return {
        "missing_summary": missing_summary,
        "quality_summary": quality_summary,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "cancelled_rows": cancelled_rows,
        "cancelled_invoice_count": int(cancelled_rows["InvoiceNo"].nunique()),
        "quantity_statistics": quantity_statistics,
        "price_statistics": price_statistics,
        "quantity_outliers": quantity_outliers,
        "price_outliers": price_outliers,
    }


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Remove exact duplicates and keep completed positive-value sales.

    Records without CustomerID remain in sales_df for general sales analysis.
    They are removed only from customer_sales_df because they cannot be linked
    to an individual customer.
    """
    LOGGER.info("Cleaning transactions")

    clean_df = df.drop_duplicates().copy()

    clean_df["InvoiceNo"] = clean_df["InvoiceNo"].astype(str).str.strip()
    clean_df["StockCode"] = clean_df["StockCode"].astype(str).str.strip()
    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], errors="coerce")
    clean_df["IsCancelled"] = (
        clean_df["InvoiceNo"].str.upper().str.startswith("C")
    )

    sales_df = clean_df.loc[
        (~clean_df["IsCancelled"])
        & (clean_df["Quantity"] > 0)
        & (clean_df["UnitPrice"] > 0)
        & (clean_df["InvoiceDate"].notna())
    ].copy()

    sales_df["Revenue"] = sales_df["Quantity"] * sales_df["UnitPrice"]

    customer_sales_df = sales_df.dropna(subset=["CustomerID"]).copy()
    customer_sales_df["CustomerID"] = (
        customer_sales_df["CustomerID"].astype(int).astype(str)
    )

    LOGGER.info("Rows after duplicate removal: %s", f"{len(clean_df):,}")
    LOGGER.info("Valid completed-sale rows: %s", f"{len(sales_df):,}")
    LOGGER.info(
        "Valid rows with CustomerID: %s", f"{len(customer_sales_df):,}"
    )

    return clean_df, sales_df, customer_sales_df


def create_sales_summaries(sales_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate transactions by month, product, country, and invoice."""
    LOGGER.info("Creating sales summaries")

    sales_df = sales_df.copy()
    sales_df["InvoiceMonth"] = (
        sales_df["InvoiceDate"].dt.to_period("M").astype(str)
    )

    monthly_sales = (
        sales_df.groupby("InvoiceMonth", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("InvoiceMonth")
    )

    # Aggregate by StockCode so the same product is not split when its
    # description was entered differently on different transaction lines.
    product_descriptions = (
        sales_df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(
            lambda values: (
                values.mode().iloc[0]
                if not values.mode().empty
                else values.iloc[0]
            )
        )
        .rename("Description")
        .reset_index()
    )

    top_products = (
        sales_df.groupby("StockCode", as_index=False)
        .agg(
            TotalQuantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .merge(product_descriptions, on="StockCode", how="left")
        [
            [
                "StockCode",
                "Description",
                "TotalQuantity",
                "Revenue",
                "Orders",
            ]
        ]
        .sort_values("TotalQuantity", ascending=False)
    )

    country_sales = (
        sales_df.groupby("Country", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Orders=("InvoiceNo", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )

    order_values = (
        sales_df.groupby("InvoiceNo", as_index=False)
        .agg(
            OrderValue=("Revenue", "sum"),
            OrderQuantity=("Quantity", "sum"),
            InvoiceDate=("InvoiceDate", "max"),
            Country=("Country", "first"),
        )
        .sort_values("OrderValue", ascending=False)
    )

    monthly_sales.to_csv(OUTPUT_DIR / "monthly_sales.csv", index=False)
    top_products.to_csv(OUTPUT_DIR / "product_performance.csv", index=False)
    country_sales.to_csv(OUTPUT_DIR / "country_performance.csv", index=False)
    order_values.to_csv(OUTPUT_DIR / "order_values.csv", index=False)

    return {
        "monthly_sales": monthly_sales,
        "top_products": top_products,
        "country_sales": country_sales,
        "order_values": order_values,
    }


def create_sales_visualisations(
    sales_df: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    """Create the six sales and purchasing-behaviour graphs required."""
    LOGGER.info("Creating sales and purchase-behaviour charts")

    monthly_sales = summaries["monthly_sales"]
    top_products = summaries["top_products"]
    country_sales = summaries["country_sales"]
    order_values = summaries["order_values"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly_sales["InvoiceMonth"],
        monthly_sales["Revenue"],
        marker="o",
    )
    plt.title("Monthly Sales Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.25)
    save_figure("01_monthly_sales_trend.png")

    top_10_products = (
        top_products.head(10).sort_values("TotalQuantity").copy()
    )
    top_10_products["ProductLabel"] = (
        top_10_products["Description"]
        .fillna(top_10_products["StockCode"])
        .astype(str)
        .str.slice(0, 40)
    )

    plt.figure(figsize=(11, 7))
    plt.barh(
        top_10_products["ProductLabel"],
        top_10_products["TotalQuantity"],
    )
    plt.title("Top 10 Products by Quantity Sold")
    plt.xlabel("Total Quantity")
    plt.ylabel("Product")
    save_figure("02_top_products_by_quantity.png")

    top_10_countries = country_sales.head(10).sort_values("Revenue")
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_countries["Country"], top_10_countries["Revenue"])
    plt.title("Top 10 Countries by Sales Revenue")
    plt.xlabel("Revenue (£)")
    plt.ylabel("Country")
    save_figure("03_revenue_by_country.png")

    order_value_cap = order_values["OrderValue"].quantile(0.99)
    order_values_for_plot = order_values.loc[
        order_values["OrderValue"] <= order_value_cap,
        "OrderValue",
    ]
    plt.figure(figsize=(10, 6))
    plt.hist(order_values_for_plot, bins=50)
    plt.title("Distribution of Order Values (up to 99th Percentile)")
    plt.xlabel("Order Value (£)")
    plt.ylabel("Number of Orders")
    save_figure("04_order_value_distribution.png")

    # The frequency and product quantity charts are created after customer
    # features have been engineered.
    quantity_cap = sales_df["Quantity"].quantile(0.99)
    plt.figure(figsize=(10, 6))
    plt.hist(
        sales_df.loc[sales_df["Quantity"] <= quantity_cap, "Quantity"],
        bins=40,
    )
    plt.title("Product Quantity per Transaction Line (up to 99th Percentile)")
    plt.xlabel("Quantity")
    plt.ylabel("Number of Transaction Lines")
    save_figure("06_product_quantity_patterns.png")


def engineer_customer_features(
    customer_sales_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one analytical record per customer."""
    LOGGER.info("Engineering customer-level features")

    reference_date = (
        customer_sales_df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    customer_features = (
        customer_sales_df.groupby("CustomerID")
        .agg(
            LastPurchaseDate=("InvoiceDate", "max"),
            FirstPurchaseDate=("InvoiceDate", "min"),
            FrequencyOrders=("InvoiceNo", "nunique"),
            MonetaryRevenue=("Revenue", "sum"),
            ProductDiversity=("StockCode", "nunique"),
            TotalQuantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    customer_features["RecencyDays"] = (
        reference_date - customer_features["LastPurchaseDate"]
    ).dt.days

    customer_features["CustomerTenureDays"] = (
        customer_features["LastPurchaseDate"]
        - customer_features["FirstPurchaseDate"]
    ).dt.days

    customer_features["AverageOrderValue"] = (
        customer_features["MonetaryRevenue"]
        / customer_features["FrequencyOrders"]
    )

    customer_features = customer_features[
        [
            "CustomerID",
            "RecencyDays",
            "FrequencyOrders",
            "MonetaryRevenue",
            "ProductDiversity",
            "TotalQuantity",
            "AverageOrderValue",
            "CustomerTenureDays",
            "FirstPurchaseDate",
            "LastPurchaseDate",
        ]
    ]

    customer_features.to_csv(
        OUTPUT_DIR / "customer_features.csv",
        index=False,
    )

    LOGGER.info(
        "Created features for %s customers", f"{len(customer_features):,}"
    )
    return customer_features


def create_customer_frequency_chart(customer_features: pd.DataFrame) -> None:
    """Plot customer order frequency up to the 99th percentile."""
    frequency_cap = customer_features["FrequencyOrders"].quantile(0.99)

    plt.figure(figsize=(10, 6))
    plt.hist(
        customer_features.loc[
            customer_features["FrequencyOrders"] <= frequency_cap,
            "FrequencyOrders",
        ],
        bins=30,
    )
    plt.title("Customer Purchase Frequency (up to 99th Percentile)")
    plt.xlabel("Number of Unique Orders")
    plt.ylabel("Number of Customers")
    save_figure("05_customer_purchase_frequency.png")


def prepare_clustering_data(
    customer_features: pd.DataFrame,
) -> dict[str, Any]:
    """
    Cap outliers, apply log transformation and scaling, then perform PCA.
    """
    LOGGER.info("Preparing clustering features")

    cluster_feature_names = [
        "RecencyDays",
        "FrequencyOrders",
        "MonetaryRevenue",
        "ProductDiversity",
    ]

    x_original = customer_features[cluster_feature_names].copy()

    # Capping reduces the influence of extreme customers without deleting them.
    upper_caps = x_original.quantile(0.99)
    x_capped = x_original.clip(upper=upper_caps, axis=1)

    # log1p reduces right skew and works safely when a value is zero.
    x_logged = np.log1p(x_capped)

    # StandardScaler prevents MonetaryRevenue from dominating distance.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_logged)

    # Two PCA components support a clear two-dimensional cluster visualisation.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(
        x_pca,
        columns=["PC1", "PC2"],
        index=customer_features.index,
    )

    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=cluster_feature_names,
        columns=["PC1", "PC2"],
    )
    pca_loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    upper_caps.rename("Cap").to_csv(OUTPUT_DIR / "clustering_feature_caps.csv")

    return {
        "feature_names": cluster_feature_names,
        "x_original": x_original,
        "upper_caps": upper_caps,
        "x_logged": x_logged,
        "x_scaled": x_scaled,
        "scaler": scaler,
        "pca": pca,
        "x_pca": x_pca,
        "pca_df": pca_df,
        "pca_loadings": pca_loadings,
    }


def evaluate_cluster_numbers(x_pca: np.ndarray) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    """Fit K-Means for k=2 through k=8 and calculate cluster-quality metrics."""
    LOGGER.info("Evaluating cluster counts from 2 to 8")

    evaluation_rows: list[dict[str, float | int]] = []
    fitted_models: dict[int, dict[str, Any]] = {}

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        )
        labels = model.fit_predict(x_pca)

        evaluation_rows.append(
            {
                "NumberOfClusters": k,
                "Inertia": float(model.inertia_),
                "SilhouetteScore": float(
                    silhouette_score(x_pca, labels)
                ),
                "CalinskiHarabaszScore": float(
                    calinski_harabasz_score(x_pca, labels)
                ),
                "DaviesBouldinIndex": float(
                    davies_bouldin_score(x_pca, labels)
                ),
            }
        )

        fitted_models[k] = {"model": model, "labels": labels}

    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_df.to_csv(
        OUTPUT_DIR / "cluster_number_evaluation.csv",
        index=False,
    )

    return evaluation_df, fitted_models


def create_cluster_selection_charts(evaluation_df: pd.DataFrame) -> None:
    """Create elbow and silhouette charts used to compare k values."""
    plt.figure(figsize=(9, 5))
    plt.plot(
        evaluation_df["NumberOfClusters"],
        evaluation_df["Inertia"],
        marker="o",
    )
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.xticks(evaluation_df["NumberOfClusters"])
    save_figure("07_elbow_method.png")

    plt.figure(figsize=(9, 5))
    plt.plot(
        evaluation_df["NumberOfClusters"],
        evaluation_df["SilhouetteScore"],
        marker="o",
    )
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(evaluation_df["NumberOfClusters"])
    save_figure("08_silhouette_by_k.png")


def assign_segment_names(
    segmented_customers: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Convert arbitrary cluster numbers into business-friendly segment names."""
    profile_lookup = cluster_profiles.set_index("Cluster")

    champion_score = (
        -profile_lookup["MeanRecency"]
        + 20 * profile_lookup["MeanFrequency"]
        + profile_lookup["MeanMonetaryValue"] / 100
        + profile_lookup["MeanProductDiversity"]
    )
    champion_cluster = int(champion_score.idxmax())

    remaining_after_champion = profile_lookup.drop(index=champion_cluster)
    at_risk_cluster = int(
        remaining_after_champion["MeanRecency"].idxmax()
    )

    remaining_clusters = [
        cluster
        for cluster in profile_lookup.index
        if cluster not in [champion_cluster, at_risk_cluster]
    ]
    regular_cluster = int(
        profile_lookup.loc[
            remaining_clusters,
            "MeanMonetaryValue",
        ].idxmax()
    )
    occasional_cluster = int(
        [
            cluster
            for cluster in remaining_clusters
            if cluster != regular_cluster
        ][0]
    )

    segment_name_map = {
        champion_cluster: "Champions / Loyal High-Value",
        regular_cluster: "Regular / Promising",
        occasional_cluster: "Recent Occasional / Low-Value",
        at_risk_cluster: "At-Risk / Lapsed",
    }

    segmented_customers = segmented_customers.copy()
    cluster_profiles = cluster_profiles.copy()

    segmented_customers["SegmentName"] = (
        segmented_customers["Cluster"].map(segment_name_map)
    )
    cluster_profiles["SegmentName"] = (
        cluster_profiles["Cluster"].map(segment_name_map)
    )

    return segmented_customers, cluster_profiles, segment_name_map


def fit_final_segmentation(
    customer_features: pd.DataFrame,
    clustering_data: dict[str, Any],
    fitted_models: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Use the chosen four-cluster model and profile the resulting segments."""
    LOGGER.info("Fitting final %s-cluster solution", CHOSEN_K)

    final_model = fitted_models[CHOSEN_K]["model"]
    final_labels = fitted_models[CHOSEN_K]["labels"]

    segmented_customers = customer_features.copy()
    segmented_customers["Cluster"] = final_labels

    pca_df = clustering_data["pca_df"].copy()
    pca_df["Cluster"] = final_labels

    cluster_profiles = (
        segmented_customers.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            MeanRecency=("RecencyDays", "mean"),
            MedianRecency=("RecencyDays", "median"),
            MeanFrequency=("FrequencyOrders", "mean"),
            MedianFrequency=("FrequencyOrders", "median"),
            MeanMonetaryValue=("MonetaryRevenue", "mean"),
            MedianMonetaryValue=("MonetaryRevenue", "median"),
            MeanProductDiversity=("ProductDiversity", "mean"),
            MedianProductDiversity=("ProductDiversity", "median"),
            TotalRevenue=("MonetaryRevenue", "sum"),
        )
        .reset_index()
    )

    cluster_profiles["CustomerSharePct"] = (
        cluster_profiles["Customers"]
        / cluster_profiles["Customers"].sum()
        * 100
    )
    cluster_profiles["RevenueSharePct"] = (
        cluster_profiles["TotalRevenue"]
        / cluster_profiles["TotalRevenue"].sum()
        * 100
    )

    (
        segmented_customers,
        cluster_profiles,
        segment_name_map,
    ) = assign_segment_names(segmented_customers, cluster_profiles)

    marketing_action_map = {
        "Champions / Loyal High-Value": (
            "VIP rewards, early access, referrals, premium bundles "
            "and personalised recommendations"
        ),
        "Regular / Promising": (
            "Loyalty points, cross-selling, free-shipping thresholds "
            "and frequency-building offers"
        ),
        "Recent Occasional / Low-Value": (
            "Welcome campaigns, second-purchase discounts "
            "and product education"
        ),
        "At-Risk / Lapsed": (
            "Reactivation offers, feedback surveys and "
            "time-limited win-back campaigns"
        ),
    }

    segmented_customers["RecommendedMarketingAction"] = (
        segmented_customers["SegmentName"].map(marketing_action_map)
    )

    profile_columns = [
        "Cluster",
        "SegmentName",
        "Customers",
        "CustomerSharePct",
        "MeanRecency",
        "MedianRecency",
        "MeanFrequency",
        "MedianFrequency",
        "MeanMonetaryValue",
        "MedianMonetaryValue",
        "MeanProductDiversity",
        "MedianProductDiversity",
        "RevenueSharePct",
        "TotalRevenue",
    ]
    cluster_profiles = cluster_profiles[profile_columns]

    return {
        "final_model": final_model,
        "final_labels": final_labels,
        "segmented_customers": segmented_customers,
        "cluster_profiles": cluster_profiles,
        "pca_df": pca_df,
        "segment_name_map": segment_name_map,
        "marketing_action_map": marketing_action_map,
    }


def create_segment_charts(segmentation: dict[str, Any]) -> None:
    """Create PCA and segment share visualisations."""
    pca_df = segmentation["pca_df"]
    segment_name_map = segmentation["segment_name_map"]
    cluster_profiles = segmentation["cluster_profiles"].copy()

    plt.figure(figsize=(11, 7))
    for cluster_number in sorted(pca_df["Cluster"].unique()):
        cluster_points = pca_df.loc[
            pca_df["Cluster"] == cluster_number
        ]
        plt.scatter(
            cluster_points["PC1"],
            cluster_points["PC2"],
            s=18,
            alpha=0.65,
            label=segment_name_map[int(cluster_number)],
        )
    plt.title("Customer Segments in Two PCA Dimensions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    save_figure("09_customer_segments_pca.png")




def evaluate_final_model(
    x_pca: np.ndarray,
    final_labels: np.ndarray,
    pca: PCA,
) -> pd.DataFrame:
    """Calculate the final cluster evaluation metrics."""
    final_silhouette = float(silhouette_score(x_pca, final_labels))
    final_calinski_harabasz = float(
        calinski_harabasz_score(x_pca, final_labels)
    )
    final_davies_bouldin = float(
        davies_bouldin_score(x_pca, final_labels)
    )
    pca_variance = float(pca.explained_variance_ratio_.sum())

    evaluation_results = pd.DataFrame(
        {
            "Metric": [
                "Silhouette Score",
                "Calinski-Harabasz Score",
                "Davies-Bouldin Index",
                "PCA Variance Retained",
            ],
            "Value": [
                final_silhouette,
                final_calinski_harabasz,
                final_davies_bouldin,
                pca_variance,
            ],
            "Interpretation": [
                "Moderate cluster cohesion and separation.",
                "Higher value supports separation between customer groups.",
                "Below 1 indicates reasonably compact groups.",
                "Share of original feature information retained by two components.",
            ],
        }
    )

    evaluation_results.to_csv(
        OUTPUT_DIR / "final_model_evaluation.csv",
        index=False,
    )
    return evaluation_results


def export_results(
    inspection: dict[str, Any],
    summaries: dict[str, pd.DataFrame],
    customer_features: pd.DataFrame,
    cluster_evaluation: pd.DataFrame,
    segmentation: dict[str, Any],
    evaluation_results: pd.DataFrame,
) -> None:
    """Export tables to CSV, Excel, and reusable model files."""
    LOGGER.info("Exporting final results")

    segmented_customers = segmentation["segmented_customers"]
    cluster_profiles = segmentation["cluster_profiles"]

    segmented_customers.to_csv(
        OUTPUT_DIR / "customer_segments.csv",
        index=False,
    )
    cluster_profiles.to_csv(
        OUTPUT_DIR / "cluster_profiles.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUTPUT_DIR / "online_retail_analysis_results.xlsx",
        engine="xlsxwriter",
        engine_kwargs={"options": {"constant_memory": True}},
    ) as writer:
        inspection["missing_summary"].to_excel(
            writer,
            sheet_name="Missing Values",
            index=False,
        )
        inspection["quality_summary"].to_excel(
            writer,
            sheet_name="Data Quality",
            index=False,
        )
        summaries["monthly_sales"].to_excel(
            writer,
            sheet_name="Monthly Sales",
            index=False,
        )
        summaries["top_products"].head(100).to_excel(
            writer,
            sheet_name="Top Products",
            index=False,
        )
        summaries["country_sales"].to_excel(
            writer,
            sheet_name="Country Sales",
            index=False,
        )
        summaries["order_values"].to_excel(
            writer,
            sheet_name="Order Values",
            index=False,
        )
        customer_features.to_excel(
            writer,
            sheet_name="Customer Features",
            index=False,
        )
        segmented_customers.to_excel(
            writer,
            sheet_name="Customer Segments",
            index=False,
        )
        cluster_profiles.to_excel(
            writer,
            sheet_name="Cluster Profiles",
            index=False,
        )
        cluster_evaluation.to_excel(
            writer,
            sheet_name="Cluster Evaluation",
            index=False,
        )
        evaluation_results.to_excel(
            writer,
            sheet_name="Final Evaluation",
            index=False,
        )


def save_model_objects(
    clustering_data: dict[str, Any],
    segmentation: dict[str, Any],
) -> None:
    """Save the fitted preprocessing and clustering objects with joblib."""
    LOGGER.info("Saving model objects")

    model_bundle = {
        "feature_names": clustering_data["feature_names"],
        "upper_caps": clustering_data["upper_caps"],
        "scaler": clustering_data["scaler"],
        "pca": clustering_data["pca"],
        "kmeans": segmentation["final_model"],
        "segment_name_map": segmentation["segment_name_map"],
        "marketing_action_map": segmentation["marketing_action_map"],
    }

    joblib.dump(model_bundle, MODEL_DIR / "customer_segmentation_bundle.joblib")


def create_text_summary(
    df: pd.DataFrame,
    sales_df: pd.DataFrame,
    customer_features: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    segmentation: dict[str, Any],
    evaluation_results: pd.DataFrame,
) -> str:
    """Create a short plain-text summary of the main findings."""
    profiles = segmentation["cluster_profiles"]
    champion_row = profiles.loc[
        profiles["SegmentName"] == "Champions / Loyal High-Value"
    ].iloc[0]

    summary_text = f"""
ONLINE RETAIL ANALYSIS SUMMARY
==============================

Transaction rows: {len(df):,}
Valid completed-sale lines: {len(sales_df):,}
Completed orders: {summaries['order_values']['InvoiceNo'].nunique():,}
Completed-sales revenue: £{sales_df['Revenue'].sum():,.2f}
Customers modelled: {len(customer_features):,}

Key segmentation result:
Champions represent {champion_row['CustomerSharePct']:.2f}% of customers and
generate {champion_row['RevenueSharePct']:.2f}% of identifiable-customer revenue.

Final model metrics:
Silhouette Score: {evaluation_results.loc[evaluation_results['Metric'] == 'Silhouette Score', 'Value'].iloc[0]:.3f}
Davies-Bouldin Index: {evaluation_results.loc[evaluation_results['Metric'] == 'Davies-Bouldin Index', 'Value'].iloc[0]:.3f}
Calinski-Harabasz Score: {evaluation_results.loc[evaluation_results['Metric'] == 'Calinski-Harabasz Score', 'Value'].iloc[0]:,.2f}
PCA Variance Retained: {evaluation_results.loc[evaluation_results['Metric'] == 'PCA Variance Retained', 'Value'].iloc[0] * 100:.2f}%
""".strip()

    (OUTPUT_DIR / "analysis_summary.txt").write_text(
        summary_text,
        encoding="utf-8",
    )
    return summary_text
