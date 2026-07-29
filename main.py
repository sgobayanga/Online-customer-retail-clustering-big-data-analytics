"""Run the complete Online Retail analysis from PyCharm or a terminal."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, create_output_folders, resolve_data_path
from retail_pipeline import (
    clean_transactions,
    create_cluster_selection_charts,
    create_customer_frequency_chart,
    create_sales_summaries,
    create_sales_visualisations,
    create_segment_charts,
    create_text_summary,
    engineer_customer_features,
    evaluate_cluster_numbers,
    evaluate_final_model,
    export_results,
    fit_final_segmentation,
    inspect_dataset,
    load_dataset,
    prepare_clustering_data,
    print_table,
    save_model_objects,
)


def configure_logging() -> None:
    """Write progress to both the PyCharm Run window and a log file."""
    log_file = OUTPUT_DIR / "analysis.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def parse_arguments() -> argparse.Namespace:
    """Allow an optional Excel path when running from a terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyse the UCI Online Retail dataset and create "
            "customer segments."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Optional full path to Online Retail.xlsx. "
            "The default location is data/Online Retail.xlsx."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the analysis in a clear sequence."""
    create_output_folders()
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    try:
        data_path = resolve_data_path(args.data)

        print("\nONLINE RETAIL CUSTOMER SEGMENTATION")
        print("=" * 44)
        print("Dataset:", data_path)
        print("Results folder:", OUTPUT_DIR)

        print("\n[1/10] Loading the Excel dataset...")
        df = load_dataset(data_path)
        print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
        print(df.head().to_string(index=False))

        print("\n[2/10] Inspecting missing values, duplicates and cancellations...")
        inspection = inspect_dataset(df)
        print_table("Missing-value summary", inspection["missing_summary"], rows=8)
        print_table("Data-quality summary", inspection["quality_summary"], rows=10)
        print(
            "\nUnique cancelled invoices:",
            f"{inspection['cancelled_invoice_count']:,}",
        )
        print("\nQuantity IQR assessment:", inspection["quantity_outliers"])
        print("Unit price IQR assessment:", inspection["price_outliers"])

        print("\n[3/10] Cleaning completed sales...")
        _, sales_df, customer_sales_df = clean_transactions(df)
        print("Valid completed-sale lines:", f"{len(sales_df):,}")
        print(
            "Valid lines with CustomerID:",
            f"{len(customer_sales_df):,}",
        )
        print(
            "Completed-sales revenue:",
            f"£{sales_df['Revenue'].sum():,.2f}",
        )

        print("\n[4/10] Creating sales summaries and graphs...")
        summaries = create_sales_summaries(sales_df)
        create_sales_visualisations(sales_df, summaries)
        print_table("Monthly sales", summaries["monthly_sales"], rows=15)
        print_table("Top products", summaries["top_products"], rows=10)
        print_table("Top countries", summaries["country_sales"], rows=10)

        print("\n[5/10] Engineering customer-level features...")
        customer_features = engineer_customer_features(customer_sales_df)
        create_customer_frequency_chart(customer_features)
        print_table("Customer feature preview", customer_features, rows=10)

        print("\n[6/10] Transforming, scaling and applying PCA...")
        clustering_data = prepare_clustering_data(customer_features)
        print(
            "PCA variance retained:",
            f"{clustering_data['pca'].explained_variance_ratio_.sum() * 100:.2f}%",
        )
        print_table(
            "PCA loadings",
            clustering_data["pca_loadings"].reset_index(
                names="Feature"
            ),
            rows=10,
        )

        print("\n[7/10] Comparing cluster numbers from 2 to 8...")
        cluster_evaluation, fitted_models = evaluate_cluster_numbers(
            clustering_data["x_pca"]
        )
        create_cluster_selection_charts(cluster_evaluation)
        print_table(
            "Cluster-number evaluation",
            cluster_evaluation.round(4),
            rows=10,
        )

        print("\n[8/10] Fitting and profiling the four customer segments...")
        segmentation = fit_final_segmentation(
            customer_features,
            clustering_data,
            fitted_models,
        )
        create_segment_charts(segmentation)
        print_table(
            "Customer segment profiles",
            segmentation["cluster_profiles"].round(2),
            rows=10,
        )

        print("\n[9/10] Evaluating the final segmentation model...")
        evaluation_results = evaluate_final_model(
            clustering_data["x_pca"],
            segmentation["final_labels"],
            clustering_data["pca"],
        )
        print_table(
            "Final model evaluation",
            evaluation_results.round(4),
            rows=10,
        )

        print("\n[10/10] Exporting results and model files...")
        export_results(
            inspection=inspection,
            summaries=summaries,
            customer_features=customer_features,
            cluster_evaluation=cluster_evaluation,
            segmentation=segmentation,
            evaluation_results=evaluation_results,
        )
        save_model_objects(clustering_data, segmentation)

        summary_text = create_text_summary(
            df=df,
            sales_df=sales_df,
            customer_features=customer_features,
            summaries=summaries,
            segmentation=segmentation,
            evaluation_results=evaluation_results,
        )

        print("\n" + summary_text)
        print("\nAnalysis completed successfully.")
        print("Open this folder to see all results:")
        print(OUTPUT_DIR.resolve())

    except FileNotFoundError as error:
        print("\nDATASET ERROR")
        print("=" * 30)
        print(error)
        raise SystemExit(1) from error

    except PermissionError as error:
        print("\nPERMISSION ERROR")
        print("=" * 30)
        print(
            "Close any output Excel or CSV file that is currently open, "
            "then run the program again."
        )
        print(error)
        raise SystemExit(1) from error

    except Exception as error:
        logger.exception("The analysis stopped because of an error.")
        print("\nANALYSIS FAILED")
        print("=" * 30)
        print(f"{type(error).__name__}: {error}")
        print(
            "\nThe full error has been saved in outputs/analysis.log. "
            "The traceback is also shown below."
        )
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
