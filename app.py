# pip install streamlit joblib scikit-learn pandas numpy matplotlib
# streamlit run app.py
"""
California Housing Price Prediction Demo
Based on a RandomForestRegressor model
"""

import warnings

warnings.filterwarnings("ignore")

import os

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------
# Data / image loaders
# ---------------------------------------------------------------------

# You can adjust these paths to match your local environment.


BASE_DIR = Path(__file__).resolve().parent

IMAGE_CANDIDATES = [
    BASE_DIR / "california.png",
    BASE_DIR / "images" / "end_to_end_project" / "california.png",
]

HOUSING_CSV_CANDIDATES = [
    BASE_DIR / "housing.csv",
    BASE_DIR / "datasets" / "housing" / "housing.csv",
]

# IMAGE_CANDIDATES = [
#     "https://github.com/tjluo-ucas/data-science/blob/main/california.png",
#     "images/end_to_end_project/california.png",
#     "/Users/abc/handson-ml2/images/end_to_end_project/california.png",
# ]

# HOUSING_CSV_CANDIDATES = [
#     "https://github.com/tjluo-ucas/data-science/blob/main/housing.csv",
#     "datasets/housing/housing.csv",
#     "/Users/abc/handson-ml2/datasets/housing/housing.csv",
# ]


def _find_first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data
def load_california_image():
    """Load background image of California map (if available)."""
    path = _find_first_existing_path(IMAGE_CANDIDATES)
    if path is None:
        return None
    return mpimg.imread(path)


@st.cache_data
def load_housing_data():
    """Load original housing dataset (for background scatter)."""
    path = _find_first_existing_path(HOUSING_CSV_CANDIDATES)
    if path is None:
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class HousingPricePredictor:
    """California housing price predictor."""

    def __init__(
        self,
        model_path: str = "best_housing_model.pkl",
        feature_info_path: str = "feature_info.pkl",
    ):
        self.model_path = model_path
        self.feature_info_path = feature_info_path

        # Categories for the 'ocean_proximity' feature
        self.ocean_options = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]

        # Default feature configuration
        self._set_default_features()

        # Load model (or create a demo model if loading fails)
        self.load_model()

    def _set_default_features(self):
        """Set default feature configuration."""
        self.numerical_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        self.categorical_features = ["ocean_proximity"]
        self.all_features = self.numerical_features + self.categorical_features
        self.feature_info = {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "ocean_proximity_categories": self.ocean_options,
        }

    def load_model(self):
        """Load the model pipeline from disk, or fall back to a demo model."""
        try:
            print("Loading model pipeline...")
            self.pipeline = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")

            # Try to load feature info (optional)
            try:
                info = joblib.load(self.feature_info_path)
                self.numerical_features = info.get(
                    "numerical_features", self.numerical_features
                )
                self.categorical_features = info.get(
                    "categorical_features", self.categorical_features
                )
                self.all_features = self.numerical_features + self.categorical_features
                self.feature_info = info
                print(f"✅ Feature info loaded from {self.feature_info_path}")
            except Exception as e:
                print(
                    f"⚠ Feature info file not found or invalid. "
                    f"Using default features. Detail: {e}"
                )

        except Exception as e:
            print(f"❌ Failed to load model from disk: {e}")
            print("Creating a demo model for illustration...")
            self._create_demo_model()

    def _create_demo_model(self):
        """Create and train a demo model using synthetic data."""
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            print("Building demo model pipeline...")

            # Preprocessing pipeline
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            full_pipeline = ColumnTransformer(
                [
                    ("num", num_pipeline, self.numerical_features),
                    ("cat", cat_pipeline, self.categorical_features),
                ]
            )

            model = RandomForestRegressor(n_estimators=50, random_state=42)

            self.pipeline = Pipeline(
                [
                    ("preprocessor", full_pipeline),
                    ("regressor", model),
                ]
            )

            self._train_demo_model()
            print("✅ Demo model created and trained successfully.")

        except Exception as e:
            print(f"❌ Failed to create demo model: {e}")
            self.pipeline = None

    def _train_demo_model(self):
        """Train the demo model on synthetic data."""
        if self.pipeline is None:
            return

        print("Training demo model with synthetic data...")

        np.random.seed(42)
        n_samples = 500

        X_train = pd.DataFrame(
            {
                "longitude": np.random.uniform(-124, -114, n_samples),
                "latitude": np.random.uniform(32, 42, n_samples),
                "housing_median_age": np.random.randint(1, 52, n_samples),
                "total_rooms": np.random.randint(100, 5000, n_samples),
                "total_bedrooms": np.random.randint(50, 1000, n_samples),
                "population": np.random.randint(200, 3000, n_samples),
                "households": np.random.randint(100, 1000, n_samples),
                "median_income": np.random.uniform(0.5, 15, n_samples),
                "ocean_proximity": np.random.choice(self.ocean_options, n_samples),
            }
        )

        y_train = (
            200_000
            + X_train["median_income"] * 50_000
            + X_train["total_rooms"] * 50
            + (42 - X_train["latitude"]) * 5000
            + np.random.normal(0, 50_000, n_samples)
        )

        self.pipeline.fit(X_train[self.all_features], y_train)

        # Save demo model (optional)
        try:
            joblib.dump(self.pipeline, self.model_path)
            joblib.dump(self.feature_info, self.feature_info_path)
            print(f"✅ Demo model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠ Could not save demo model: {e}")

    def predict_single(self, input_data: dict):
        """Predict price for a single sample (dictionary input)."""
        if self.pipeline is None:
            print("❌ Model is not initialized.")
            return None

        try:
            df = pd.DataFrame([input_data])
            df = df[self.all_features]
            prediction = self.pipeline.predict(df)[0]
            return float(prediction)
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None

    def predict_batch(self, input_df: pd.DataFrame):
        """Predict prices for a batch of samples."""
        if self.pipeline is None:
            print("❌ Model is not initialized.")
            return None

        try:
            input_df = input_df[self.all_features]
            predictions = self.pipeline.predict(input_df)
            return predictions.astype(float)
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            return None

    def generate_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """Generate sample data for demo."""
        np.random.seed(42)

        samples = []
        for _ in range(n_samples):
            sample = {
                "longitude": np.random.uniform(-124, -114),
                "latitude": np.random.uniform(32, 42),
                "housing_median_age": np.random.randint(1, 52),
                "total_rooms": np.random.randint(100, 5000),
                "total_bedrooms": np.random.randint(50, 1000),
                "population": np.random.randint(200, 3000),
                "households": np.random.randint(100, 1000),
                "median_income": np.round(np.random.uniform(0.5, 15), 2),
                "ocean_proximity": np.random.choice(self.ocean_options),
            }
            samples.append(sample)

        return pd.DataFrame(samples)


@st.cache_resource
def get_predictor():
    """Cache the predictor so the model is not reloaded on every rerun."""
    return HousingPricePredictor()


# ---------------------------------------------------------------------
# Plotting: draw predicted house on California map
# ---------------------------------------------------------------------
def plot_house_on_map(
    longitude: float,
    latitude: float,
    price: float | None = None,
    housing_df: pd.DataFrame | None = None,
    background_img=None,
):
    """Create a matplotlib figure with background scatter + predicted house."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Background image
    if background_img is not None:
        ax.imshow(
            background_img,
            extent=[-124.55, -113.80, 32.45, 42.05],  # left, right, bottom, top
            alpha=0.5,
            aspect="auto",
            zorder=0,
        )

    # 2. Background scatter of all houses
    if housing_df is not None:
        scatter = ax.scatter(
            x=housing_df["longitude"],
            y=housing_df["latitude"],
            s=housing_df["population"] / 100,
            c=housing_df["median_house_value"],
            cmap="jet",
            alpha=0.4,
            edgecolors="none",
            zorder=1,
        )

        prices = housing_df["median_house_value"]
        tick_values = np.linspace(prices.min(), prices.max(), 11)
        cbar = fig.colorbar(scatter, ax=ax, ticks=tick_values)
        cbar.ax.set_yticklabels([f"${int(v/1000)}k" for v in tick_values])
        cbar.set_label("Median House Value", fontsize=12)

    # 3. The predicted house
    ax.scatter(
        [longitude],
        [latitude],
        color="red",
        edgecolors="black",
        s=200,
        marker="*",
        label="Predicted house",
        zorder=3,
    )

    label = "Predicted house"
    if price is not None:
        label += f"\n${price:,.0f}"

    ax.annotate(
        label,
        xy=(longitude, latitude),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        zorder=4,
    )

    # 4. Axes and title
    ax.set_xlim(-124.55, -113.80)
    ax.set_ylim(32.45, 42.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("California Housing Prices and Predicted House", fontsize=14)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="California Housing Price Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 California Housing Price Predictor")
    st.markdown(
        "This app predicts median house prices in California using a "
        "**Random Forest** regression model. "
        "If no trained model file is found, a demo model trained on synthetic data will be used."
    )

    predictor = get_predictor()

    if predictor.pipeline is None:
        st.error("Model is not available. Please check your model files.")
        st.stop()

    st.sidebar.header("Prediction Mode")
    mode = st.sidebar.radio(
        "Choose how you want to use the app:",
        ["Single prediction", "Batch demo"],
        index=0,
    )

    if mode == "Single prediction":
        st.header("Single Sample Prediction")

        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                longitude = st.number_input(
                    "Longitude",
                    min_value=-125.0,
                    max_value=-113.0,
                    value=-118.49,
                    format="%.2f",
                )
                latitude = st.number_input(
                    "Latitude",
                    min_value=32.0,
                    max_value=43.0,
                    value=34.26,
                    format="%.2f",
                )
                housing_median_age = st.slider(
                    "Housing median age (years)",
                    min_value=1,
                    max_value=52,
                    value=28,
                )
                total_rooms = st.number_input(
                    "Total rooms",
                    min_value=1,
                    max_value=40_000,
                    value=2_635,
                    step=1,
                )

            with col2:
                total_bedrooms = st.number_input(
                    "Total bedrooms",
                    min_value=1,
                    max_value=10_000,
                    value=537,
                    step=1,
                )
                population = st.number_input(
                    "Population",
                    min_value=1,
                    max_value=40_000,
                    value=1_425,
                    step=1,
                )
                households = st.number_input(
                    "Households",
                    min_value=1,
                    max_value=10_000,
                    value=499,
                    step=1,
                )
                median_income = st.number_input(
                    "Median income (10,000 USD / year)",
                    min_value=0.5,
                    max_value=15.0,
                    value=3.87,
                    step=0.01,
                    format="%.2f",
                )
                ocean_proximity = st.selectbox(
                    "Ocean proximity",
                    options=predictor.ocean_options,
                    index=predictor.ocean_options.index("<1H OCEAN"),
                )

            submitted = st.form_submit_button("Predict price")

        if submitted:
            input_sample = {
                "longitude": longitude,
                "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms,
                "total_bedrooms": total_bedrooms,
                "population": population,
                "households": households,
                "median_income": median_income,
                "ocean_proximity": ocean_proximity,
            }

            prediction = predictor.predict_single(input_sample)

            if prediction is not None:
                st.success(f"Estimated house price: **${prediction:,.2f}**")
                cny = prediction * 7.2  # simple reference conversion
                st.caption(f"Approximate price in CNY: ¥{cny:,.2f} (for reference only)")

                with st.expander("Show input features"):
                    st.json(input_sample)

                # --- NEW: draw the predicted house on the California map ---
                st.subheader("Predicted House Location on California Map")

                background_img = load_california_image()
                housing_df = load_housing_data()

                fig = plot_house_on_map(
                    longitude=longitude,
                    latitude=latitude,
                    price=prediction,
                    housing_df=housing_df,
                    background_img=background_img,
                )
                st.pyplot(fig)
            else:
                st.error("Prediction failed. Please check your inputs or model files.")

    else:
        st.header("Batch Demo with Synthetic Samples")

        n_samples = st.slider(
            "Number of demo samples",
            min_value=1,
            max_value=50,
            value=5,
        )

        if st.button("Generate samples and predict"):
            samples_df = predictor.generate_sample_data(n_samples)
            predictions = predictor.predict_batch(samples_df)

            if predictions is not None:
                result_df = samples_df.copy()
                result_df["predicted_price"] = predictions
                result_df["predicted_price_USD"] = result_df["predicted_price"].map(
                    lambda x: f"${x:,.0f}"
                )

                st.subheader("Prediction Results")
                st.dataframe(result_df, use_container_width=True)

            else:
                st.error("Batch prediction failed. Please check the model.")


if __name__ == "__main__":
    main()
