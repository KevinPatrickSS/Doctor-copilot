"""
SHAP Explainability Module
Provides interpretable explanations for ML predictions
"""

import numpy as np
import shap
import pickle


class SHAPExplainer:
    """
    SHAP-based explainer for cardiac risk models.
    Provides both local (per-patient) and global explanations.
    """

    def __init__(self, model, model_type='tree', feature_names=None):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer = None
        self.base_value = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_expected_value(self, ev):
        """Handle scalar / list / ndarray expected_value safely."""
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[0])
        return float(ev)

    def _normalize_shap_values(self, shap_values):
        """
        Normalize SHAP output across versions and model types.
        """
        # New SHAP versions → ndarray directly
        if isinstance(shap_values, np.ndarray):
            return shap_values

        # Older SHAP → list of arrays (class-wise)
        if isinstance(shap_values, list):
            # Binary classification → positive class
            return shap_values[-1]

        raise ValueError("Unsupported SHAP value format")

    # ------------------------------------------------------------------
    # Initializer
    # ------------------------------------------------------------------
    def initialize_explainer(self, X_background=None):
        print(f"Initializing SHAP {self.model_type} explainer...")

        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
            print("TreeExplainer initialized")

        elif self.model_type == 'linear':
            if X_background is None:
                raise ValueError("X_background required for LinearExplainer")
            self.explainer = shap.LinearExplainer(self.model, X_background)
            print("LinearExplainer initialized")

        else:
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")

            background = (
                shap.sample(X_background, 100)
                if len(X_background) > 100
                else X_background
            )

            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
            print("KernelExplainer initialized")

        # Base value (population baseline risk)
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self._normalize_expected_value(
                self.explainer.expected_value
            )

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------
    def explain_prediction(self, X_patient, top_n=5):
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        if X_patient.ndim == 1:
            X_patient = X_patient.reshape(1, -1)

        shap_values = self._normalize_shap_values(
            self.explainer.shap_values(X_patient)
        )

        patient_shap = shap_values[0]

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"Feature_{i}" for i in range(len(patient_shap))]
        )

        feature_impact = {}
        for i, name in enumerate(feature_names):
            feature_impact[name] = {
                "shap_value": float(patient_shap[i]),
                "feature_value": float(X_patient[0, i]),
                "abs_shap": abs(float(patient_shap[i]))
            }

        sorted_features = sorted(
            feature_impact.items(),
            key=lambda x: x[1]["abs_shap"],
            reverse=True
        )

        top_features = sorted_features[:top_n]

        positive = [(n, d) for n, d in top_features if d["shap_value"] > 0]
        negative = [(n, d) for n, d in top_features if d["shap_value"] < 0]

        explanation_text = self._generate_explanation_text(
            positive, negative
        )

        return {
            "top_features": dict(top_features),
            "positive_contributors": dict(positive),
            "negative_contributors": dict(negative),
            "explanation_text": explanation_text,
            "base_value": self.base_value,
            "shap_values_array": patient_shap.tolist()
        }

    # ------------------------------------------------------------------
    # Global explanation
    # ------------------------------------------------------------------
    def get_global_feature_importance(self, X_data, max_samples=1000):
        if self.explainer is None:
            raise ValueError("Explainer not initialized")

        if len(X_data) > max_samples:
            idx = np.random.choice(len(X_data), max_samples, replace=False)
            X_data = X_data[idx]

        print(f"Computing global SHAP values for {len(X_data)} samples...")

        shap_values = self._normalize_shap_values(
            self.explainer.shap_values(X_data)
        )

        mean_abs = np.abs(shap_values).mean(axis=0)

        feature_names = (
            self.feature_names
            if self.feature_names is not None
            else [f"Feature_{i}" for i in range(len(mean_abs))]
        )

        importance = dict(
            sorted(
                zip(feature_names, mean_abs),
                key=lambda x: x[1],
                reverse=True
            )
        )

        print("\nTop 10 Most Important Features:")
        for i, (f, v) in enumerate(list(importance.items())[:10], 1):
            print(f"  {i}. {self._make_readable_feature_name(f)}: {v:.4f}")

        return importance

    # ------------------------------------------------------------------
    # Explanation text
    # ------------------------------------------------------------------
    def _generate_explanation_text(self, positive, negative):
        lines = []

        if positive:
            lines.append("Factors INCREASING risk:")
            for i, (f, d) in enumerate(positive[:3], 1):
                lines.append(
                    f"  {i}. {self._make_readable_feature_name(f)} "
                    f"(value: {d['feature_value']:.2f})"
                )

        if negative:
            lines.append("\nFactors DECREASING risk:")
            for i, (f, d) in enumerate(negative[:3], 1):
                lines.append(
                    f"  {i}. {self._make_readable_feature_name(f)} "
                    f"(value: {d['feature_value']:.2f})"
                )

        if not lines:
            lines.append("No dominant risk factors identified.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Human-readable names
    # ------------------------------------------------------------------
    def _make_readable_feature_name(self, feature_name):
        mapping = {
            "age": "Patient age",
            "is_male": "Male sex",
            "has_angina": "History of angina",
            "exang_binary": "Exercise-induced angina",
            "lv_hypertrophy": "Left ventricular hypertrophy",
            "trestbps": "Resting blood pressure",
            "thalch": "Maximum heart rate",
            "oldpeak": "ST depression",
            "chol": "Cholesterol level",
            "fbs_binary": "Fasting blood sugar",
            "age_over_65": "Age over 65",
            "high_cholesterol": "High cholesterol",
            "hypertension": "Hypertension",
            "reduced_exercise_capacity": "Reduced exercise capacity",
            "significant_st_depression": "Significant ST depression"
        }
        return mapping.get(feature_name, feature_name.replace("_", " ").title())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_explainer(self, filepath="shap_explainer.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "explainer": self.explainer,
                    "model_type": self.model_type,
                    "feature_names": self.feature_names,
                    "base_value": self.base_value,
                },
                f,
            )
        print(f"SHAP explainer saved to {filepath}")


if __name__ == "__main__":
    print("SHAP Explainability Module")
    print("="*60)
    print("\nThis module provides:")
    print("1. Patient-level explanations (why this patient is high/low risk)")
    print("2. Global feature importance (which features matter most overall)")
    print("3. Human-readable explanation text")