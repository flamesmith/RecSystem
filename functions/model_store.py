"""Model store utilities.

Provides a single source of truth for artifact naming and I/O so that the
training script and inference code never hard-code paths in two places.

Usage (training):
    from functions.model_store import save_model
    save_model(my_model, 'ials')          # → models/ials.joblib

Usage (inference):
    from functions.model_store import load_model
    from functions.ials_implicit_package import IALSImplicitRecommender
    model = load_model(IALSImplicitRecommender, 'ials')
"""

import os

import joblib

# Default directory for serialized model artifacts, relative to the project root.
# The project root is resolved as two levels up from this file (functions/ → root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')


def get_model_path(model_name, models_dir=None):
    """Return the full path for a model artifact file.

    Args:
        model_name: Logical name for the model (e.g. 'ials', 'item_knn').
        models_dir: Directory to look in.  Defaults to <project_root>/models/.

    Returns:
        Absolute path string ending in '<model_name>.joblib'.
    """
    models_dir = models_dir or DEFAULT_MODELS_DIR
    return os.path.join(models_dir, f"{model_name}.joblib")


def save_model(model, model_name, models_dir=None):
    """Serialize a trained model to disk.

    Calls model.save(path) so each class can override serialization if needed.
    Falls back to joblib.dump for any object that does not implement save().

    Args:
        model: Trained model instance (must have a save(path) method, or be
               serializable with joblib).
        model_name: Logical name used to derive the file name.
        models_dir: Target directory.  Defaults to <project_root>/models/.
    """
    models_dir = models_dir or DEFAULT_MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    path = get_model_path(model_name, models_dir)

    if hasattr(model, 'save'):
        model.save(path)
    else:
        joblib.dump(model, path)

    print(f"Saved '{model_name}' → {path}")
    return path


def load_model(model_class, model_name, models_dir=None):
    """Load a serialized model from disk.

    Calls model_class.load(path) so each class can override deserialization.
    Falls back to joblib.load for any class that does not implement load().

    Args:
        model_class: The class of the model to load (used for its load()
                     classmethod, e.g. IALSImplicitRecommender).
        model_name: Logical name used to derive the file name.
        models_dir: Source directory.  Defaults to <project_root>/models/.

    Returns:
        A ready-to-use model instance.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    models_dir = models_dir or DEFAULT_MODELS_DIR
    path = get_model_path(model_name, models_dir)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved artifact found for '{model_name}' at {path}. "
            "Run the training script first."
        )

    if hasattr(model_class, 'load'):
        model = model_class.load(path)
    else:
        model = joblib.load(path)

    print(f"Loaded '{model_name}' from {path}")
    return model


def list_saved_models(models_dir=None):
    """List all model artifacts currently saved in models_dir.

    Returns:
        List of logical model names (filename stems without extension).
    """
    models_dir = models_dir or DEFAULT_MODELS_DIR
    if not os.path.isdir(models_dir):
        return []
    return [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(models_dir))
        if f.endswith('.joblib')
    ]
