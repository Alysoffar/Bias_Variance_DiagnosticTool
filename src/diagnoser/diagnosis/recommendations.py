
def recommendation(diagnosis: str) -> str:
    """Get recommendation based on diagnosis label."""
    recommendations = {
        "high_bias": "Consider using a more complex model or adding more features.",
        "high_variance": "Consider using regularization, gathering more training data, or simplifying the model.",
        "balanced": "Your model is well-balanced. Continue monitoring performance.",
        "insufficient_data": "Not enough data to make a diagnosis. Collect more training examples."
    }
    return recommendations.get(diagnosis, "No recommendation available.")