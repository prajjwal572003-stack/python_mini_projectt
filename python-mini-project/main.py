class Prediction:
    def __init__(self, value, r2_score, slope, intercept, mean_absolute_error):
        self.value = value
        self.r2_score = r2_score
        self.slope = slope
        self.intercept = intercept
        self.mean_absolute_error = mean_absolute_error

    def __str__(self):
        return (
            f"Predicted Value: {self.value}\n"
            f"R2 Score: {self.r2_score}\n"
            f"Slope: {self.slope}\n"
            f"Intercept: {self.intercept}\n"
            f"Mean Absolute Error: {self.mean_absolute_error}"
        )