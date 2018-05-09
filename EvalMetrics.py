

# Mean absolute Percentage Error
def mape(predictions, y_eval):
    errors = abs(y_eval - predictions)
    ape = errors / abs(y_eval) * 100
    mape = sum(ape) / len(y_eval)

    return mape[0]


# Mean Squared Error
def mse(predictions, y_eval):
    errors = (predictions - y_eval) ** 2
    mse = sum(errors) / len(y_eval)

    return mse[0]


# Mean Error Relative
def mer(predictions, y_eval):
    errors = abs(y_eval - predictions)
    ape = errors / (sum(y_eval) / len(y_eval)) * 100
    mer = sum(ape) / len(y_eval)

    return mer[0]


# Percentage Error
def pe(predictions, y_eval):
    errors = y_eval - predictions
    pe = errors / y_eval * 100

    return pe.reshape(-1)
