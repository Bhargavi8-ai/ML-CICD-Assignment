
from model import train_model


def test_model_trains_successfully():
    model, X_test, y_test, accuracy = train_model()

    assert model is not None
    assert len(X_test) == len(y_test)
    assert len(y_test) > 0


def test_predictions_have_valid_classes():
    model, X_test, _, _ = train_model()

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1, 2})


def test_model_accuracy():
    _, _, _, accuracy = train_model()

    assert 0.80 <= accuracy <= 1.0
