"""Train a tiny classifier on fixed-point features."""

import numpy as np

from anderson_acceleration import (
    equilibrium_features,
    fit_softmax_readout,
    make_equilibrium_weights,
    make_two_moons,
    readout_accuracy,
)


def main() -> None:
    inputs, labels = make_two_moons(n_samples=180, noise=0.08, seed=11)
    train_x, test_x = inputs[:120], inputs[120:]
    train_y, test_y = labels[:120], labels[120:]

    raw_readout = fit_softmax_readout(
        train_x,
        train_y,
        learning_rate=0.15,
        epochs=300,
        weight_decay=1e-3,
        seed=2,
    )

    weights = make_equilibrium_weights(
        input_dim=2,
        hidden_dim=32,
        recurrent_scale=0.72,
        seed=3,
    )
    train_features = equilibrium_features(train_x, weights, memory=5)
    test_features = equilibrium_features(test_x, weights, memory=5)
    eq_readout = fit_softmax_readout(
        train_features.hidden_states,
        train_y,
        learning_rate=0.18,
        epochs=400,
        weight_decay=5e-4,
        seed=4,
    )

    print("Two-moons classifier")
    print(f"raw train accuracy:         {raw_readout.train_accuracy:.3f}")
    print(f"raw test accuracy:          {readout_accuracy(test_x, test_y, raw_readout.weights, raw_readout.bias):.3f}")
    print(f"implicit train accuracy:    {eq_readout.train_accuracy:.3f}")
    print(
        "implicit test accuracy:     "
        f"{readout_accuracy(test_features.hidden_states, test_y, eq_readout.weights, eq_readout.bias):.3f}"
    )
    print(f"feature convergence rate:   {test_features.convergence_rate:.3f}")
    print(f"mean solver iterations:     {np.mean(test_features.iterations):.1f}")
    print(f"max residual:               {max(test_features.residuals):.2e}")


if __name__ == "__main__":
    main()
