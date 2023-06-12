import time
import csv
import sys

sys.path.append("methods")
from methods import vanillaHD
from methods import adaptHD
from methods import onlineHD
from methods import adjustHD
from methods import adjustSemiHD
from methods import compHD
from methods import multiCentroidHD
from methods import adaptHDIterative
from methods import onlineHDIterative
from methods import adjustHDIterative
from methods import quantHDIterative
from methods import sparseHDIterative
from methods import neuralHDIterative
from methods import distHDIterative
from methods import intRVFL
from methods import semiHDIterative
from methods import LeHDC

results_file = "adjust/results" + str(time.time()) + ".csv"

with open(results_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "name",
            "accuracy",
            "train_time",
            "test_time",
            "dimensions",
            "method",
            "encoding",
            "iterations",
            "lr",
            "chunks",
            "threshold",
            "reduce_subclass",
            "model_quantize",
            "epsilon",
            "alpha",
            "beta",
            "theta",
            "r",
            "lazy_regeneration",
            "model_neural",
            "lazy_regeneration",
            "model_neural",
            "partial_data",
            "robustness_failed_dimensions",
            "weight_decay",
            "learning_rate",
            "dropout_rate",
        ]
    )


def select_model(
    train_ds,
    test_ds,
    train_loader,
    test_loader,
    num_classes,
    num_feat,
    encode,
    model,
    device,
    dataset,
    method="add",
    encoding="density",
    iterations=10,
    dimensions=10000,
    lr=1,
    chunks=10,
    threshold=0.03,
    reduce_subclasses="drop",
    model_quantize="binary",
    lazy_regeneration=5,
    model_neural="reset",
    epsilon=0.01,
    alpha=4,
    beta=2,
    theta=1,
    r=0.05,
    model_sparse="class",
    s=0.5,
    partial_data=False,
    robustness=[],
    weight_decay=0,
    learning_rate=0,
    dropout_rate=0,
):
    if method == "add":
        vanillaHD.train_vanillaHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "adjust":
        adjustHD.train_adjustHD(
            train_ds,
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "adjust_semi":
        adjustSemiHD.train_adjustSemiHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "adapt":
        adaptHD.train_adaptHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "online":
        onlineHD.train_onlineHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "comp":
        compHD.train_compHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "multicentroid":
        multiCentroidHD.train_multicentroidHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "adapt_iterative":
        adaptHDIterative.train_adaptHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "online_iterative":
        onlineHDIterative.train_onlineHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "adjust_iterative":
        adjustHDIterative.train_adjustHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "quant_iterative":
        quantHDIterative.train_quantHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "sparse_iterative":
        sparseHDIterative.train_sparseHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "neural_iterative":
        neuralHDIterative.train_neuralHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "dist_iterative":
        distHDIterative.train_distHD(
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "rvfl":
        intRVFL.train_rvfl(
            dataset,
            train_ds,
            test_ds,
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )
    elif method == "semi":
        semiHDIterative.train_semiHD(
            train_ds,
            test_ds,
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )

    elif method == "lehdc":
        LeHDC.train_LeHDC(
            train_ds,
            test_ds,
            train_loader,
            test_loader,
            num_classes,
            encode,
            model,
            device,
            dataset.name,
            method,
            encoding,
            iterations,
            dimensions,
            lr,
            chunks,
            threshold,
            reduce_subclasses,
            model_quantize,
            epsilon,
            model_sparse,
            s,
            alpha,
            beta,
            theta,
            r,
            partial_data,
            robustness,
            lazy_regeneration,
            model_neural,
            weight_decay,
            learning_rate,
            dropout_rate,
            results_file,
        )


configs = [
    {
        "method": "add",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "adapt",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "online",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "adjust",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "adapt_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 5,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "online_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 5,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "adjust_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 5,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "quant_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1.5,
        "epsilon": 0.01,
        "model_quantize": "binary",
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "sparse_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": 0.01,
        "model_quantize": None,
        "model_sparse": "class",
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": 0.5,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "comp",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": 2,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "neural_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": "class",
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": "reset",
        "r": 0.1,
        "alpha": 4,
        "beta": 2,
        "theta": 1,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "dist_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": "class",
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": 0.05,
        "alpha": 4,
        "beta": 2,
        "theta": 1,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "multicentroid",
        "multi_reduce_subclass": "drop",
        "threshold": 0.03,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "rvfl",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "semi",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": 0.05,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
    {
        "method": "lehdc",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": 5e-2,
        "learning_rate": 1e-2,
        "dropout_rate": 0.5,
    },
    {
        "method": "adjust_semi",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": None,
        "model_quantize": None,
        "model_sparse": None,
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": None,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
]

configs = [
    {
        "method": "sparse_iterative",
        "multi_reduce_subclass": None,
        "threshold": None,
        "lr": 1,
        "epsilon": 0.01,
        "model_quantize": None,
        "model_sparse": "class",
        "sparsity": None,
        "lazy_regeneration": None,
        "model_neural": None,
        "r": None,
        "alpha": None,
        "beta": None,
        "theta": None,
        "chunks": None,
        "s": 0.5,
        "weight_decay": None,
        "learning_rate": None,
        "dropout_rate": None,
    },
]

# METHODS = ["add","adapt","online","adjust","comp","adapt_iterative","online_iterative","adjust_iterative",
# "quant_iterative","sparse_iterative","neural_iterative","dist_iterative","multicentroid","rvfl"]
