import pandas as pd
import warnings

warnings.filterwarnings("ignore")

latex = 0
pand = 1
train_time = 0
test_time = 0

embeddings_order = [
    "bundle",
    "sequence",
    "ngram",
    "hashmap",
    "density",
    "flocet",
    "generic",
    "random",
    "sinusoid",
    "fractional",
]
embeddings_order = [0, 7, 5, 4, 1, 2, 3, 6, 8]
# embeddings_order = [0, 1, 2, 3, 4, 5, 6, 7]

df = pd.read_csv(
    "/Users/verges/Documents/PhD/TorchHd/torchhd/HDCArena/results/proves/encodings_arena"
)

for i in df["encoding"].unique():
    variance_accuracy_by_dimension_and_method = (
        df[df["encoding"] == i].groupby(["name", "method"])["accuracy"].std()
    )
    # print(variance_accuracy_by_dimension_and_method.mean())


mean_of_encoding = (
    df.groupby(["encoding"])["accuracy"].mean().round(3).reset_index().T
)[embeddings_order]

var_of_encoding = (
    df.groupby(["encoding", "name", "method"])["accuracy"]
    .std()
    .groupby("encoding")
    .mean()
    .reset_index()
    .T
)[embeddings_order]

mean_of_encoding_train_time = (
    df.groupby(["encoding"])["train_time"].mean().round(3).reset_index().T
)[embeddings_order]

mean_of_encoding_test_time = (
    df.groupby(["encoding"])["test_time"].mean().round(3).reset_index().T
)[embeddings_order]


var_of_encoding_train_time = (
    df.groupby(["encoding", "name", "method"])["train_time"]
    .std()
    .groupby("encoding")
    .mean()
    .reset_index()
    .T
)[embeddings_order]


mean_of_encoding_train_time = df.groupby(["encoding"])["train_time"]
mean_of_encoding_train_time = mean_of_encoding_train_time.agg(["mean"]).round(3).T

var_of_encoding_train_time = df.groupby(["encoding"])["train_time"]
var_of_encoding_train_time = var_of_encoding_train_time.agg(["std"]).round(3).T

if pand:
    print(mean_of_encoding)
    print(var_of_encoding)
    if train_time:
        print(mean_of_encoding_train_time)
    if test_time:
        print(mean_of_encoding_test_time)

if latex:
    latex_table = mean_of_encoding.to_latex(
        index=False, caption="Encodings accuracy mean"
    )
    print(latex_table)
    # pd.options.display.float_format = "{:.2e}".format
    latex_table = var_of_encoding.to_latex(
        index=False, caption="Encodings accuracy variance"
    )
    print(latex_table)
    pd.options.display.float_format = None

    if train_time:
        latex_table = mean_of_encoding_train_time.to_latex(
            index=False, caption="Encodings train time mean"
        )
        print(latex_table)

        latex_table = var_of_encoding_train_time.to_latex(
            index=False, caption="Encodings train time var"
        )
        print(latex_table)

    if test_time:
        latex_table = mean_of_encoding_test_time.to_latex(
            index=False, caption="Encodings test time mean"
        )
        print(latex_table)

var_of_encoding = (
    df.groupby(["encoding", "method", "dimensions", "name"])["accuracy"]
    .std()
    .round(3)
    .T
)
embeddings_order = [
    "bundle",
    "sequence",
    "ngram",
    "hashmap",
    "density",
    "flocet",
    "generic",
    "random",
    "sinusoid",
    "fractional",
]

print(mean_of_encoding_train_time)
print(var_of_encoding_train_time)
# for i in embeddings_order:
#    print(var_of_encoding_train_time[i].mean().round(3),end=",")

# for i in embeddings_order:
#    print(mean_of_encoding[i].mean().round(3),end=",")
