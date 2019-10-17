package(
    default_visibility = ["//visibility:public"],
)


cc_library(
    name = "bert_options",
    srcs = [
        "bert_options.h",
    ],
    hdrs = [
        "bert_options.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)


cc_library(
    name = "bert_embedding",
    srcs = [
        "bert_embedding.cc",
    ],
    hdrs = [
        "bert_embedding.h",
    ],
    deps = [
        ":bert_options",
        "//radish/layers:layer_norm",
        "//radish/layers:embedding_layer",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)



cc_library(
    name = "bert_attention",
    srcs = [
        "bert_attention.cc",
    ],
    hdrs = [
        "bert_attention.h",
    ],
    deps = [
        ":bert_options",
        "//radish/layers:layer_norm",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "bert_layer",
    srcs = [
        "bert_layer.cc",
    ],
    hdrs = [
        "bert_layer.h",
    ],
    deps = [
        ":bert_options",
        ":bert_attention",
        "//radish/layers:layer_norm",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "bert_encoder",
    srcs = [
        "bert_encoder.cc",
    ],
    hdrs = [
        "bert_encoder.h",
    ],
    deps = [
        ":bert_options",
        ":bert_layer",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "bert_model",
    srcs = [
        "bert_model.cc",
    ],
    hdrs = [
        "bert_model.h",
    ],
    deps = [
        ":bert_options",
        ":bert_encoder",
        ":bert_embedding",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)