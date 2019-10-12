package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "span_bert_example_parser",
    srcs = [
        "span_bert_example_parser.cc",
    ],
    hdrs = [
        "span_bert_example_parser.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train/data:example_parser",
       "//radish/utils:logging",
        "@sentencepiece//sentencepiece:error",
        "@sentencepiece//sentencepiece:sentencepiece_processor",
    ],
)


cc_library(
    name = "albert_example_parser",
    srcs = [
        "albert_example_parser.cc",
    ],
    hdrs = [
        "albert_example_parser.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train/data:example_parser",
       "//radish/utils:logging",
        "@sentencepiece//sentencepiece:error",
        "@sentencepiece//sentencepiece:sentencepiece_processor",
    ],
)

cc_library(
    name = "span_bert_model",
    srcs = [
        "span_bert_model.cc",
    ],
    hdrs = [
        "span_bert_model.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train:llb_model",
        "//radish/transformer:transformer_encoder",
    ],
)

cc_library(
    name = "albert_model",
    srcs = [
        "albert_model.cc",
    ],
    hdrs = [
        "albert_model.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train:llb_model",
        "//radish/transformer:transformer_encoder",
    ],
)

cc_binary(
    name = "train_span_bert_main",
    srcs = [
        "train_span_bert_main.cc",
    ],
    copts = [],
    deps = [
        ":span_bert_example_parser",
        ":span_bert_model",
        "//radish/utils:logging",
        "//radish/train:llb_trainer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)


cc_binary(
    name = "train_albert_main",
    srcs = [
        "train_albert_main.cc",
    ],
    copts = [],
    deps = [
        ":albert_example_parser",
        ":albert_model",
        "//radish/utils:logging",
        "//radish/train:llb_trainer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)