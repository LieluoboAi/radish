

cc_library(
    name = "query_same_parser",
    srcs = [
        "query_same_parser.cc",
    ],
    hdrs = [
        "query_same_parser.h",
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
    name = "xnli_example_parser",
    srcs = [
        "xnli_example_parser.cc",
    ],
    hdrs = [
        "xnli_example_parser.h",
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
    name = "query_same_model",
    srcs = [
        "query_same_model.cc",
    ],
    hdrs = [
        "query_same_model.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train:llb_model",
        "//radish/transformer:transformer_encoder",
    ],
)

cc_binary(
    name = "train_qs_main",
    srcs = [
        "train_qs_main.cc",
    ],
    copts = [],
    deps = [
        ":query_same_parser",
        ":xnli_example_parser",
        ":query_same_model",
        "//radish/utils:logging",
        "//radish/train:llb_trainer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
