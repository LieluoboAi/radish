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
        "//train/data:example_parser",
       "//utils:logging",
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
        "//train/data:example_parser",
       "//utils:logging",
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
        "//train:llb_model",
        "//transformer:transformer_encoder",
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
        "//utils:logging",
        "//train:llb_trainer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
