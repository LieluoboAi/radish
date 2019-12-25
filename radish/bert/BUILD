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
        "//radish/train/data:example_parser",
        "//radish/utils:logging",
        "//third_party:pytorch",
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
        "//radish/bert:bert_tokenizer",
        "//radish/train/data:example_parser",
        "//radish/utils:logging",
        "//radish/utils:sentencepiece_tokenizer",
        "//third_party:pytorch",
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
        "//radish/train:llb_model",
        "//radish/transformer:transformer_encoder",
        "//third_party:pytorch",
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
        "//radish/bert/model:bert_model",
        "//radish/train:llb_model",
        "//third_party:pytorch",
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
        "//radish/train:llb_trainer",
        "//radish/utils:logging",
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
        "//radish/train:llb_trainer",
        "//radish/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "bert_tokenizer",
    srcs = [
        "bert_tokenizer.cc",
        "bert_tokenizer.h",
    ],
    copts = [],
    linkstatic = True,
    deps = [
        "//radish/utils:logging",
        "//radish/utils:text_tokenizer",
        "//radish/utils:basic_string_util",
        "@utfcpp",
        "@utf8proc",
    ],
    alwayslink = True,
)

cc_test(
    name = "bert_tokenizer_test",
    srcs = [
        "bert_tokenizer_test.cc",
    ],
    deps = [
        ":bert_tokenizer",
        "@googletest//:gtest_main",
    ],
)
