cc_library(
    name = "query_same_parser",
    srcs = [
        "query_same_parser.cc",
    ],
    hdrs = [
        "query_same_parser.h",
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
    name = "xnli_example_parser",
    srcs = [
        "xnli_example_parser.cc",
    ],
    hdrs = [
        "xnli_example_parser.h",
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
    name = "query_same_model",
    srcs = [
        "query_same_model.cc",
    ],
    hdrs = [
        "query_same_model.h",
    ],
    deps = [
        "//radish/train:llb_model",
        "//radish/transformer:transformer_encoder",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "bert_classification_model",
    srcs = [
        "bert_classification_model.cc",
    ],
    hdrs = [
        "bert_classification_model.h",
    ],
    deps = [
        "//radish/bert/model:bert_model",
        "//radish/train:llb_model",
        "//radish/train:model_io",
        "//third_party:pytorch",
    ],
)

cc_binary(
    name = "train_qs_main",
    srcs = [
        "train_qs_main.cc",
    ],
    copts = [],
    deps = [
        ":query_same_model",
        ":query_same_parser",
        ":xnli_example_parser",
        "//radish/train:llb_trainer",
        "//radish/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "train_bert_cls_finetune",
    srcs = [
        "train_bert_cls_finetune.cc",
    ],
    copts = [],
    deps = [
        ":bert_classification_model",
        ":xnli_example_parser",
        "//radish/train:llb_trainer",
        "//radish/utils:logging",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
